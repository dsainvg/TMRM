import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx

from utils.decoder_layer import DecoderLayer
from utils.config.decode import (
    DECODER_MAX_PARENTS,
    FANOUT_FIRST_MU, FANOUT_FIRST_SIGMA, FANOUT_FIRST_LO, FANOUT_FIRST_HI,
    FANOUT_SECOND_MU, FANOUT_SECOND_SIGMA, FANOUT_SECOND_LO, FANOUT_SECOND_HI,
)


# ── Wiring helpers ─────────────────────────────────────────────────────────────

def _sample_fanouts(
    n: int,
    mu: float,
    sigma: float,
    lo: int,
    hi: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample n fan-out values from a clipped Gaussian (int32)."""
    raw = rng.normal(loc=mu, scale=sigma, size=n)
    return np.clip(np.round(raw).astype(np.int32), lo, hi)


def _straight_wire(
    n_inputs: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, int]:
    """
    First-layer wiring: K = ceil(n_inputs / 16) nodes.

    Every encoder output index appears at least once across all K×16 input
    slots.  Extra slots (K*16 - n_inputs) repeat randomly from the pool.

    Returns
    -------
    parent_indices : (K, 16) int32
    K              : int
    """
    K = int(np.ceil(n_inputs / DECODER_MAX_PARENTS))
    total_slots = K * DECODER_MAX_PARENTS
    repeats = int(np.ceil(total_slots / n_inputs))
    pool = np.tile(np.arange(n_inputs, dtype=np.int32), repeats)[:total_slots]
    rng.shuffle(pool)
    return pool.reshape(K, DECODER_MAX_PARENTS), K


def _gaussian_wire(
    n_prev: int,
    fanouts: np.ndarray,        # (n_prev,) int — tickets per previous node
    k_nodes: int,               # how many decoder nodes to wire
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a shuffled fan-out pool and cut it into k_nodes chunks of 16.

    Returns
    -------
    parent_indices   : (k_nodes, 16) int32
    unconnected_mask : (n_prev,) bool — True if index absent from used pool
    """
    pool = np.repeat(np.arange(n_prev, dtype=np.int32), fanouts)
    rng.shuffle(pool)
    used = pool[: k_nodes * DECODER_MAX_PARENTS]
    parent_indices = used.reshape(k_nodes, DECODER_MAX_PARENTS)
    used_set = set(used.tolist())
    unconnected_mask = np.array(
        [i not in used_set for i in range(n_prev)], dtype=bool
    )
    return parent_indices, unconnected_mask


def _terminal_wire(
    n_prev: int,
    k_budget: int,              # max decoder nodes allowed by node budget
    rng: np.random.Generator,
) -> tuple[np.ndarray, int, np.ndarray]:
    """
    Terminal wiring used when max_nodes is hit.

    Each previous-layer node index appears in **at most one** input slot across
    all terminal decoder nodes — no duplication.  Previous nodes that cannot
    be assigned a slot become output nodes of their own (previous) layer.

    Parameters
    ----------
    n_prev   : number of previous-layer nodes
    k_budget : maximum decoder nodes permitted by the remaining node budget
    rng      : NumPy RNG

    Returns
    -------
    parent_indices  : (k_terminal, 16) int32 — wiring for terminal layer
                      (k_terminal may be < k_budget if n_prev is the bottleneck)
    k_terminal      : actual number of terminal decoder nodes built
    leftover_mask   : (n_prev,) bool — True for previous nodes not consumed
                      (these become output nodes of the previous layer)

    Edge cases
    ----------
    n_prev < 16          → k_terminal = 0; all n_prev previous nodes are leftover
    n_prev // 16 < k_budget → unique-wiring capacity is the binding constraint
    n_prev >= k_budget*16 → budget is the binding constraint
    """
    # How many full decoders can be uniquely wired from n_prev nodes?
    k_from_prev = n_prev // DECODER_MAX_PARENTS   # each decoder needs 16 unique inputs
    k_terminal  = min(k_budget, k_from_prev)      # binding constraint wins

    # Shuffle all previous indices — each appears exactly once
    shuffled = np.arange(n_prev, dtype=np.int32)
    rng.shuffle(shuffled)

    if k_terminal == 0:
        # Cannot build any decoder (n_prev < 16) — all previous nodes are leftover
        return np.empty((0, DECODER_MAX_PARENTS), dtype=np.int32), 0, np.ones(n_prev, dtype=bool)

    used_count    = k_terminal * DECODER_MAX_PARENTS
    used          = shuffled[:used_count]
    leftover_idxs = shuffled[used_count:]           # indices not consumed
    parent_indices = used.reshape(k_terminal, DECODER_MAX_PARENTS)

    leftover_mask = np.zeros(n_prev, dtype=bool)
    leftover_mask[leftover_idxs] = True

    return parent_indices, k_terminal, leftover_mask


# ── DecoderCluster ─────────────────────────────────────────────────────────────

class DecoderCluster(eqx.Module):
    """
    A vertically stacked sequence of DecoderLayers whose wiring obeys the
    Gaussian fan-out constraints from the architecture spec.

    Parameters
    ----------
    n_layers  : Target number of hidden decoder layers (L).  One extra (L+1-th)
                layer is appended after L complete, unless max_nodes is hit first.
    max_nodes : Hard cap on total decoder nodes across all layers (N).
    n_inputs  : Number of (n, n) encoder output matrices fed to the cluster.
                Typically 64 × n_modalities.
    key       : JAX PRNG key.

    Wiring Algorithm
    ----------------
    Layer 0  — straight wiring:
        K0 = ceil(n_inputs / 16).  The n_inputs encoder outputs are tiled into
        K0×16 slots (shuffled); every encoder output appears at least once.

    Layers 1 … L+1  — Gaussian wiring:
        For each of the M = (previous layer node count) nodes, sample a fan-out
        f_i from a clipped Gaussian.  Build pool = repeat(node_index, f_i),
        shuffle, trim to floor(total/16)×16, then cut into K = trimmed/16
        chunks of 16 consecutive entries.

        First half  (layer_idx < midpoint): μ=18, σ=3, clipped [8,  24].
        Second half (layer_idx ≥ midpoint): μ=12, σ=3, clipped [6,  20].

        Nodes from the previous layer that have zero tickets in the used
        portion of the pool are "unconnected" and are marked as output nodes
        of that layer.

    Stopping Conditions
    -------------------
    max_nodes hit first:
        Detected when the Gaussian fan-out pool would produce more nodes than
        the remaining budget.  The terminal layer uses ``_terminal_wire``
        instead of ``_gaussian_wire``:
          - Previous-layer indices are shuffled once; each appears AT MOST ONCE
            across all terminal decoder input slots (no duplicates).
          - k_terminal = min(budget, n_prev // 16)  — unique-wiring capacity
            and budget are both hard constraints.
          - Previous-layer nodes not consumed → output nodes of their layer.
          - All k_terminal terminal nodes → output nodes.
          - Edge case n_prev < 16: k_terminal = 0; no decoder layer is created,
            all previous nodes become output nodes directly.

    n_layers+1 layers built (L target + 1 extra) without hitting max_nodes:
        The extra layer is terminal; all its nodes are output nodes.

    Forward Pass
    ------------
    encoder_out   : (n_inputs, n, n)
    encoder_flags : (n_inputs,) bool

    Returns
    -------
    out   : (total_output_nodes, n, n)
    flags : (total_output_nodes,) bool
    """

    layers: list             # List[DecoderLayer]
    output_node_indices: list  # List[np.ndarray]  — per-layer output node indices

    def __init__(
        self,
        n_layers: int,
        max_nodes: int,
        n_inputs: int,
        key: jax.Array,
    ):
        # Convert a JAX key to a NumPy seed for deterministic NumPy RNG
        seed = int(jax.random.randint(key, (), 0, 2**31 - 1))
        rng = np.random.default_rng(seed)

        layers_built: list = []
        # output_slots[l] is None until this layer's output nodes are determined
        output_slots: list = []
        total_nodes = 0

        # ── Layer 0: straight wiring from encoder outputs ──────────────────────
        pi0, K0 = _straight_wire(n_inputs, rng)
        key, lkey = jax.random.split(key)
        layers_built.append(DecoderLayer(pi0, lkey))
        output_slots.append(None)
        total_nodes += K0
        n_prev = K0

        # The total number of layers to attempt building is n_layers+1 (target)
        # + the 1 extra = layers at indices 1 … n_layers+1 (inclusive).
        # Midpoint splits first-half / second-half Gaussian params.
        # Total span: layer indices 0 … n_layers+1  →  n_layers+2 layers.
        midpoint = (n_layers + 2) // 2

        # ── Layers 1 … n_layers+1 ──────────────────────────────────────────────
        for layer_idx in range(1, n_layers + 2):

            # ── Budget check ───────────────────────────────────────────────────
            nodes_remaining = max_nodes - total_nodes
            if nodes_remaining <= 0:
                # Previous layer consumed the exact budget — it is already terminal
                break

            # ── Gaussian fan-out params ────────────────────────────────────────
            if layer_idx >= midpoint:
                mu, sigma, lo, hi = (
                    FANOUT_SECOND_MU, FANOUT_SECOND_SIGMA,
                    FANOUT_SECOND_LO, FANOUT_SECOND_HI,
                )
            else:
                mu, sigma, lo, hi = (
                    FANOUT_FIRST_MU, FANOUT_FIRST_SIGMA,
                    FANOUT_FIRST_LO, FANOUT_FIRST_HI,
                )

            fanouts = _sample_fanouts(n_prev, mu, sigma, lo, hi, rng)

            # Guard: degenerate pool (tiny n_prev or very unlucky Gaussian draw)
            pool_total = int(fanouts.sum())
            if pool_total < DECODER_MAX_PARENTS:
                min_f = max(lo, int(np.ceil(DECODER_MAX_PARENTS / n_prev)))
                fanouts = np.maximum(fanouts, min_f)
                pool_total = int(fanouts.sum())

            max_from_pool = pool_total // DECODER_MAX_PARENTS

            # ── Stopping condition: max_nodes hit ──────────────────────────────
            # Budget is the binding constraint — switch to terminal unique wiring.
            # Each previous-layer node feeds at most one input slot (no duplicates).
            if nodes_remaining < max_from_pool:
                pi, k_terminal, leftover_mask = _terminal_wire(n_prev, nodes_remaining, rng)

                # Previous-layer nodes not consumed → output nodes of that layer
                leftover_idxs = np.where(leftover_mask)[0].astype(np.int32)
                if leftover_idxs.size > 0:
                    prev = output_slots[-1]
                    output_slots[-1] = (
                        leftover_idxs if prev is None
                        else np.union1d(prev, leftover_idxs).astype(np.int32)
                    )

                if k_terminal == 0:
                    # n_prev < 16 — cannot build any decoder at all.
                    # All previous nodes already marked above; nothing more to do.
                    break

                key, lkey = jax.random.split(key)
                layers_built.append(DecoderLayer(pi, lkey))
                # All terminal nodes are output nodes
                output_slots.append(np.arange(k_terminal, dtype=np.int32))
                total_nodes += k_terminal
                break

            # ── Normal Gaussian wiring (budget not yet exhausted) ──────────────
            k_this = max_from_pool
            pi, unconnected_mask = _gaussian_wire(n_prev, fanouts, k_this, rng)
            key, lkey = jax.random.split(key)
            layers_built.append(DecoderLayer(pi, lkey))
            output_slots.append(None)

            # Previous-layer nodes absent from used pool → output nodes
            unconn = np.where(unconnected_mask)[0].astype(np.int32)
            if unconn.size > 0:
                prev = output_slots[-2]
                output_slots[-2] = (
                    unconn if prev is None
                    else np.union1d(prev, unconn).astype(np.int32)
                )

            total_nodes += k_this
            n_prev = k_this

            # ── Stopping condition: extra layer just completed ─────────────────
            if layer_idx == n_layers + 1:
                output_slots[-1] = np.arange(k_this, dtype=np.int32)
                break

        # ── Safety net: last layer must always be marked terminal ──────────────
        if output_slots and output_slots[-1] is None:
            last_k = layers_built[-1].parent_indices.shape[0]
            output_slots[-1] = np.arange(last_k, dtype=np.int32)

        # ── Fill remaining None slots (intermediate layers, all nodes consumed) ─
        for i in range(len(output_slots)):
            if output_slots[i] is None:
                output_slots[i] = np.array([], dtype=np.int32)

        self.layers = layers_built
        self.output_node_indices = output_slots

    @property
    def n_output_nodes(self) -> int:
        """Total number of output nodes across all layers."""
        return sum(idx.size for idx in self.output_node_indices)

    def __call__(
        self,
        encoder_out: jax.Array,    # (n_inputs, n, n)
        encoder_flags: jax.Array,  # (n_inputs,) bool
    ):
        """
        Forward pass through all layers, then gather output nodes.

        The Python ``for`` loop over ``self.layers`` unrolls at XLA trace time
        (fixed list length), so shapes are static and compilation succeeds.

        Returns
        -------
        out   : (total_output_nodes, n, n)
        flags : (total_output_nodes,) bool
        """
        all_out:   list[jax.Array] = []
        all_flags: list[jax.Array] = []

        cur_out, cur_flags = encoder_out, encoder_flags
        for layer in self.layers:
            o, f = layer(cur_out, cur_flags)
            all_out.append(o)
            all_flags.append(f)
            cur_out, cur_flags = o, f

        # Gather per-layer output nodes (skip layers with no output nodes)
        parts_out:   list[jax.Array] = []
        parts_flags: list[jax.Array] = []
        for l_idx, idx in enumerate(self.output_node_indices):
            if idx.size > 0:
                parts_out.append(all_out[l_idx][idx])
                parts_flags.append(all_flags[l_idx][idx])

        return jnp.concatenate(parts_out, axis=0), jnp.concatenate(parts_flags, axis=0)

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx

from utils.decoder import Decoder
from utils.config.trainparams import (
    DECODER_ACTIVATION_THRESHOLD,
    DECODER_TOP_K_EXTRACT,
    DECODER_INTERACT_RANKS,
    DECODER_PRESERVE_RANKS,
    DECODER_INTERMEDIATE_CHANNELS,
    DECODER_HIDDEN_CHANNELS,
    DECODER_OUT_CHANNELS,
)


# Precomputed static pair indices for DECODER_INTERACT_RANKS choose 2
_IDX1, _IDX2 = np.triu_indices(DECODER_INTERACT_RANKS, k=1)   # each shape (28,)
_N_PAIRS = len(_IDX1)                                           # 28 = C(8, 2)


class DecoderLayer(eqx.Module):
    """
    Decoder Layer: K independent Decoder nodes connected to the previous layer
    via a pre-computed static adjacency wiring array.

    The caller (e.g. DecoderCluster) is responsible for computing parent_indices
    using the constrained Gaussian fan-out wiring algorithm described in the
    architecture spec.  This layer stores the wiring and executes the K
    decoders in parallel.

    Call signature:
        prev_outputs:   (M, n, n) — previous layer's output matrices
        prev_is_active: (M,)      — previous layer's activity flags (bool)

    Returns:
        out:        (K, n, n)
        out_active: (K,) bool

    Implementation note
    -------------------
    All K decoder passes are implemented as vectorised batch-einsum operations
    without eqx.filter_vmap at call time.  For the pairwise batched matmul
    (K, 28, n, n) the K and pair dims are flattened to a single K*28 leading
    dim before the matmul, so XLA sees at most two batch dims under an outer
    jax.vmap (e.g. batched training), avoiding the 3-batch-dim axis-ordering
    bug in JAX's JIT+vmap matmul lowering.
    The decoder threshold gate uses jnp.where (lax.select) instead of lax.cond
    for full vmap compatibility.
    """

    # K stacked Decoder nodes (weight arrays have leading dim K)
    decoders: Decoder
    # Adjacency: (K, 16) int32 — which prev-layer outputs feed each decoder.
    parent_indices: np.ndarray

    def __init__(self, parent_indices: np.ndarray, key: jax.Array):
        """
        parent_indices : (K, 16) int32 numpy array — pre-computed wiring.
        key            : JAX PRNG key for weight initialisation.
        """
        self.parent_indices = np.asarray(parent_indices, dtype=np.int32)
        k_nodes = self.parent_indices.shape[0]

        dec_keys = jax.random.split(key, k_nodes)
        self.decoders = eqx.filter_vmap(lambda k: Decoder(k))(dec_keys)

    def __call__(self, prev_outputs: jax.Array, prev_is_active: jax.Array):
        """
        prev_outputs:  (M, n, n)  — previous layer output matrices
        prev_is_active: (M,)       — previous layer active flags (bool)

        Returns:
            out:        (K, n, n)
            out_active: (K,) bool
        """
        # ── Gather inputs for all K decoders ─────────────────────────────────
        gathered       = prev_outputs[self.parent_indices]    # (K, 16, n, n)
        gathered_flags = prev_is_active[self.parent_indices]  # (K, 16) bool

        K = self.parent_indices.shape[0]
        n = prev_outputs.shape[1]
        m = prev_outputs.shape[2]
        P = _N_PAIRS  # 28

        # ── Gate: ≥ 12/16 active parents ────────────────────────────────────
        gate_active = (
            jnp.sum(gathered_flags, axis=-1) >= DECODER_ACTIVATION_THRESHOLD
        )  # (K,)

        # ── Log-absolute-det proxy scores ────────────────────────────────────
        # slogdet operates on the last two dims; gathered is (K, 16, n, n)
        _, logabsdet = jnp.linalg.slogdet(gathered)          # (K, 16)

        # Mask inactive inputs so they cannot win the top-k contest
        scores = jnp.where(gathered_flags, logabsdet, -jnp.inf)  # (K, 16)

        # ── Top-K selection ───────────────────────────────────────────────────
        _, indices = jax.lax.top_k(scores, DECODER_TOP_K_EXTRACT)  # (K, 12)

        # Gather top-K matrices per decoder: (K, 12, n, n)
        top_x = gathered[jnp.arange(K)[:, None], indices]   # (K, 12, n, n)

        # Split into interact (8) and preserve (4)
        top_interact = top_x[:, :DECODER_INTERACT_RANKS]         # (K, 8, n, n)
        preserve     = top_x[:, DECODER_INTERACT_RANKS:
                              DECODER_INTERACT_RANKS + DECODER_PRESERVE_RANKS]  # (K, 4, n, n)

        # ── Pairwise batched matmul (flat-reshape trick) ──────────────────────
        # Flatten (K, 28) → (K*28) so the matmul has ONE pre-existing batch dim.
        # Under an outer B-vmap, XLA then sees (B, K*28) = 2 batch dims, which
        # is within the safe limit for JAX's jit+vmap lowering.
        left  = top_interact[:, _IDX1]               # (K, 28, n, n)
        right = top_interact[:, _IDX2]               # (K, 28, n, n)
        left_flat  = left.reshape(K * P, n, m)       # (K*28, n, m)
        right_flat = right.reshape(K * P, n, m)      # (K*28, n, m)
        pairs = jnp.matmul(left_flat, right_flat).reshape(K, P, n, m)  # (K, 28, n, m)

        # ── Form intermediate representation ─────────────────────────────────
        concat = jnp.concatenate([pairs, preserve], axis=1)  # (K, 32, n, m)

        # ── conv1: (K, 32) → (K, HIDDEN_CH=4) ────────────────────────────────
        # Weight after K-vmap: (K, HIDDEN=4, INTER=32, 1, 1) → (K, 4, 32)
        # Bias after K-vmap:   (K, 4, 1, 1)
        w1 = self.decoders.conv1.weight[..., 0, 0]  # (K, 4, 32)
        b1 = self.decoders.conv1.bias                # (K, 4, 1, 1)
        hidden = (jnp.einsum('koi,kihm->kohm', w1, concat) + b1)  # (K, 4, n, m)

        # ── conv2: (K, 4) → (K, 1) ───────────────────────────────────────────
        # Weight: (K, 1, 4, 1, 1) → (K, 1, 4)
        # Bias:   (K, 1, 1, 1)
        w2 = self.decoders.conv2.weight[..., 0, 0]  # (K, 1, 4)
        b2 = self.decoders.conv2.bias                # (K, 1, 1, 1)
        pre_act = (jnp.einsum('koi,kihm->kohm', w2, hidden) + b2)  # (K, 1, n, m)
        active_out = jax.nn.swish(pre_act).squeeze(1)               # (K, n, m)

        # ── Gate: zero-out decoders below activation threshold ────────────────
        # jnp.where (lax.select) is vmap-compatible; gradients are masked to
        # zero for inactive decoder nodes automatically via VJP.
        out = jnp.where(
            gate_active[:, None, None],
            active_out,
            jnp.zeros_like(active_out),
        )  # (K, n, m)

        return out, gate_active  # (K, n, m), (K,)

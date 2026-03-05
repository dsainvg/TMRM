"""
Multi-problem TMRM Model.

Wraps a shared backbone (EncoderLayer + DecoderCluster) with per-problem
encoder masks and independent FC heads.  Each problem randomly selects a
subset of encoder slots at initialisation; at forward time a Python-int
``problem_idx`` picks the corresponding mask and FC head.

Usage
-----
>>> from utils.config.model import ModelConfig, ProblemConfig
>>> cfg = ModelConfig(
...     n=8,
...     n_encoders=4,
...     n_decoder_layers=2,
...     max_decoder_nodes=50,
...     problems=(
...         ProblemConfig(n_encoders_used=2, fc_out_features=10),
...         ProblemConfig(n_encoders_used=3, fc_out_features=64, fc_activation='identity'),
...     ),
... )
>>> model = Model(cfg, key=jax.random.key(0))
>>> xs = jax.random.normal(jax.random.key(1), (cfg.n_encoders, 1, cfg.n, cfg.n))
>>> out_p0 = model(0, xs)   # problem 0 -> shape (10,)
>>> out_p1 = model(1, xs)   # problem 1 -> shape (64,)
"""

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx

from utils.encoder_layer import EncoderLayer
from utils.decoder_cluster import DecoderCluster
from utils.pa_layer import PALayer
from utils.config.model import ModelConfig
from utils.config.trainparams import ENCODER_STACK_OUT_CHANNELS


# ── Module-level helpers ───────────────────────────────────────────────────────

def _slot_bar(n: int, active: set) -> str:
    """Return a compact visual bar like  [X] [ ] [X] [X] [ ] [X]  for slot activity."""
    return "  ".join("[X]" if i in active else " . " for i in range(n))


def _fanout_bar(n: int, fanout: np.ndarray) -> str:
    """Return a bar showing fan-out per node: [0]  [3]   .   [1] ...
    Nodes with zero fan-out are shown as ' . ', others show their count.
    When n > 18, shows first 15 then '...' then last 3."""
    MAX_SHOW = 18
    def _cell(i: int) -> str:
        v = int(fanout[i])
        return f"[{v}]" if v > 0 else " . "
    if n <= MAX_SHOW:
        parts = [_cell(i) for i in range(n)]
    else:
        head = [_cell(i) for i in range(MAX_SHOW - 3)]
        tail = [_cell(i) for i in range(n - 3, n)]
        parts = head + ["..."] + tail
    return "  ".join(parts)


class Model(eqx.Module):
    """
    Multi-problem TMRM network.

    A single shared backbone (EncoderLayer -> DecoderCluster) serves all
    problems.  Each problem has:
      - a random encoder mask selecting which of the ``n_encoders`` slots
        are active (determined once at init, frozen thereafter), and
      - its own independent PALayer (Port Adapter) that reduces decoder
        output channels to the task-specific output shape via a 1×1 conv.

    ``problem_idx`` is a **Python int** (not a JAX-traced value).  Each
    distinct problem therefore traces its own XLA program — this is
    unavoidable because port adapters may have different output shapes.

    Parameters
    ----------
    config : ModelConfig
        Frozen dataclass holding all structural hyperparameters.
    key : jax.Array
        JAX PRNG key for weight initialisation and mask generation.
    """

    # ── static metadata (invisible to JAX gradient / pytree) ──────────────
    config: ModelConfig = eqx.field(static=True)

    # ── shared trainable backbone ─────────────────────────────────────────
    encoder_layer: EncoderLayer
    decoder_cluster: DecoderCluster

    # ── per-problem trainable port adapters ───────────────────────────────
    port_adapters: list  # list[PALayer]

    # ── per-problem non-trainable masks (plain NumPy, compile-time const) ─
    # Static: masks are fixed at init and must not be traced through JIT
    # (avoids np→jax type mutation that retriggers XLA compilation).
    encoder_masks: list = eqx.field(static=True)  # list[np.ndarray] bool

    def __init__(self, config: ModelConfig, key: jax.Array):
        self.config = config

        k_enc, k_dec, k_fc, k_mask = jax.random.split(key, 4)

        # ── Shared encoder layer ──────────────────────────────────────────
        self.encoder_layer = EncoderLayer(
            n_inputs=config.n_encoders, key=k_enc,
        )

        # ── Shared decoder cluster ────────────────────────────────────────
        n_encoder_outputs = config.n_encoders * ENCODER_STACK_OUT_CHANNELS
        self.decoder_cluster = DecoderCluster(
            n_layers=config.n_decoder_layers,
            max_nodes=config.max_decoder_nodes,
            n_inputs=n_encoder_outputs,
            key=k_dec,
        )

        # ── Per-problem Port Adapters ───────────────────────────────────────
        # 1×1 conv: n_output_nodes → pa_out_channels (e.g. 15 → 4)
        # Params: 15 × 4 + 4 = 64  (vs 15,424 for a dense FC 240→64)
        pa_in = self.decoder_cluster.n_output_nodes  # e.g. 15
        pa_keys = jax.random.split(k_fc, config.n_problems)
        self.port_adapters = [
            PALayer(
                in_channels=pa_in,
                out_channels=config.problems[i].pa_out_channels,
                activation=config.problems[i].pa_activation,
                key=pa_keys[i],
            )
            for i in range(config.n_problems)
        ]

        # ── Per-problem encoder masks (round-robin then random) ───────────
        # Shuffle all encoder slots once to establish a canonical random
        # order.  Problems draw consecutive windows from this cyclic order,
        # so every slot is visited before any slot is reused.  Once the
        # window wraps past the end, later problems naturally begin to share
        # slots with earlier ones — giving the "common places" overlap after
        # full coverage.
        seed = int(jax.random.randint(k_mask, (), 0, 2**31 - 1))
        rng = np.random.default_rng(seed)
        base_order = rng.permutation(config.n_encoders)  # one global shuffle

        masks = []
        offset = 0
        for p in config.problems:
            # Consecutive window into the cyclic base order.
            raw = np.array([
                base_order[i % config.n_encoders]
                for i in range(offset, offset + p.n_encoders_used)
            ])
            # De-duplicate wrap-around collisions while preserving order.
            seen: set = set()
            chosen = []
            for idx in raw.tolist():
                if idx not in seen:
                    seen.add(idx)
                    chosen.append(idx)
            mask = np.zeros(config.n_encoders, dtype=bool)
            mask[chosen] = True
            masks.append(mask)
            offset += p.n_encoders_used
        self.encoder_masks = masks

        # ── Print architecture immediately after init ─────────────────────
        self.print_architecture()

    # ── architecture printer ───────────────────────────────────────────────

    def print_architecture(self) -> None:
        """Print a human-readable summary of the network topology.

        Called automatically at the end of ``__init__``.  Shows:

        * Backbone structure (encoder stacks, decoder layers, output nodes)
        * Per-problem encoder masks (active / inactive / shared / exclusive)
        * Decoder layer node counts and output nodes
        * FC head shapes
        * Parameter counts
        """
        W = 65
        SEP  = "=" * W
        SEP2 = "-" * W

        cfg = self.config
        n_enc = cfg.n_encoders

        def _banner(title: str) -> str:
            return f"  {title}"

        print(SEP)
        print(f"  TMRM Network  |  grid={cfg.n}x{cfg.n}  |  problems={cfg.n_problems}")
        print(SEP)

        # ── Encoder backbone ───────────────────────────────────────────────
        from utils.config.trainparams import (
            ENCODER_STACK_DEPTH,
            ENCODER_INTER_STACK_CHANNELS,
            ENCODER_STAGE2_COUNT,
            ENCODER_OUT_CHANNELS,
            ENCODER_STACK_OUT_CHANNELS,
        )
        print(_banner("ENCODER LAYER"))
        print(SEP2)
        print(f"  Slots              : {n_enc}")
        print(f"  Tree depth         : {ENCODER_STACK_DEPTH}  (1 root -> {ENCODER_STAGE2_COUNT} leaves)")
        print(f"  Root out-channels  : {ENCODER_INTER_STACK_CHANNELS}")
        print(f"  Leaf out-channels  : {ENCODER_OUT_CHANNELS}")
        print(f"  Channels per slot  : {ENCODER_STACK_OUT_CHANNELS}  ({ENCODER_STAGE2_COUNT}x{ENCODER_OUT_CHANNELS})")
        print(f"  Total enc channels : {n_enc * ENCODER_STACK_OUT_CHANNELS}  ({n_enc}x{ENCODER_STACK_OUT_CHANNELS})")

        # ── Decoder cluster ────────────────────────────────────────────────
        dc = self.decoder_cluster
        n_enc_ch = n_enc * ENCODER_STACK_OUT_CHANNELS  # total encoder channels

        from utils.config.trainparams import (
            DECODER_MAX_PARENTS,
            DECODER_ACTIVATION_THRESHOLD,
            DECODER_TOP_K_EXTRACT,
            DECODER_INTERACT_RANKS,
            DECODER_PRESERVE_RANKS,
            DECODER_INTERMEDIATE_CHANNELS,
            DECODER_HIDDEN_CHANNELS,
            DECODER_OUT_CHANNELS,
            FANOUT_FIRST_MU, FANOUT_FIRST_SIGMA, FANOUT_FIRST_LO, FANOUT_FIRST_HI,
            FANOUT_SECOND_MU, FANOUT_SECOND_SIGMA, FANOUT_SECOND_LO, FANOUT_SECOND_HI,
        )
        import math as _math
        n_pairs = _math.comb(DECODER_INTERACT_RANKS, 2)   # C(8,2) = 28

        # ── Decoder node schema (every node uses identical internal pipeline) ──
        print()
        print(_banner("DECODER NODE SCHEMA  (applies to every decoder node)"))
        print(SEP2)
        print(f"  Inputs / node      : {DECODER_MAX_PARENTS} parent matrices (n x n each)")
        print(f"  Activation gate    : >= {DECODER_ACTIVATION_THRESHOLD}/{DECODER_MAX_PARENTS} parents active  "
              f"(else output zeros)")
        print(f"  Scoring            : log-absolute-determinant (slogdet) per input")
        print(f"  Top-K extract      : {DECODER_TOP_K_EXTRACT} candidates selected by score")
        print(f"    Interact slice   : top {DECODER_INTERACT_RANKS}  -> {n_pairs} pairwise matmuls  "
              f"C({DECODER_INTERACT_RANKS},2)")
        print(f"    Preserve slice   : next {DECODER_PRESERVE_RANKS}  -> passed through unchanged")
        print(f"  Intermediate ch    : {DECODER_INTERMEDIATE_CHANNELS}  "
              f"({n_pairs} pairs + {DECODER_PRESERVE_RANKS} preserved)")
        print(f"  Conv1              : {DECODER_INTERMEDIATE_CHANNELS} -> {DECODER_HIDDEN_CHANNELS} ch  (1x1 conv)")
        print(f"  Conv2              : {DECODER_HIDDEN_CHANNELS} -> {DECODER_OUT_CHANNELS} ch  (1x1 conv, swish)")
        print(f"  Output / node      : {DECODER_OUT_CHANNELS} channel  (n x n matrix)")

        # ── Cluster summary ────────────────────────────────────────────────
        print()
        print(_banner("DECODER CLUSTER"))
        print(SEP2)
        print(f"  Layers built       : {len(dc.layers)}")
        midpoint = (cfg.n_decoder_layers + 2) // 2
        print(f"  Gaussian fan-out   : L0 straight-wire, "
              f"L1..{midpoint-1} mu={FANOUT_FIRST_MU} s={FANOUT_FIRST_SIGMA} "
              f"[{FANOUT_FIRST_LO},{FANOUT_FIRST_HI}], "
              f"L{midpoint}+ mu={FANOUT_SECOND_MU} s={FANOUT_SECOND_SIGMA} "
              f"[{FANOUT_SECOND_LO},{FANOUT_SECOND_HI}]")
        print(f"  Total output nodes : {dc.n_output_nodes}")
        print(f"  PA input channels  : {dc.n_output_nodes}  (spatial {dc.n_output_nodes}x{cfg.n}x{cfg.n})")
        # node count per layer
        counts = [lyr.parent_indices.shape[0] for lyr in dc.layers]
        print(f"  Nodes per layer    : {counts}  (total wired = {sum(counts)})")

        # ── Decoder connectivity (encoder-slot-sharing style) ──────────────
        print()
        print(_banner("DECODER CLUSTER CONNECTIVITY"))
        print(SEP2)

        for li, (layer, out_idx) in enumerate(zip(dc.layers, dc.output_node_indices)):
            pi_arr = layer.parent_indices   # (K, 16) int32
            K      = pi_arr.shape[0]
            n_out  = out_idx.size
            total_slots = K * DECODER_MAX_PARENTS

            if li == 0:
                # ── Encoder -> Layer 0: group 64-ch bundles by encoder slot ──
                n_slots = n_enc
                ch_per  = ENCODER_STACK_OUT_CHANNELS
                slot_conn_count = np.zeros(n_slots, dtype=int)   # #L0 nodes slot feeds
                slot_slot_count = np.zeros(n_slots, dtype=int)   # #wiring slots used
                for s in range(n_slots):
                    lo_ch, hi_ch = s * ch_per, (s + 1) * ch_per
                    mask = (pi_arr >= lo_ch) & (pi_arr < hi_ch)  # (K,16) bool
                    slot_conn_count[s] = int(np.any(mask, axis=1).sum())
                    slot_slot_count[s] = int(mask.sum())
                active_slots   = set(int(s) for s in range(n_slots) if slot_conn_count[s] > 0)
                inactive_slots = set(range(n_slots)) - active_slots
                bar = _fanout_bar(n_slots, slot_conn_count)
                print(f"  Encoder slots -> Layer 0")
                print(f"    Input channels   : {n_enc_ch}  ({n_enc} slots x {ch_per} ch)")
                print(f"    Output nodes     : {K}")
                print(f"    Total wire slots : {total_slots}  ({K} nodes x {DECODER_MAX_PARENTS} parents)")
                print(f"    Slot fan-out bar : {bar}")
                print(f"      L0 nodes fed/slot  : {slot_conn_count.tolist()}")
                print(f"      wire slots/enc slot: {slot_slot_count.tolist()}")
                print(f"    Active enc slots : {sorted(active_slots)}")
                print(f"    Unused enc slots : {sorted(inactive_slots) or '(none)'}")
            else:
                # ── Layer (li-1) -> Layer li ──────────────────────────────
                prev_layer  = dc.layers[li - 1]
                n_prev      = prev_layer.parent_indices.shape[0]
                # actual count of appearances in pi_arr (each node can appear multiple times)
                node_count  = np.array([(pi_arr == node).sum() for node in range(n_prev)], dtype=int)
                connected   = set(int(i) for i in range(n_prev) if node_count[i] > 0)
                unconnected = set(range(n_prev)) - connected
                prev_out_idx = dc.output_node_indices[li - 1]
                actual_out   = set(prev_out_idx.tolist())
                bar = _fanout_bar(n_prev, node_count)
                fwd_only  = connected - actual_out
                out_only  = actual_out - connected
                fwd_and_out = connected & actual_out

                print(f"  Layer {li-1} -> Layer {li}")
                print(f"    Prev nodes       : {n_prev}  ->  {K} new nodes")
                print(f"    Total wire slots : {total_slots}  ({K} x {DECODER_MAX_PARENTS})")
                print(f"    Unique parents   : {len(connected)}  "
                      f"({n_prev - len(connected)} unconnected below)")
                print(f"    Duplicate hits   : {int(node_count.sum()) - len(connected)}  "
                      f"(same node wired into multiple decoder slots)")
                nc_nonzero = node_count[node_count > 0]
                if nc_nonzero.size > 0:
                    print(f"    Fan-out stats    : "
                          f"min={nc_nonzero.min()}  max={nc_nonzero.max()}  "
                          f"mean={nc_nonzero.mean():.1f}  sum={int(nc_nonzero.sum())}")
                print(f"    Wire count/node  : {bar}")
                print(f"    Forward only     : {sorted(fwd_only) or '(none)'}")
                print(f"    Output only      : {sorted(out_only) or '(none)'}  (skipped to output)")
                print(f"    Forward + output : {sorted(fwd_and_out) or '(none)'}")
                print(f"    Unconnected      : {sorted(unconnected) or '(none)'}")

            # ── Per-layer output fate ──────────────────────────────────────
            if n_out > 0:
                out_list = (out_idx.tolist() if n_out <= 20
                            else out_idx[:10].tolist() + ["..."] + out_idx[-5:].tolist())
                print(f"    Output nodes     : {n_out}  ->  {out_list}")
            else:
                print(f"    Output nodes     : 0  (all feed into next layer)")

            # ── Per-decoder parent list (trimmed for large layers) ─────────
            if K <= 12:
                print(f"    Per-decoder parents (node <- [parent indices]):")
                for di in range(K):
                    role = " <OUT>" if di in set(out_idx.tolist()) else ""
                    print(f"      node {di:3d}{role}: {pi_arr[di].tolist()}")
            else:
                print(f"    Per-decoder parents  (first 6 / last 3 of {K} nodes):")
                for di in list(range(6)) + list(range(K - 3, K)):
                    if di == 6:
                        print(f"      ...")
                    role = " <OUT>" if di in set(out_idx.tolist()) else ""
                    print(f"      node {di:3d}{role}: {pi_arr[di].tolist()}")
            print()

        # ── Per-problem encoder masks ──────────────────────────────────────
        print()
        print(_banner("ENCODER MASKS  (per problem)"))
        print(SEP2)

        all_active = [set(np.where(m)[0].tolist()) for m in self.encoder_masks]
        all_inactive = [set(range(n_enc)) - a for a in all_active]

        for pi in range(cfg.n_problems):
            active_idx = sorted(all_active[pi])
            inactive_idx = sorted(all_inactive[pi])
            used_str   = _slot_bar(n_enc, all_active[pi])
            print(f"  Problem {pi}  |  {len(active_idx)}/{n_enc} active")
            print(f"    slots:     {used_str}")
            print(f"    active:    {active_idx}")
            print(f"    inactive:  {inactive_idx}")

        # Cross-problem overlap (only meaningful when > 1 problem)
        if cfg.n_problems > 1:
            print()
            print(_banner("ENCODER SLOT SHARING"))
            print(SEP2)
            shared = all_active[0].copy()
            for a in all_active[1:]:
                shared &= a
            print(f"  Shared by ALL    : {sorted(shared) or '(none)'}")
            for pi in range(cfg.n_problems):
                exclusive = all_active[pi] - set().union(*(all_active[j] for j in range(cfg.n_problems) if j != pi))
                print(f"  Exclusive to P{pi} : {sorted(exclusive) or '(none)'}")
            never_used = set(range(n_enc)) - set().union(*all_active)
            print(f"  Never active     : {sorted(never_used) or '(none)'}")

            # Side-by-side slot bars for each problem
            print()
            print("  Slot key: [X]=active  .=inactive")
            for pi in range(cfg.n_problems):
                bar = _slot_bar(n_enc, all_active[pi])
                print(f"    P{pi}: {bar}")

        # ── Port Adapters ─────────────────────────────────────────────────────────
        print()
        print(_banner("PORT ADAPTERS"))
        print(SEP2)
        pa_in = self.decoder_cluster.n_output_nodes
        for pi, (pa, pcfg) in enumerate(zip(self.port_adapters, cfg.problems)):
            pa_out = pcfg.pa_out_channels
            pa_params = pa_in * pa_out + pa_out
            print(
                f"  Problem {pi}  |  "
                f"in_ch={pa_in}  out_ch={pa_out}  "
                f"output=({pa_out},{cfg.n},{cfg.n})  "
                f"activation={pcfg.pa_activation}  "
                f"params={pa_params:,}"
            )

        # ── Parameter counts ───────────────────────────────────────────────
        print()
        print(_banner("PARAMETER COUNT"))
        print(SEP2)
        params = self.count_params()
        print(f"  EncoderLayer     : {params['encoder_layer']:>10,}")
        print(f"  DecoderCluster   : {params['decoder_cluster']:>10,}")
        for pi, pa_p in enumerate(params['port_adapters']):
            print(f"  Port Adapter {pi}   : {pa_p:>10,}")
        print(SEP2)
        print(f"  TOTAL            : {params['total']:>10,}")
        print(SEP)
        print()

    # ── forward pass ──────────────────────────────────────────────────────

    def __call__(self, problem_idx: int, xs: jax.Array) -> jax.Array:
        """
        Run the network for a single problem.

        Parameters
        ----------
        problem_idx : int
            **Python int** selecting the problem head (0-based).
            Each distinct value traces a separate XLA program.
        xs : jax.Array, shape ``(n_encoders, 1, n, n)``
            Full encoder input tensor.  Slots not in this problem's
            encoder mask are deactivated via the ``is_active`` flags.

        Returns
        -------
        jax.Array, shape ``(pa_out_channels * n * n,)``
            Task-specific output produced by the selected Port Adapter.
        """
        # Encoder mask -> JAX bool array (literal at trace time)
        flags = jnp.array(self.encoder_masks[problem_idx])

        # Shared encoder
        enc_out, enc_flags = self.encoder_layer(xs, flags)

        # Shared decoder cluster
        dec_out, _ = self.decoder_cluster(enc_out, enc_flags)

        # Problem-specific Port Adapter: 1×1 conv over (n_nodes, n, n) → flattened
        return self.port_adapters[problem_idx](dec_out)

    # ── helpers ────────────────────────────────────────────────────────────

    @property
    def n_problems(self) -> int:
        """Number of problem heads."""
        return self.config.n_problems

    @property
    def n_decoder_output_nodes(self) -> int:
        """Total output nodes across all DecoderCluster layers."""
        return self.decoder_cluster.n_output_nodes

    @property
    def pa_in_channels(self) -> int:
        """Number of input channels to each Port Adapter (= decoder output nodes)."""
        return self.decoder_cluster.n_output_nodes

    def active_encoder_indices(self, problem_idx: int) -> np.ndarray:
        """Return the encoder slot indices that are active for a problem.

        Useful for logging / debugging which encoders a problem uses.

        Returns
        -------
        np.ndarray of int
            Sorted array of active encoder indices.
        """
        return np.where(self.encoder_masks[problem_idx])[0]

    def count_params(self) -> dict:
        """Count trainable parameters broken down by component.

        Returns
        -------
        dict with keys:
            'encoder_layer'  : int
            'decoder_cluster': int
            'port_adapters'  : list[int]  (one per problem)
            'total'          : int
        """
        def _sum_leaves(module):
            return sum(
                x.size for x in jax.tree_util.tree_leaves(
                    eqx.filter(module, eqx.is_inexact_array)
                )
            )

        enc = _sum_leaves(self.encoder_layer)
        dec = _sum_leaves(self.decoder_cluster)
        pa  = [_sum_leaves(h) for h in self.port_adapters]
        return {
            'encoder_layer':   enc,
            'decoder_cluster': dec,
            'port_adapters':   pa,
            'total': enc + dec + sum(pa),
        }

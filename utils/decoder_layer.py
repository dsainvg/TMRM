import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx

from utils.decoder import Decoder
from utils.config.decode import DECODER_MAX_PARENTS


class DecoderLayer(eqx.Module):
    """
    Decoder Layer: K independent Decoder nodes connected to the previous layer
    via a static adjacency wiring array.

    At init time, each of the K Decoder nodes is randomly assigned
    DECODER_MAX_PARENTS (16) parent indices sampled (with replacement) from
    the M outputs of the previous layer.  This parent_indices array is
    fixed for the lifetime of the model — it is a static JAX constant baked
    in as a pytree leaf.

    Call signature:
        prev_outputs: (M, n, n)  — previous layer's output matrices
        prev_is_active: (M,)     — previous layer's activity flags (bool)

    Returns:
        out:        (K, 1, n, n)
        out_active: (K,) bool

    The Decoder's own threshold gate (>= 12/16 active parents) governs
    whether each node produces a real output or a zero tensor.
    """

    # K stacked Decoder nodes (weight arrays have leading dim K)
    decoders: Decoder
    # Adjacency: (K, 16) int32 — which prev-layer outputs feed each decoder.
    # Stored as a NumPy array so Equinox treats it as static pytree structure
    # (not a traced JAX leaf).  XLA therefore sees the exact gather indices as
    # compile-time literals and can constant-fold through the gather — no
    # eqx.field(static=True) required, no warning produced.
    parent_indices: np.ndarray

    def __init__(self, k_nodes: int, n_prev_outputs: int, key: jax.Array):
        key_dec, key_wire = jax.random.split(key)

        # ── Wiring ────────────────────────────────────────────────────────────
        # Each decoder randomly selects DECODER_MAX_PARENTS indices (with
        # replacement) from n_prev_outputs available parent slots.
        # replace=True means no constraint on n_prev_outputs size.
        wire_keys = jax.random.split(key_wire, k_nodes)
        # Compute wiring with JAX then immediately materialise as NumPy so the
        # array is stored as static pytree metadata (compile-time constant).
        self.parent_indices = np.array(
            jax.vmap(
                lambda k: jax.random.choice(
                    k, n_prev_outputs, shape=(DECODER_MAX_PARENTS,), replace=True
                )
            )(wire_keys)
        )  # (K, 16) numpy int32

        # ── Decoder nodes ─────────────────────────────────────────────────────
        dec_keys = jax.random.split(key_dec, k_nodes)
        self.decoders = eqx.filter_vmap(lambda k: Decoder(k))(dec_keys)

    def __call__(self, prev_outputs: jax.Array, prev_is_active: jax.Array):
        """
        prev_outputs:  (M, n, n)  — previous layer output matrices
        prev_is_active: (M,)       — previous layer active flags (bool)

        Returns:
            out:        (K, 1, n, n)
            out_active: (K,) bool
        """
        # Gather parent inputs and flags for each decoder via static adjacency.
        # parent_indices: (K, 16)  →  gathered: (K, 16, n, n)
        gathered       = prev_outputs[self.parent_indices]        # (K, 16, n, n)
        gathered_flags = prev_is_active[self.parent_indices]      # (K, 16) bool

        # Run all K decoders in parallel — each receives (16, n, n) + (16,) flags
        return eqx.filter_vmap(lambda dec, x, f: dec(x, f))(
            self.decoders, gathered, gathered_flags
        )

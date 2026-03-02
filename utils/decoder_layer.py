import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx

from utils.decoder import Decoder


class DecoderLayer(eqx.Module):
    """
    Decoder Layer: K independent Decoder nodes connected to the previous layer
    via a pre-computed static adjacency wiring array.

    The caller (e.g. DecoderCluster) is responsible for computing parent_indices
    using the constrained Gaussian fan-out wiring algorithm described in the
    architecture spec.  This layer simply stores the wiring and executes the K
    decoders in parallel.

    Call signature:
        prev_outputs:   (M, n, n) — previous layer's output matrices
        prev_is_active: (M,)      — previous layer's activity flags (bool)

    Returns:
        out:        (K, n, n)
        out_active: (K,) bool

    The Decoder's own threshold gate (>= 12/16 active parents) governs
    whether each node produces a real output or a zero tensor.
    """

    # K stacked Decoder nodes (weight arrays have leading dim K)
    decoders: Decoder
    # Adjacency: (K, 16) int32 — which prev-layer outputs feed each decoder.
    # Stored as a NumPy array so Equinox treats it as static pytree structure
    # (not a traced JAX leaf).  XLA therefore sees the exact gather indices as
    # compile-time literals and can constant-fold through the gather.
    parent_indices: np.ndarray

    def __init__(self, parent_indices: np.ndarray, key: jax.Array):
        """
        parent_indices : (K, 16) int32 numpy array — pre-computed wiring.
                         K (number of decoder nodes) is inferred from axis 0.
        key            : JAX PRNG key for weight initialisation.
        """
        # Ensure stored as numpy int32 (compile-time constant for XLA)
        self.parent_indices = np.asarray(parent_indices, dtype=np.int32)
        k_nodes = self.parent_indices.shape[0]

        # ── Decoder nodes ─────────────────────────────────────────────────────
        dec_keys = jax.random.split(key, k_nodes)
        self.decoders = eqx.filter_vmap(lambda k: Decoder(k))(dec_keys)

    def __call__(self, prev_outputs: jax.Array, prev_is_active: jax.Array):
        """
        prev_outputs:  (M, n, n)  — previous layer output matrices
        prev_is_active: (M,)       — previous layer active flags (bool)

        Returns:
            out:        (K, n, n)   — Decoder squeezes the channel dim, vmap stacks K results
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

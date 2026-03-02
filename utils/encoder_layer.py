import jax
import jax.numpy as jnp
import equinox as eqx

from utils.encoder import Encoder
from utils.config.encode import ENCODER_IN_CHANNELS, ENCODER_OUT_CHANNELS
from utils.config.encoder_layer import (
    ENCODER_INTER_STACK_CHANNELS,
    ENCODER_STAGE2_COUNT,
    ENCODER_STACK_OUT_CHANNELS,
)


class EncoderLayer(eqx.Module):
    """
    Encoder Layer: N independent tree encoder stacks.

    Each stack is a 1-to-8 branching tree of Encoder nodes (9 nodes total per input):

        Stage 1 (root)  : 1 Encoder   (1, n, n) -> (8, n, n)
                                        │
                              split into 8 single channels
                                        │
        Stage 2 (leaves): 8 Encoders  (1, n, n) -> (8, n, n)  each
                                        │
                              concatenated -> (64, n, n)  per input

    All N stacks are vmapped in parallel then reshaped so the layer maps:
        xs:   (N, 1, n, n)  +  is_active_flags: (N,)
        ->    (64*N, n, n)  +  out_active:       (64*N,)

    Each leaf flag covers 8 output channels; all 8 leaves produce (8,) flags
    which are repeated to (64,) per stack, broadcast to (64*N,) across stacks.
    Inactive flags short-circuit via each Encoder's own lax.cond gate.
    """

    # N stage-1 root encoders, params batched over axis 0 (size N)
    stage1_encs: Encoder
    # N×8 stage-2 leaf encoders, params batched over axes (N, 8)
    stage2_encs: Encoder

    def __init__(self, n_inputs: int, key: jax.Array):
        n_leaves = ENCODER_STAGE2_COUNT  # 8

        k_s1, k_s2 = jax.random.split(key)

        # (N,) keys for stage-1 roots
        keys_s1 = jax.random.split(k_s1, n_inputs)

        # (N*8,) keys for stage-2 leaves, reshaped to (N, 8, key_shape)
        keys_s2_flat = jax.random.split(k_s2, n_inputs * n_leaves)
        keys_s2 = keys_s2_flat.reshape((n_inputs, n_leaves) + keys_s2_flat.shape[1:])

        # Build N stage-1 encoders: (1, n, n) -> (8, n, n)
        self.stage1_encs = eqx.filter_vmap(
            lambda k: Encoder(k, in_channels=ENCODER_IN_CHANNELS, out_channels=ENCODER_INTER_STACK_CHANNELS)
        )(keys_s1)

        # Build N×8 stage-2 leaf encoders: (1, n, n) -> (8, n, n) each
        self.stage2_encs = eqx.filter_vmap(
            eqx.filter_vmap(
                lambda k: Encoder(k, in_channels=ENCODER_IN_CHANNELS, out_channels=ENCODER_OUT_CHANNELS)
            )
        )(keys_s2)

    def __call__(self, xs: jax.Array, is_active_flags: jax.Array):
        """
        xs:              (N, 1, n, n)  — N single-channel input matrices
        is_active_flags: (N,)          — per-input activity flags (bool)

        Returns:
            out:        (64*N, n, n)  — all stacks concatenated on axis 0
            out_active: (64*N,) bool  — per-leaf flag repeated over 8 channels each
        """
        n_leaves  = ENCODER_STAGE2_COUNT         # 8
        leaf_out  = ENCODER_OUT_CHANNELS          # 8 channels per leaf

        def _run_one_stack(enc1, enc2s, x, active):
            # Stage 1 (root): (1, n, n) -> (8, n, n)
            out1, act1 = enc1(x, active)
            n, m = out1.shape[1], out1.shape[2]

            # Split root output into 8 single-channel inputs for the leaves
            channels = out1.reshape(n_leaves, 1, n, m)          # (8, 1, n, n)
            acts     = jnp.broadcast_to(act1, (n_leaves,))       # (8,)

            # Stage 2 (leaves): 8 × (1, n, n) -> 8 × (8, n, n)
            outs2, acts2 = eqx.filter_vmap(
                lambda enc, ch, act: enc(ch, act)
            )(enc2s, channels, acts)
            # outs2: (8, 8, n, n) | acts2: (8,)

            # Flatten leaves: (8, 8, n, n) -> (64, n, n)
            out = outs2.reshape(n_leaves * leaf_out, n, m)

            # Each of 8 leaf flags covers leaf_out=8 output channels
            out_active = jnp.repeat(acts2, leaf_out)             # (64,)

            return out, out_active

        # stacked_out:    (N, 64, n, n)
        # stacked_active: (N, 64)
        stacked_out, stacked_active = eqx.filter_vmap(_run_one_stack)(
            self.stage1_encs, self.stage2_encs, xs, is_active_flags
        )

        n, m = stacked_out.shape[2], stacked_out.shape[3]
        out        = stacked_out.reshape(-1, n, m)   # (64*N, n, n)
        out_active = stacked_active.reshape(-1)       # (64*N,)

        return out, out_active

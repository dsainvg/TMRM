import jax
import jax.numpy as jnp
import equinox as eqx

from utils.encoder import Encoder
from utils.config.encode import ENCODER_IN_CHANNELS
from utils.config.encoder_layer import (
    ENCODER_INTER_STACK_CHANNELS,
    ENCODER_STACK_OUT_CHANNELS,
)


class EncoderLayer(eqx.Module):
    """
    Encoder Layer: N independent 2-encoder stacks.

    Each stack processes one (1, n, n) input through two sequential Encoder
    nodes and produces a (64, n, n) output:

        Stage 1 — (1,  n, n) -> (8,  n, n)   [ENCODER_IN_CHANNELS -> ENCODER_INTER_STACK_CHANNELS]
        Stage 2 — (8,  n, n) -> (64, n, n)   [ENCODER_INTER_STACK_CHANNELS -> ENCODER_STACK_OUT_CHANNELS]

    All N stacks are then concatenated along the channel axis so the layer maps:
        xs:   (N, 1, n, n)    +  is_active_flags: (N,)
        ->    (64*N, n, n)    +  out_active:       (64*N,)

    Each stack's single flag is broadcast across its 64 output channels, making
    the output directly consumable by DecoderLayer which expects (M, n, n) + (M,).

    Inactive flags propagate naturally through each Encoder's own lax.cond gate
    — no separate layer-level gate is needed.
    """

    # N stacked stage-1 encoders (weight arrays have leading dim N)
    stage1_encs: Encoder
    # N stacked stage-2 encoders (weight arrays have leading dim N)
    stage2_encs: Encoder

    def __init__(self, n_inputs: int, key: jax.Array):
        keys = jax.random.split(key, 2 * n_inputs)
        keys_s1 = keys[:n_inputs]
        keys_s2 = keys[n_inputs:]

        # Build N stage-1 encoders: (1, n, n) -> (8, n, n)
        self.stage1_encs = eqx.filter_vmap(
            lambda k: Encoder(k, in_channels=ENCODER_IN_CHANNELS, out_channels=ENCODER_INTER_STACK_CHANNELS)
        )(keys_s1)

        # Build N stage-2 encoders: (8, n, n) -> (64, n, n)
        self.stage2_encs = eqx.filter_vmap(
            lambda k: Encoder(k, in_channels=ENCODER_INTER_STACK_CHANNELS, out_channels=ENCODER_STACK_OUT_CHANNELS)
        )(keys_s2)

    def __call__(self, xs: jax.Array, is_active_flags: jax.Array):
        """
        xs:              (N, 1, n, n)  — N single-channel input matrices
        is_active_flags: (N,)          — per-input activity flags (bool)

        Returns:
            out:        (64*N, n, n)  — all stacks concatenated on axis 0
            out_active: (64*N,) bool  — each stack flag broadcast over 64 channels
        """
        def _run_one_stack(enc1, enc2, x, active):
            # Stage 1: (1, n, n) -> (8, n, n)
            out1, act1 = enc1(x, active)
            # Stage 2: (8, n, n) -> (64, n, n)
            out2, act2 = enc2(out1, act1)
            return out2, act2

        # stacked_out:    (N, 64, n, n)
        # stacked_active: (N,)
        stacked_out, stacked_active = eqx.filter_vmap(_run_one_stack)(
            self.stage1_encs, self.stage2_encs, xs, is_active_flags
        )

        # Flatten N stacks onto the leading axis: (N, 64, n, n) -> (64*N, n, n)
        n64, n, m = stacked_out.shape[1], stacked_out.shape[2], stacked_out.shape[3]
        out = stacked_out.reshape(-1, n, m)

        # Broadcast each stack's flag over its 64 channels: (N,) -> (64*N,)
        out_active = jnp.repeat(stacked_active, n64)

        return out, out_active

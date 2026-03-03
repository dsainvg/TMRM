import jax
import jax.numpy as jnp
import equinox as eqx

from utils.encoder import Encoder
from utils.config.trainparams import (
    ENCODER_IN_CHANNELS,
    ENCODER_OUT_CHANNELS,
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

        Implementation note
        -------------------
        Completely vmap-free at call time.  Instead of filter_vmap over N roots
        and N×8 leaves, all encoder passes are implemented as explicit batched
        einsums over the N (or N×8) leading axis.  This makes the layer safe
        under any combination of outer jax.vmap / jax.jit transforms because
        XLA only ever sees ordinary batched matrix multiplications — no nested
        vmap axes that trip the HLO shape verifier.
        """
        n_leaves = ENCODER_STAGE2_COUNT   # 8  (leaves per root)
        leaf_out  = ENCODER_OUT_CHANNELS   # 8  (channels per leaf output)

        # Static pairwise combination indices (4C2 = 6 pairs)
        idx1 = jnp.array([0, 0, 0, 1, 1, 2])
        idx2 = jnp.array([1, 2, 3, 2, 3, 3])

        N  = xs.shape[0]
        n  = xs.shape[2]
        m  = xs.shape[3]
        NL = N * n_leaves   # total leaf encoders

        # ── Stage 1: N root encoders (batched einsum, no vmap) ───────────────
        # conv1: (N, in_ch=1) → (N, EXPAND_CH=4)
        # weight shape after N-vmap: (N, 4, 1, 1, 1) → slice to (N, 4, 1)
        # bias shape after N-vmap:   (N, 4, 1, 1)  — already broadcastable over (N, 4, h, w)
        w1 = self.stage1_encs.conv1.weight[..., 0, 0]           # (N, 4, 1)
        b1 = self.stage1_encs.conv1.bias                         # (N, 4, 1, 1)
        s1c1 = (jnp.einsum('noi,nihw->nohw', w1, xs)
                + b1)                                            # (N, 4, n, m)

        # Pairwise matmul: flatten (N, pairs=6) → (N*6) so the matmul sees
        # only ONE pre-existing batch dim.  Under jit+vmap(B), XLA then sees
        # (B, N*6) = 2 batch dims — the max that works without axis-ordering
        # conflicts in XLA's HLO verifier.
        P = idx1.shape[0]  # 6
        left_flat  = s1c1[:, idx1].reshape(N * P, n, m)    # (N*6, n, m)
        right_flat = s1c1[:, idx2].reshape(N * P, n, m)    # (N*6, n, m)
        s1_pairs   = jnp.matmul(left_flat, right_flat).reshape(N, P, n, m)  # (N, 6, n, m)

        # conv2: (N, INTER=10) → (N, INTER_STACK=8)
        # weight after N-vmap: (N, 8, 10, 1, 1) → (N, 8, 10)
        # bias after N-vmap:   (N, 8, 1, 1)
        s1_concat = jnp.concatenate([s1c1, s1_pairs], axis=1)   # (N, 10, n, m)
        w2 = self.stage1_encs.conv2.weight[..., 0, 0]           # (N, 8, 10)
        b2 = self.stage1_encs.conv2.bias                         # (N, 8, 1, 1)
        s1_active = (jnp.einsum('noi,nihw->nohw', w2, s1_concat)
                     + b2)                                       # (N, 8, n, m)

        # Gate: zero inactive stacks via lax.select (jnp.where)
        s1_out = jnp.where(
            is_active_flags[:, None, None, None],
            s1_active,
            jnp.zeros_like(s1_active),
        )                                                          # (N, 8, n, m)

        # ── Prepare leaf inputs ──────────────────────────────────────────────
        # Split root output channels → NL single-channel leaf inputs
        leaf_inputs = s1_out.reshape(NL, 1, n, m)                # (NL, 1, n, m)
        leaf_flags  = jnp.repeat(is_active_flags, n_leaves)      # (NL,) bool

        # ── Stage 2: N×8 leaf encoders (batched einsum over NL, no vmap) ─────
        # Flatten double-vmapped weights/biases (N, 8, ...) → (NL, ...) at each layer
        # conv1 weights/biases:  (N, 8, 4, 1, 1, 1) → (NL, 4, 1, 1, 1) and (N, 8, 4, 1, 1) → (NL, 4, 1, 1)
        # conv2 weights/biases:  (N, 8, 8, 10, 1, 1) → (NL, 8, 10, 1, 1) and (N, 8, 8, 1, 1) → (NL, 8, 1, 1)
        s2w1 = self.stage2_encs.conv1.weight               # (N, 8, 4, 1, 1, 1)
        s2b1 = self.stage2_encs.conv1.bias                 # (N, 8, 4, 1, 1)
        s2w2 = self.stage2_encs.conv2.weight               # (N, 8, 8, 10, 1, 1)
        s2b2 = self.stage2_encs.conv2.bias                 # (N, 8, 8, 1, 1)

        w3 = s2w1.reshape(NL, *s2w1.shape[2:])[..., 0, 0]  # (NL, 4, 1)
        b3 = s2b1.reshape(NL, *s2b1.shape[2:])              # (NL, 4, 1, 1)
        w4 = s2w2.reshape(NL, *s2w2.shape[2:])[..., 0, 0]  # (NL, 8, 10)
        b4 = s2b2.reshape(NL, *s2b2.shape[2:])              # (NL, 8, 1, 1)

        # conv1 → (NL, 4, n, m)
        s2c1 = (jnp.einsum('noi,nihw->nohw', w3, leaf_inputs)
                + b3)

        # Same flat-reshape pairwise matmul for stage-2 leaves (NL instead of N)
        left_flat2  = s2c1[:, idx1].reshape(NL * P, n, m)    # (NL*6, n, m)
        right_flat2 = s2c1[:, idx2].reshape(NL * P, n, m)    # (NL*6, n, m)
        s2_pairs    = jnp.matmul(left_flat2, right_flat2).reshape(NL, P, n, m)  # (NL, 6, n, m)

        # conv2 → (NL, 8, n, m)
        s2_concat = jnp.concatenate([s2c1, s2_pairs], axis=1)   # (NL, 10, n, m)
        s2_active = (jnp.einsum('noi,nihw->nohw', w4, s2_concat)
                     + b4)                                       # (NL, 8, n, m)

        # Gate inactive leaves to zero
        s2_out = jnp.where(
            leaf_flags[:, None, None, None],
            s2_active,
            jnp.zeros_like(s2_active),
        )                                                          # (NL, 8, n, m)

        # ── Assemble final output ────────────────────────────────────────────
        out = s2_out.reshape(NL * leaf_out, n, m)               # (64*N, n, m)

        # Each leaf produces leaf_out=8 channels; all share the leaf's flag
        out_active = jnp.repeat(
            leaf_flags.reshape(N, n_leaves), leaf_out, axis=1
        ).reshape(-1)                                            # (64*N,)

        return out, out_active

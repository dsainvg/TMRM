import math
import jax
import jax.numpy as jnp
import equinox as eqx

from utils.config.trainparams import (
    ENCODER_IN_CHANNELS,
    ENCODER_EXPAND_CHANNELS,
    ENCODER_INTERMEDIATE_CHANNELS,
    ENCODER_OUT_CHANNELS,
)

class Encoder(eqx.Module):
    """
    Encoder node mapping (in_channels, n, n) -> (out_channels, n, n).

    Default configuration: (1, n, n) -> (8, n, n).
    By passing in_channels / out_channels at construction the same node type
    serves as both stage-1 and stage-2 encoders inside an EncoderLayer stack.
    """
    conv1: eqx.nn.Conv2d
    conv2: eqx.nn.Conv2d
    out_channels: int = eqx.field(static=True)

    def __init__(self, key, in_channels: int = ENCODER_IN_CHANNELS, out_channels: int = ENCODER_OUT_CHANNELS):
        # Validate that ENCODER_INTERMEDIATE_CHANNELS matches the expected derivation:
        # EXPAND channels + C(EXPAND, 2) pairwise products = intermediate width.
        _expected_intermediate = ENCODER_EXPAND_CHANNELS + math.comb(ENCODER_EXPAND_CHANNELS, 2)
        assert ENCODER_INTERMEDIATE_CHANNELS == _expected_intermediate, (
            f"Config invariant broken: ENCODER_INTERMEDIATE_CHANNELS={ENCODER_INTERMEDIATE_CHANNELS} "
            f"!= {ENCODER_EXPAND_CHANNELS} + C({ENCODER_EXPAND_CHANNELS},2) = {_expected_intermediate}"
        )
        keys = jax.random.split(key, 2)
        # in_channels-to-4 expansion
        self.conv1 = eqx.nn.Conv2d(in_channels=in_channels, out_channels=ENCODER_EXPAND_CHANNELS, kernel_size=1, key=keys[0])
        # 10-to-out_channels compression
        self.conv2 = eqx.nn.Conv2d(in_channels=ENCODER_INTERMEDIATE_CHANNELS, out_channels=out_channels, kernel_size=1, key=keys[1])
        self.out_channels = out_channels

    def __call__(self, x, is_active):
        """
        x: (1, n, n) single-channel input.  Spatial dims MUST be square (n == n);
           the pairwise batched matmul contracts the inner spatial dimension and
           requires n == m.
        is_active: strict scalar boolean determining execution.

        Always computes the active path; the inactive gate is applied via
        jax.lax.select (jnp.where) which zeroes the result when is_active=False.
        This makes the call fully vmap-compatible — no lax.cond nested inside
        vmap — enabling true SIMD-batched training with jax.vmap.
        Gradients are automatically masked to zero for inactive slots because
        jnp.where VJP multiplies the upstream gradient by where(is_active,1,0).
        """
        # ── Active path (always computed) ─────────────────────────────────────
        # Expand to 4 channels
        x_conv1 = self.conv1(x)  # (4, n, n)

        # Pairwise 4C2 = 6 interactions — explicit index arrays for Batched MatMul
        idx1 = jnp.array([0, 0, 0, 1, 1, 2])
        idx2 = jnp.array([1, 2, 3, 2, 3, 3])

        left  = x_conv1[idx1]  # (6, n, n)
        right = x_conv1[idx2]  # (6, n, n)

        # Batched MatMul over pairs
        pairs = jnp.matmul(left, right)  # (6, n, n)

        # Form 10-channel intermediate representation
        concat = jnp.concatenate([x_conv1, pairs], axis=0)  # (10, n, n)

        # Project down to out_channels
        active_out = self.conv2(concat)  # (out_channels, n, n)

        # ── Gate: zero-out inactive slots via jax.lax.select (XLA select op) ──
        # jnp.where with a scalar condition broadcasts over the array and
        # compiles to a single XLA select — no lax.cond branch, fully vmap-safe.
        _, n, m = x.shape
        out = jnp.where(is_active, active_out, jnp.zeros((self.out_channels, n, m)))
        return out, is_active

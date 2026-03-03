import math
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx

from utils.config.trainparams import (
    DECODER_MAX_PARENTS,
    DECODER_ACTIVATION_THRESHOLD,
    DECODER_TOP_K_EXTRACT,
    DECODER_INTERACT_RANKS,
    DECODER_PRESERVE_RANKS,
    DECODER_INTERMEDIATE_CHANNELS,
    DECODER_HIDDEN_CHANNELS,
    DECODER_OUT_CHANNELS,
)

class Decoder(eqx.Module):
    """
    Decoder Node routing up to 16 input matrices into exactly 1 output.
    """
    conv1: eqx.nn.Conv2d
    conv2: eqx.nn.Conv2d

    def __init__(self, key):
        # Validate derived config invariants at construction time so any
        # accidental config mismatch surfaces immediately with a clear message.
        _expected_preserve = DECODER_TOP_K_EXTRACT - DECODER_INTERACT_RANKS
        assert DECODER_PRESERVE_RANKS == _expected_preserve, (
            f"Config invariant broken: DECODER_PRESERVE_RANKS={DECODER_PRESERVE_RANKS} "
            f"!= DECODER_TOP_K_EXTRACT({DECODER_TOP_K_EXTRACT}) - "
            f"DECODER_INTERACT_RANKS({DECODER_INTERACT_RANKS}) = {_expected_preserve}"
        )
        _expected_intermediate = math.comb(DECODER_INTERACT_RANKS, 2) + DECODER_PRESERVE_RANKS
        assert DECODER_INTERMEDIATE_CHANNELS == _expected_intermediate, (
            f"Config invariant broken: DECODER_INTERMEDIATE_CHANNELS={DECODER_INTERMEDIATE_CHANNELS} "
            f"!= C({DECODER_INTERACT_RANKS},2)+{DECODER_PRESERVE_RANKS} = {_expected_intermediate}"
        )

        keys = jax.random.split(key, 2)
        self.conv1 = eqx.nn.Conv2d(in_channels=DECODER_INTERMEDIATE_CHANNELS, out_channels=DECODER_HIDDEN_CHANNELS, kernel_size=1, key=keys[0])
        self.conv2 = eqx.nn.Conv2d(in_channels=DECODER_HIDDEN_CHANNELS, out_channels=DECODER_OUT_CHANNELS, kernel_size=1, key=keys[1])

    def __call__(self, x, is_active_flags):
        """
        x: (16, n, n) stacked block from previous nodes
        is_active_flags: (16,) boolean determining active inputs
        """
        # Threshold Gate: Must have 12 or more parent signals
        gate_active = jnp.sum(is_active_flags) >= DECODER_ACTIVATION_THRESHOLD
        
        def _active_path(operand):
            inputs, mask = operand
            
            # Extract determinant proxy scores
            # slogdet returns tuple: (sign, logabsdet)
            _, slogdet = jnp.linalg.slogdet(inputs)  # slogdet shape: (16,)
            
            # Mask inactive inputs with -inf so they effectively fall out of top_k
            scores = jnp.where(mask, slogdet, -jnp.inf)
            
            # Take the primary candidates using exact sorting
            values, indices = jax.lax.top_k(scores, DECODER_TOP_K_EXTRACT)
            top_x = inputs[indices]  # (DECODER_TOP_K_EXTRACT, n, n)
            
            # Split interaction features and preserved features.
            # DECODER_PRESERVE_RANKS is used explicitly to make the slice self-documenting
            # and to ensure any config change that breaks the sum invariant is caught here.
            top_interact_x  = top_x[:DECODER_INTERACT_RANKS]                         # (DECODER_INTERACT_RANKS, n, n)
            next_preserve_x = top_x[DECODER_INTERACT_RANKS : DECODER_INTERACT_RANKS + DECODER_PRESERVE_RANKS]  # (DECODER_PRESERVE_RANKS, n, n)

            # Prepare Batched Combinations of top interact features (e.g., 8C2 = 28 combos).
            # np.triu_indices produces compile-time-constant integer arrays — no JAX tracing.
            idx1, idx2 = np.triu_indices(DECODER_INTERACT_RANKS, k=1)
            left = top_interact_x[idx1]        
            right = top_interact_x[idx2]       
            
            # Mix pairs via batched matmul
            pairs = jnp.matmul(left, right)  
            
            # Form terminal representation
            concat = jnp.concatenate([pairs, next_preserve_x], axis=0)  
            
            # Adapt the merged cross-context tensor back to standard channels
            hidden = self.conv1(concat)           # (DECODER_HIDDEN_CHANNELS, n, n)
            out = self.conv2(hidden)              # (1, n, n)
            out = jax.nn.swish(out)               # swish activation (element-wise)
            return out.squeeze(0), jnp.array(True)  # (n, n)
            
        def _inactive_path(operand):
            inputs, _ = operand
            _, n, m = inputs.shape
            # Return identical zero map when network is locally inactive
            return jnp.zeros((n, m)), jnp.array(False)
            
        out, out_active = jax.lax.cond(
            gate_active,
            _active_path,
            _inactive_path,
            (x, is_active_flags)
        )
        return out, out_active

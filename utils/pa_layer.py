import jax
import jax.numpy as jnp
import equinox as eqx

from utils.config.trainparams import PA_DEFAULT_ACTIVATION


# Static dispatch table — resolved once at import time, never traced by JAX.
_ACTIVATIONS: dict = {
    'relu':     jax.nn.relu,
    'gelu':     jax.nn.gelu,
    'tanh':     jnp.tanh,
    'sigmoid':  jax.nn.sigmoid,
    'identity': lambda x: x,
}


class PALayer(eqx.Module):
    """
    Port Adapter: a single 1×1 convolution that reduces decoder output
    channels down to the required task output channels.

    Operates directly on the ``(in_channels, n, n)`` decoder output tensor —
    no flattening required before the adapter.  Spatial structure is
    preserved throughout the reduction; only the channel axis is projected.

    Parameters
    ----------
    in_channels  : int  — number of decoder output nodes (e.g. 15)
    out_channels : int  — target task channels (e.g. 4 for N_CHANNELS_OUT)
    key          : jax.Array  — PRNG key for weight initialisation
    activation   : str  — element-wise output activation (default sigmoid)

    Call signature
    --------------
    x : (in_channels, n, n)

    Returns
    -------
    (out_channels * n * n,)  — flattened spatial output for BCE loss

    Parameter count
    ---------------
    in_channels × out_channels  (conv weights)
    + out_channels              (conv bias)
    = in_channels × out_channels + out_channels

    Example: 15 decoder nodes → 4 output channels  ⟹  15×4 + 4 = 64 params
    """

    conv:        eqx.nn.Conv2d
    activation:  callable = eqx.field(static=True)
    out_channels: int     = eqx.field(static=True)

    def __init__(
        self,
        in_channels:  int,
        out_channels: int,
        key:          jax.Array,
        activation:   str = PA_DEFAULT_ACTIVATION,
    ):
        if activation not in _ACTIVATIONS:
            raise ValueError(
                f"Unknown activation '{activation}'. "
                f"Choose from {list(_ACTIVATIONS.keys())}."
            )
        self.conv        = eqx.nn.Conv2d(in_channels, out_channels, kernel_size=1, key=key)
        self.activation  = _ACTIVATIONS[activation]
        self.out_channels = out_channels

    def __call__(self, x: jax.Array) -> jax.Array:
        """
        x : (in_channels, n, n)
        Returns : (out_channels * n * n,)
        """
        out = self.activation(self.conv(x))   # (out_channels, n, n)
        return out.reshape(-1)                # (out_channels * n * n,)

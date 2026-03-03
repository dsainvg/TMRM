import jax
import jax.numpy as jnp
import equinox as eqx

from utils.config.trainparams import FC_DEFAULT_ACTIVATION


# Static dispatch table — resolved once at import time, never traced by JAX.
_ACTIVATIONS: dict = {
    'relu':     jax.nn.relu,
    'gelu':     jax.nn.gelu,
    'tanh':     jnp.tanh,
    'sigmoid':  jax.nn.sigmoid,
    'identity': lambda x: x,
}


class FCLayer(eqx.Module):
    """
    Fully Connected Layer: a single Linear projection followed by an
    element-wise activation function.

    Sits at the terminal stage of the TMRM DAG, after all Decoder outputs
    have been flattened and concatenated.  All FC nodes are always active —
    no is_active flag propagation is required.

    Call signature:
        x: (in_features,)  — flattened input vector

    Returns:
        (out_features,)
    """

    linear:     eqx.nn.Linear
    activation: callable = eqx.field(static=True)

    def __init__(
        self,
        in_features:  int,
        out_features: int,
        key:          jax.Array,
        activation:   str = FC_DEFAULT_ACTIVATION,
    ):
        if activation not in _ACTIVATIONS:
            raise ValueError(
                f"Unknown activation '{activation}'. "
                f"Choose from {list(_ACTIVATIONS.keys())}."
            )
        self.linear     = eqx.nn.Linear(in_features, out_features, key=key)
        self.activation = _ACTIVATIONS[activation]  # stored as static pytree metadata

    def __call__(self, x: jax.Array) -> jax.Array:
        """
        x: (in_features,)
        Returns: (out_features,)
        """
        return self.activation(self.linear(x))

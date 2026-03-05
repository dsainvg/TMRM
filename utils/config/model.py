"""
Configuration dataclasses for the multi-problem TMRM Model.

These are plain frozen dataclasses (not eqx.Module) — they hold
no trainable arrays, only structural metadata.
"""

from dataclasses import dataclass, field

# Supported PA activations — kept in sync with pa_layer._ACTIVATIONS.
_VALID_ACTIVATIONS = frozenset({'relu', 'gelu', 'tanh', 'sigmoid', 'identity'})


@dataclass(frozen=True)
class ProblemConfig:
    """Description of a single problem port-adapter head.

    Attributes
    ----------
    n_encoders_used : int
        How many of the total encoder slots this problem activates.
        Must be in ``[1, ModelConfig.n_encoders]``.
    pa_out_channels : int
        Number of spatial output channels produced by the Port Adapter.
        The adapter output is ``(pa_out_channels, n, n)`` flattened to
        ``pa_out_channels * n * n`` for the loss function.
    pa_activation : str
        Activation applied after the 1×1 conv projection.
        One of ``'relu'``, ``'gelu'``, ``'tanh'``, ``'sigmoid'``, ``'identity'``.
    """
    n_encoders_used: int
    pa_out_channels: int
    pa_activation: str = 'sigmoid'


@dataclass(frozen=True)
class ModelConfig:
    """Full configuration for a multi-problem TMRM Model.

    Attributes
    ----------
    n : int
        Spatial size — all matrices are (n, n).  Must be >= 1.
    n_encoders : int
        Total encoder slots (shared across all problems).  Each problem
        randomly selects a subset of these at init.
    n_decoder_layers : int
        Target depth (L) for the DecoderCluster.
    max_decoder_nodes : int
        Hard cap on total decoder nodes across all layers.
    problems : tuple[ProblemConfig, ...]
        One entry per problem.  Must contain at least one problem.
    """
    n: int
    n_encoders: int
    n_decoder_layers: int
    max_decoder_nodes: int
    problems: tuple  # tuple[ProblemConfig, ...]

    def __post_init__(self):
        if self.n < 1:
            raise ValueError(f"n must be >= 1, got {self.n}")
        if self.n_encoders < 1:
            raise ValueError(f"n_encoders must be >= 1, got {self.n_encoders}")
        if self.n_decoder_layers < 1:
            raise ValueError(
                f"n_decoder_layers must be >= 1, got {self.n_decoder_layers}"
            )
        if self.max_decoder_nodes < 1:
            raise ValueError(
                f"max_decoder_nodes must be >= 1, got {self.max_decoder_nodes}"
            )
        if not self.problems:
            raise ValueError("problems must contain at least one ProblemConfig")

        for i, p in enumerate(self.problems):
            if not isinstance(p, ProblemConfig):
                raise TypeError(
                    f"problems[{i}] must be a ProblemConfig, got {type(p).__name__}"
                )
            if not (1 <= p.n_encoders_used <= self.n_encoders):
                raise ValueError(
                    f"problems[{i}].n_encoders_used={p.n_encoders_used} "
                    f"not in [1, {self.n_encoders}]"
                )
            if p.pa_out_channels < 1:
                raise ValueError(
                    f"problems[{i}].pa_out_channels must be >= 1, "
                    f"got {p.pa_out_channels}"
                )
            if p.pa_activation not in _VALID_ACTIVATIONS:
                raise ValueError(
                    f"problems[{i}].pa_activation='{p.pa_activation}' "
                    f"not in {sorted(_VALID_ACTIVATIONS)}"
                )

    @property
    def n_problems(self) -> int:
        """Number of problem heads."""
        return len(self.problems)

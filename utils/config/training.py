"""
Configuration dataclass for training the TMRM Model.

Provides sensible defaults for learning rate, batch size, epochs, etc.
Kept as a frozen dataclass — no trainable parameters.
"""

from dataclasses import dataclass

# ── Supported optimiser names ────────────────────────────────────────────────
_VALID_OPTIMISERS = frozenset({'adam', 'adamw', 'sgd', 'rmsprop'})

# ── Supported LR schedule names ─────────────────────────────────────────────
_VALID_SCHEDULES = frozenset({'constant', 'cosine', 'linear', 'warmup_cosine'})


@dataclass(frozen=True)
class TrainingConfig:
    """Hyperparameters for a training run.

    These are *not* model architecture parameters — they control how
    optimisation proceeds.

    Attributes
    ----------
    batch_size : int
        Number of samples per gradient step.  Start with 1 (single-sample)
        and scale up as memory allows.
    n_epochs : int
        Total training epochs.
    learning_rate : float
        Peak / base learning rate for the optimiser.
    optimiser : str
        One of ``'adam'``, ``'adamw'``, ``'sgd'``, ``'rmsprop'``.
    weight_decay : float
        L2 regularisation coefficient (used by ``adamw``; ignored by others).
    lr_schedule : str
        Learning-rate schedule type: ``'constant'``, ``'cosine'``,
        ``'linear'``, ``'warmup_cosine'``.
    warmup_steps : int
        Number of linear warm-up steps before the schedule starts.
        Only relevant when ``lr_schedule='warmup_cosine'``.
    grad_clip_norm : float
        Maximum global gradient L2 norm.  Set to ``0.0`` to disable clipping.
    log_every : int
        Print / log metrics every this many steps.
    seed : int
        Base PRNG seed for the training run (data shuffling, etc.).
    """
    batch_size: int = 1
    n_epochs: int = 50
    learning_rate: float = 3e-4
    optimiser: str = 'adam'
    weight_decay: float = 0.0
    lr_schedule: str = 'constant'
    warmup_steps: int = 0
    grad_clip_norm: float = 0.0
    log_every: int = 10
    seed: int = 0

    def __post_init__(self):
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.n_epochs < 1:
            raise ValueError(f"n_epochs must be >= 1, got {self.n_epochs}")
        if self.learning_rate <= 0:
            raise ValueError(
                f"learning_rate must be > 0, got {self.learning_rate}"
            )
        if self.optimiser not in _VALID_OPTIMISERS:
            raise ValueError(
                f"optimiser='{self.optimiser}' not in {sorted(_VALID_OPTIMISERS)}"
            )
        if self.weight_decay < 0:
            raise ValueError(
                f"weight_decay must be >= 0, got {self.weight_decay}"
            )
        if self.lr_schedule not in _VALID_SCHEDULES:
            raise ValueError(
                f"lr_schedule='{self.lr_schedule}' not in "
                f"{sorted(_VALID_SCHEDULES)}"
            )
        if self.warmup_steps < 0:
            raise ValueError(
                f"warmup_steps must be >= 0, got {self.warmup_steps}"
            )
        if self.grad_clip_norm < 0:
            raise ValueError(
                f"grad_clip_norm must be >= 0, got {self.grad_clip_norm}"
            )
        if self.log_every < 1:
            raise ValueError(f"log_every must be >= 1, got {self.log_every}")

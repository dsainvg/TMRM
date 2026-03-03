"""
Task and data configuration for the 4×4 Sudoku TMRM training run.

Contains all path constants, dataset URLs, and task-specific
dimensionality parameters.  Kept as a frozen dataclass so it can be
passed around like any other config object.
"""

import pathlib
from dataclasses import dataclass, field


@dataclass(frozen=True)
class DataConfig:
    """Paths, URLs, and task-dimensionality constants.

    Attributes
    ----------
    dataset_url : str
        Remote URL from which the .npz dataset is fetched when missing.
    data_dir : pathlib.Path
        Local directory where the dataset file is stored.
    dataset_filename : str
        Name of the .npz file inside ``data_dir``.
    checkpoint_dir : pathlib.Path
        Directory for saving model checkpoints.
    n : int
        Spatial grid size (n×n cells).  4 for the 4×4 Sudoku task.
    n_channels_in : int
        Number of one-hot input channels per cell.
        Digits {0,1,2,3,4} → 5 channels.
    n_channels_out : int
        Number of one-hot output channels per cell.
        Digits {1,2,3,4} → 4 channels.
    """

    dataset_url: str = (
        "https://raw.githubusercontent.com/dsainvg/SUDOKU/main/outputs/dataset_4x4.npz"
    )
    data_dir: pathlib.Path = pathlib.Path("data")
    dataset_filename: str = "dataset_4x4.npz"
    checkpoint_dir: pathlib.Path = pathlib.Path("checkpoints")

    # Task dimensionality ─────────────────────────────────────────────────────
    n: int = 4               # grid size (n×n)
    n_channels_in: int = 5   # digits 0–4 → 5 one-hot channels
    n_channels_out: int = 4  # digits 1–4 → 4 one-hot channels

    # ── Derived properties ────────────────────────────────────────────────────

    @property
    def dataset_path(self) -> pathlib.Path:
        """Full path to the local dataset file."""
        return self.data_dir / self.dataset_filename

    @property
    def fc_out(self) -> int:
        """Flattened FC output size: n_channels_out × n × n."""
        return self.n_channels_out * self.n * self.n


# ── Default instance (used as module-level default) ───────────────────────────
DATA_CFG = DataConfig()

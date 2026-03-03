"""
DataConfig — schema only.  All values live in trainparams.py.
"""

import pathlib
from dataclasses import dataclass


@dataclass(frozen=True)
class DataConfig:
    """Paths, URLs, and task-dimensionality constants.

    All values must be supplied explicitly — see trainparams.py.
    """

    dataset_url:      str
    data_dir:         str
    dataset_filename: str
    checkpoint_dir:   str
    n:                int
    n_channels_in:    int
    n_channels_out:   int

    @property
    def data_dir_path(self) -> pathlib.Path:
        return pathlib.Path(self.data_dir)

    @property
    def checkpoint_dir_path(self) -> pathlib.Path:
        return pathlib.Path(self.checkpoint_dir)

    @property
    def dataset_path(self) -> pathlib.Path:
        return pathlib.Path(self.data_dir) / self.dataset_filename

    @property
    def fc_out(self) -> int:
        """Flattened FC output size: n_channels_out × n × n."""
        return self.n_channels_out * self.n * self.n


"""
Docstring reference: https://numpydoc.readthedocs.io/en/latest/format.html
Python style reference: https://google.github.io/styleguide/pyguide.html
"""

import yaml
from pathlib import Path
from filelock import FileLock
import torch
import pandas as pd
from typing import Any


# ----------------------------------------------------------------------------------------------------------------------
def get_device(
        prefer_mps:
        bool = True
) -> torch.device:
    """
    Return the best available torch device.

    Parameters
    ----------
    prefer_mps : bool
        Whether to prefer Apple Metal Performance Shaders (MPS) over CPU.

    Returns
    -------
    torch.device
        Torch device.

    """

    if prefer_mps and torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ----------------------------------------------------------------------------------------------------------------------
def get_config_from_yaml(
        file_path
) -> Any:
    """
    Get configuration from YAML file.

    Parameters
    ----------
    file_path : str
        Target file path.

    Returns
    -------
    Any
        Loaded YAML file.

    """

    # Load YAML config
    with open(file_path, "r") as f:
        config = yaml.safe_load(f)

    return config


class AutoAppendingDataFrame:
    def __init__(self, path: str, buffer_size: int = 1):
        """
        Auto-saving DataFrame that buffers rows and writes atomically after N batches.
        
        :param path: Path to the Parquet file.
        :param batch_size: Number of rows to buffer before saving.
        """
        self.path = Path(path)
        self.buffer_size = buffer_size
        self.lock = FileLock(str(self.path) + ".lock")
        self.buffer = []
        self.columns = None

    def append(self, df_rows: pd.DataFrame):
        """Append rows to buffer and commit if threshold reached."""
        if self.columns is None:
            self.columns = list(df_rows.columns)

            # checking for column consistency
            if list(df_rows.columns) != self.columns:
                raise ValueError(f"Column mismatch: expected {self.columns}, got {list(df_rows.columns)}")

        self.buffer.append(df_rows)

        if len(self.buffer) >= self.buffer_size:
            self._commit()

    def _commit(self):
        """Commit buffered rows to disk atomically."""
        if not self.buffer:
            return
        # Merge buffer into main DataFrame
        df_new_data = pd.concat(self.buffer, ignore_index=True)
        self.buffer.clear()

        # Concurrent appending
        with self.lock:
            file_exists = self.path.exists() and self.path.stat().st_size > 0
            # If file doesn't exist or is empty, write header once
            df_new_data.to_csv(
                self.path,
                mode="a",                # append
                header=not file_exists,  # write header only on first write
                index=False
            )

    def flush(self):
        """Force commit of any buffered rows."""
        self._commit()

    def view(self) -> pd.DataFrame:
        """Return a copy of the current DataFrame."""
        return self.df.copy()

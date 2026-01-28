"""
Docstring reference: https://numpydoc.readthedocs.io/en/latest/format.html
Python style reference: https://google.github.io/styleguide/pyguide.html
"""

import yaml
import torch
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

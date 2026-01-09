import yaml

import torch


# ----------------------------------------------------------------------------------------------------------------------
def get_device(prefer_mps: bool = True) -> torch.device:
    """
    Return the best available torch device.
    Args:
        prefer_mps (bool): Whether to prefer Apple Metal Performance Shaders (MPS) over CPU.
    """
    if prefer_mps and torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_config_from_yaml(file_path):
    # Load YAML config
    with open(file_path, "r") as f:
        config = yaml.safe_load(f)

    return config



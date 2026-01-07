import torch

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


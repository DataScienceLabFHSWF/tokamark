import torch
import torch.nn as nn
import torch.nn.functional as F

from layer_factory import ComposeLayers


class Encoder(nn.Module):
    """
    layers_spec : list[
        {"name": name, "args":args, kwargs: "kwargs"},
        {...},
        ...
        ]
    """

    def __init__(self, layers_spec: list):
        super().__init__()
        self.encoder = ComposeLayers(layers_spec)
        
    def forward(self, x):
        return self.encoder(x)
        
class Decoder(nn.Module):
    """
    layers_spec : list[
        {"name": name, "args":args, kwargs: "kwargs"},
        {...},
        ...
        ]
    """

    def __init__(self, layers_spec: list):
        super().__init__()
        self.decoder = ComposeLayers(layers_spec)
        
    def forward(self, x):
        return self.decoder(x)

        
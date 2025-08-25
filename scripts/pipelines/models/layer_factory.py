
import torch
import torch.nn as nn

# Define layer wrappers
class Conv1dLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args, **kwargs):
        return nn.Conv1d(*args, **kwargs)

class ConvT1dLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args, **kwargs):
        return nn.ConvTranspose1d(*args, **kwargs)

class LinearLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args, **kwargs):
        return nn.Linear(*args, **kwargs)

class ReluLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        return nn.ReLU()

# Factory for creating layers
class MakeLayer:
    _layer_registry = {
        "conv_1d": Conv1dLayer,
        "convT_1d": ConvT1dLayer,
        "linear_layer": LinearLayer,
        "relu": ReluLayer
    }

    def __init__(self, layer_name):
        if not isinstance(layer_name, str):
            raise TypeError(f"Layer name must be a string, got {type(layer_name).__name__}")
        if layer_name not in self._layer_registry:
            raise ValueError(f"Unknown layer '{layer_name}', available: {list(self._layer_registry.keys())}")
        self.layer_class = self._layer_registry[layer_name]

    def __call__(self, *args, **kwargs):
        return self.layer_class().forward(*args, **kwargs)

# Compose multiple layers into a sequential model
class ComposeLayers(nn.Module):
    def __init__(self, layer_specs: list):
        super().__init__()
        layers = []
        for spec in layer_specs["layers"]:
            if isinstance(spec, dict):
                try:
                    layer_name = spec["name"]
                    args = spec.get("args", [])
                    kwargs = spec.get("kwargs", {})
                    layer = MakeLayer(layer_name)(*args, **kwargs)
                    layers.append(layer)
                except KeyError as e:
                    print(f"{e}")
            else:
                raise ValueError("Each layer spec must be a dict with 'name', optional 'args', and 'kwargs'")
        self.sequence = nn.Sequential(*layers)

    def forward(self, x):
        return self.sequence(x)

 
    
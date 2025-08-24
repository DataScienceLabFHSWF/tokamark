import torch
import torch.nn as nn
import torch.nn.functional as F

from conv1d_vae_model import Conv1DVAE, loss_function
from beta_vae_model import BetaVAE


class Encoder():
    _model_registry = {
        "conv1d_vae": Conv1DVAE,
        "beta_vae": BetaVAE
    }
    
    def __init__(self, model):
        if not isinstance(model, str):
            raise TypeError(f"Model name must be a string, got {type(model).__name__}")
        if model not in self._model_registry:
            raise ValueError(f"Unknown model '{model}', available: {list(self._model_registry.keys())}")
        try:
            self.model_class = self._model_registry[model]
        except Exception as e:
            raise RuntimeError(f"Unexpected error while initializing model '{model}': {e}")

    def __call__(self, *args, **kwargs):
        """Initialize the chosen model with provided args/kwargs"""
        return self.model_class(*args, **kwargs)

def test_conv1dvae():
    
    beta = 1
    in_channels = 10
    input_length = 100
    out_channels = 5
    latent_dim = 3
    kernel_size = 10
    stride = 5
    padding = 0
    factor = 2

    # Create model via Encoder factory
    conv1d_vae = Encoder("conv1d_vae")
    model = conv1d_vae(
        beta=beta,
        in_channels=in_channels,
        input_length=input_length,
        out_channels=out_channels,
        latent_dim=latent_dim,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        factor=factor,
    )

    # Dummy input
    x = torch.randn(16, in_channels, input_length)

    # Forward pass
    x_recon, mu, logvar = model(x)

    # Loss computation
    total_loss, recon_loss, kl_loss = loss_function(beta, x_recon, x, mu, logvar)
    
    # === Assertions ===
    assert x_recon.shape == x.shape, f"Reconstruction shape mismatch: {x_recon.shape} vs {x.shape}"
    assert mu.shape[0] == x.shape[0] and mu.shape[1] == latent_dim, \
        f"Mu shape mismatch: {mu.shape}, expected ({x.shape[0]}, {latent_dim})"
    assert logvar.shape == mu.shape, f"Logvar shape mismatch: {logvar.shape} vs {mu.shape}"
    assert total_loss.dim() == 0, "Total loss should be a scalar"
    assert recon_loss.dim() == 0, "Reconstruction loss should be a scalar"
    assert kl_loss.dim() == 0, "KL loss should be a scalar"

    print(" Conv1DVAE test: ")
    print(f"Input: {x.shape}")
    print(f"Reconstruction: {x_recon.shape}")
    print(f"Mu: {mu.shape}, Logvar: {logvar.shape}")
    print(f"Loss: total={total_loss.item():.4f}, recon={recon_loss.item():.4f}, kl={kl_loss.item():.4f}")


if __name__ == "__main__":
    test_conv1dvae()
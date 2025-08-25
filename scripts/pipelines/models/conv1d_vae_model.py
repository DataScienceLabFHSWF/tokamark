import torch
import torch.nn as nn
import torch.nn.functional as F

from encoder_decoder import Encoder, Decoder

class Conv1dVAE(nn.Module):
    def __init__(self, encoder_layers_specs, decoder_layers_specs, vae_specs):
        super().__init__()
        
        try:
            self.in_channels = encoder_layers_specs["layers"][0]["kwargs"]["in_channels"]
            self.latent_dim = vae_specs["latent_dim"]
            self.input_length= vae_specs["input_length"]

        except KeyError as e:
            print(f"{e}")
            raise ValueError(f"Missing required encoder spec key: {e}")
        
        # =============== Encoder =====================
        self.encoder = Encoder(encoder_layers_specs)
         # Find shape after encoding
        dummy_input = torch.zeros(1, self.in_channels, self.input_length)
        self.encode_out = self.encoder(dummy_input)
        conv_out_dim= self.encode_out.shape[1] * self.encode_out.shape[2] #(out_channels*factor * L_out = 1+ (L_in - kernel_size + 2*padding) / stride)
        
        
        # =============== VAE =====================
        # Linear map from encoder output to the latent space.
        self.fc_mu = nn.Linear(conv_out_dim, self.latent_dim)
        self.fc_logvar = nn.Linear(conv_out_dim, self.latent_dim)


        # =============== Decoder =====================
        self.fc_decode = nn.Linear(self.latent_dim, conv_out_dim)
        self.decoder = Decoder(decoder_layers_specs)
        
    def encode(self, x):
        encoded = self.encoder(x)
        encoded = torch.flatten(encoded, start_dim=1)
        mu = self.fc_mu(encoded)
        logvar = self.fc_logvar(encoded)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        decoded = self.fc_decode(z)
        decoded = decoded.view(decoded.size(0), self.encode_out.shape[1], self.encode_out.shape[2])
        x_recon = self.decoder(decoded)
        return x_recon

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar
        
def loss_function(beta, reconstruction, target, mu, logvar):
    """
    β-VAE loss function
    
    Returns
    -------
    loss : torch.Tensor
        Total loss
    reconstruction_loss : torch.Tensor
        Reconstruction loss component
    kl_loss : torch.Tensor
        KL divergence loss component
    """
    masked_target = target[:,:,:reconstruction.shape[2]]
    reconstruction_loss = F.mse_loss(reconstruction, masked_target, reduction='mean')        
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())        
    total_loss = reconstruction_loss + beta * kl_loss
    return total_loss, reconstruction_loss, kl_loss
    
            

def test_conv1d_vae():
    # Model hyperparameters
    beta = 1
    in_channels = 10
    input_length = 100
    out_channels = 5
    latent_dim = 3
    kernel_size = 10
    stride = 5
    padding = 0

    vae_specs = {"beta":beta, "latent_dim": latent_dim, "input_length":input_length}
    
    # Encoder layer specs
    encoder_layers_specs = {
        "layers": [
            {
                "name": "conv_1d", 
                "kwargs": {
                    "in_channels": in_channels,
                    "out_channels": out_channels,
                    "kernel_size": kernel_size,
                    "stride": stride, 
                    "padding": padding}
            },
            
            {"name": "relu"}
        ]
    }

    # Decoder layer specs
    decoder_layers_specs = {
        "layers": [
            {"name": "convT_1d", 
             "kwargs": {
                "in_channels": out_channels,
                "out_channels":in_channels,
                "kernel_size": kernel_size,
                "stride": stride, 
                "padding": padding}},
            {"name": "relu"}
        ]
    }

    # Create model
    model = Conv1dVAE(encoder_layers_specs, decoder_layers_specs, vae_specs)

    # Dummy input
    x = torch.randn(4, in_channels, input_length)  # batch size = 4

    # Forward pass
    x_recon, mu, logvar = model(x)

    # Compute loss
    loss, recon_loss, kl_loss = loss_function(beta, x_recon, x, mu, logvar)

    # Print results
    print("Reconstructed shape:", x_recon.shape)
    print("Loss:", loss.item())
    print("Reconstruction Loss:", recon_loss.item())
    print("KL Divergence Loss:", kl_loss.item())

if __name__ =="__main__":
    test_conv1d_vae()
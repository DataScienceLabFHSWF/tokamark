import torch
import torch.nn as nn
import torch.nn.functional as F
       
        
class Conv1DVAE(nn.Module):
    def __init__(self,  
                 beta, 
                 in_channels,
                 input_length,
                 out_channels, 
                 latent_dim, 
                 kernel_size, 
                 stride, 
                 padding,
                 factor = 2):
        
        super().__init__()
        self.beta = beta
        self.in_channels = in_channels
        self.input_length = input_length
        self.out_channels = out_channels
        self.latent_dim = latent_dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.factor = factor
        self.scaled_kernel_size = max(1, self.kernel_size // self.factor)
        
        # =============== Encoder =====================
        self.encoder = nn.Sequential(
            nn.Conv1d(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding),
            nn.ReLU(),
            nn.Conv1d(self.out_channels, self.out_channels*self.factor, self.scaled_kernel_size, self.stride, self.padding),
            nn.ReLU(),
        )
    
        # Find shape after Conv1d
        dummy_input = torch.zeros(1, self.in_channels, self.input_length)
        self.encode_out = self.encoder(dummy_input)
        conv_out_dim= self.encode_out.shape[1] * self.encode_out.shape[2] #(out_channels*factor * L_out = 1+ (L_in - kernel_size + 2*padding) / stride)
        
        # Linear map from Conv1d output to the latent space,(multivariate guassian).
        self.fc_mu = nn.Linear(conv_out_dim, latent_dim)
        self.fc_logvar = nn.Linear(conv_out_dim, latent_dim)
        
        # =============== Decoder =====================
        self.fc_decode = nn.Linear(latent_dim, conv_out_dim)
        
        # A linear map from the output of Conv1d to the latent space
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(self.out_channels*self.factor, self.out_channels, self.scaled_kernel_size, self.stride, self.padding),
            nn.ReLU(),
            nn.ConvTranspose1d(self.out_channels, self.in_channels, self.kernel_size, self.stride, self.padding),
            nn.ReLU(),
        )
        
    def encode(self, x):
        h = self.encoder(x)
        h = torch.flatten(h, start_dim=1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.fc_decode(z)
        h = h.view(h.size(0), self.encode_out.shape[1], self.encode_out.shape[2])
        x_recon = self.decoder(h)
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
    reconstruction_loss = F.mse_loss(reconstruction, target, reduction='mean')        
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())        
    total_loss = reconstruction_loss + beta * kl_loss
    return total_loss, reconstruction_loss, kl_loss
    
        
# ===== Example usage =====
if __name__ == "__main__":

    beta = 1
    in_channels = 10
    input_length = 100
    x = torch.randn(100, in_channels, input_length)
    model = Conv1DVAE( 
                    beta, 
                    in_channels,
                    input_length,
                    out_channels = 5, 
                    latent_dim = 3, 
                    kernel_size = 10, 
                    stride = 5, 
                    padding = 0,
                    factor = 2,
                )
    
    
    x_recon, mu, logvar = model(x)
    total_loss, reconstruction_loss, kl_loss = loss_function(beta, x_recon, x, mu, logvar)
    print("Input: ", x.shape)
    print("Reconstruction:", x_recon.shape)
    print("Mu: ", mu.shape)
    print("Logvar: ", logvar.shape)  
    print("total_loss: ", total_loss)    
        
            
        
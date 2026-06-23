import torch
import torch.nn as nn

class LatentNoiseAttack(nn.Module):
    """Applies additive Gaussian noise to latent representations during training.
    
    Args:
        std_min (float): Minimum standard deviation of the noise.
        std_max (float): Maximum standard deviation of the noise.
    """
    def __init__(self, std_min=0.0, std_max=0.1):
        super().__init__()
        self.std_min = std_min
        self.std_max = std_max

    def forward(self, z):
        """Applies random Gaussian noise to latents.
        
        Args:
            z (torch.Tensor): Input latent tensor of shape (B, C, H, W).
            
        Returns:
            torch.Tensor: Noisy latent tensor of shape (B, C, H, W).
        """
        if not self.training:
            return z
            
        B = z.shape[0]
        std = torch.rand(B, 1, 1, 1, device=z.device) * (self.std_max - self.std_min) + self.std_min
        noise = torch.randn_like(z)
        return z + noise * std

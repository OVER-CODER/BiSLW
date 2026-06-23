import torch
import torch.nn as nn
import torch.nn.functional as F

class JpegSimAttack(nn.Module):
    """Simulates JPEG compression artifacts in image space.
    
    Performs spatial decoding, downsampling, bilinear upsampling, and latent re-encoding.
    
    Args:
        vae (nn.Module): VAE Wrapper to decode and encode latent tensors.
        quality_min (int): Minimum simulated quality factor.
        quality_max (int): Maximum simulated quality factor.
    """
    def __init__(self, vae, quality_min=50, quality_max=100):
        super().__init__()
        self.vae = vae
        self.quality_min = quality_min
        self.quality_max = quality_max

    def forward(self, z):
        """Applies simulated JPEG compression to latents.
        
        Args:
            z (torch.Tensor): Input latent tensor of shape (B, C, H, W).
            
        Returns:
            torch.Tensor: Attacked latent tensor of shape (B, C, H, W).
        """
        if not self.training:
            return z
            
        images = self.vae.decode(z)
        B, C, H, W = images.shape
        
        # Downsample and upsample to simulate high-frequency losses
        scale = torch.rand(1, device=z.device).item() * 0.5 + 0.5
        h_small, w_small = int(H * scale), int(W * scale)
        
        images_down = F.interpolate(images, size=(h_small, w_small), mode='bilinear', align_corners=False)
        images_up = F.interpolate(images_down, size=(H, W), mode='bilinear', align_corners=False)
        
        z_attacked = self.vae.encode(images_up)
        return z_attacked

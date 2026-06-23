import torch
import torch.nn as nn
import torch.nn.functional as F

class ResizeCropAttack(nn.Module):
    """Applies crop and resize attack through spatial decoding and re-encoding.
    
    Args:
        vae (nn.Module): VAE Wrapper to decode and encode latent tensors.
        scale_min (float): Minimum crop scaling factor.
        scale_max (float): Maximum crop scaling factor.
    """
    def __init__(self, vae, scale_min=0.8, scale_max=1.0):
        super().__init__()
        self.vae = vae
        self.scale_min = scale_min
        self.scale_max = scale_max

    def forward(self, z):
        """Applies spatial cropping and resizing to latents.
        
        Args:
            z (torch.Tensor): Input latent tensor of shape (B, C, H, W).
            
        Returns:
            torch.Tensor: Attacked latent tensor of shape (B, C, H, W).
        """
        if not self.training:
            return z
            
        images = self.vae.decode(z)
        B, C, H, W = images.shape
        
        scale = torch.rand(1, device=z.device).item() * (self.scale_max - self.scale_min) + self.scale_min
        new_size = int(min(H, W) * scale)
        
        if new_size < H or new_size < W:
            top = torch.randint(0, H - new_size + 1, (1,)).item()
            left = torch.randint(0, W - new_size + 1, (1,)).item()
            
            cropped = images[:, :, top:top+new_size, left:left+new_size]
            images_attacked = F.interpolate(cropped, size=(H, W), mode='bilinear', align_corners=False)
        else:
            images_attacked = images
            
        z_attacked = self.vae.encode(images_attacked)
        return z_attacked

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResizeCropAttack(nn.Module):
    """
    Resize and Crop attack.
    """
    def __init__(self, vae, scale_min=0.8, scale_max=1.0):
        super().__init__()
        self.vae = vae
        self.scale_min = scale_min
        self.scale_max = scale_max

    def forward(self, z):
        if not self.training:
            return z
            
        # Decode -> Attack -> Encode
        images = self.vae.decode(z)
        B, C, H, W = images.shape
        
        # Random scale
        scale = torch.rand(1, device=z.device).item() * (self.scale_max - self.scale_min) + self.scale_min
        new_size = int(min(H, W) * scale)
        
        # Random crop
        if new_size < H or new_size < W:
            # Resize first? Or crop then resize?
            # "Resize + crop" usually means resize to something, then crop, or crop then resize back.
            # Let's do: Random Crop -> Resize back to original (to keep tensor size const)
            
            top = torch.randint(0, H - new_size + 1, (1,)).item()
            left = torch.randint(0, W - new_size + 1, (1,)).item()
            
            cropped = images[:, :, top:top+new_size, left:left+new_size]
            images_attacked = F.interpolate(cropped, size=(H, W), mode='bilinear', align_corners=False)
        else:
            images_attacked = images
            
        z_attacked = self.vae.encode(images_attacked)
        return z_attacked

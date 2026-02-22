import torch
import torch.nn as nn

class WatermarkDecoder(nn.Module):
    """
    Recovers watermark from a latent band.
    Uses global pooling (no spatial reliance).
    """
    def __init__(self, input_channels=4, watermark_dim=64, hidden_dim=64):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, stride=2), # Downsample
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, stride=2), # Downsample
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((1, 1)), # Global pooling
            nn.Flatten(),
            nn.Linear(hidden_dim, watermark_dim)
        )

    def forward(self, z):
        """
        Args:
            z: (B, C, H, W) latent band
        Returns:
            w_pred: (B, w_dim)
        """
        return self.net(z)

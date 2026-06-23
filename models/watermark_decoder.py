import torch
import torch.nn as nn
import torch.nn.functional as F

class WatermarkDecoder(nn.Module):
    """Standard watermark decoder using average pooling.
    
    Extracts the watermark from a latent band using global average pooling.
    
    Args:
        input_channels (int): Number of input latent channels.
        watermark_dim (int): Dimension of the output watermark vector.
        hidden_dim (int): Number of hidden features channels.
    """
    def __init__(self, input_channels=4, watermark_dim=64, hidden_dim=64):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, stride=2),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, stride=2),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(hidden_dim, watermark_dim)
        )

    def forward(self, z):
        """Forward pass for watermark extraction.
        
        Args:
            z (torch.Tensor): Latent band tensor of shape (B, C, H, W).
            
        Returns:
            torch.Tensor: Extracted watermark prediction of shape (B, w_dim).
        """
        return self.net(z)


class RobustWatermarkDecoder(nn.Module):
    """Spatially-invariant watermark decoder with multi-scale global pooling.
    
    Extracts global statistical moments (mean and standard deviation) across
    multiple parallel convolutional pathways to achieve robustness against
    geometric transformations like cropping and rotation.
    
    Args:
        input_channels (int): Number of input latent channels.
        watermark_dim (int): Dimension of the output watermark vector.
        hidden_dim (int): Number of hidden features channels.
    """
    def __init__(self, input_channels=4, watermark_dim=32, hidden_dim=32):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        self.channel_expand = nn.Sequential(
            nn.Conv2d(input_channels, hidden_dim, kernel_size=1),
            nn.SiLU(),
        )
        
        # Path 1: 1x1 Convolution mapping
        self.path1 = nn.Sequential(
            nn.Conv2d(input_channels, hidden_dim, kernel_size=1),
            nn.SiLU(),
        )
        
        # Path 2: 3x3 Convolution mapping
        self.path2 = nn.Sequential(
            nn.Conv2d(input_channels, hidden_dim, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        
        # Path 3: Deeper 3x3 mapping for local context
        self.path3 = nn.Sequential(
            nn.Conv2d(input_channels, hidden_dim, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        
        # Feature dimensions: (mean + std) for each path
        total_features = 3 * hidden_dim * 2 + input_channels * 2
        
        self.mlp = nn.Sequential(
            nn.Linear(total_features, hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, watermark_dim)
        )
        
    def _global_stats(self, x):
        """Extracts spatially invariant global channel mean and standard deviation."""
        mean = x.mean(dim=(2, 3))
        std = x.std(dim=(2, 3)) + 1e-6
        return torch.cat([mean, std], dim=1)
    
    def forward(self, z):
        """Forward pass for robust watermark extraction.
        
        Args:
            z (torch.Tensor): Latent band tensor of shape (B, C, H, W).
            
        Returns:
            torch.Tensor: Extracted watermark prediction of shape (B, w_dim).
        """
        stats_input = self._global_stats(z)
        stats_p1 = self._global_stats(self.path1(z))
        stats_p2 = self._global_stats(self.path2(z))
        stats_p3 = self._global_stats(self.path3(z))
        
        features = torch.cat([stats_input, stats_p1, stats_p2, stats_p3], dim=1)
        
        return self.mlp(features)

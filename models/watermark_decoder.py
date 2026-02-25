import torch
import torch.nn as nn
import torch.nn.functional as F


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


class RobustWatermarkDecoder(nn.Module):
    """
    Spatially-invariant watermark decoder with multi-scale global pooling.
    
    Key improvements for crop/rotation robustness:
    1. Early global pooling at multiple scales
    2. Channel statistics (mean + std) instead of just mean
    3. Parallel extraction paths that are location-independent
    """
    def __init__(self, input_channels=4, watermark_dim=32, hidden_dim=32):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Initial feature extraction (1x1 conv - fully location independent)
        self.channel_expand = nn.Sequential(
            nn.Conv2d(input_channels, hidden_dim, kernel_size=1),
            nn.SiLU(),
        )
        
        # Multi-scale global pooling paths
        # Each path: Conv -> GlobalPool -> features
        
        # Path 1: Direct global stats from input
        self.path1 = nn.Sequential(
            nn.Conv2d(input_channels, hidden_dim, kernel_size=1),
            nn.SiLU(),
        )
        
        # Path 2: 3x3 conv then global pool
        self.path2 = nn.Sequential(
            nn.Conv2d(input_channels, hidden_dim, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        
        # Path 3: Deeper features with spatial reduction
        self.path3 = nn.Sequential(
            nn.Conv2d(input_channels, hidden_dim, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.SiLU(),
        )
        
        # Each path produces: mean + std = 2 * hidden_dim
        # 3 paths = 6 * hidden_dim total
        # Plus original input stats: 2 * input_channels
        total_features = 3 * hidden_dim * 2 + input_channels * 2
        
        # MLP to combine all global features
        self.mlp = nn.Sequential(
            nn.Linear(total_features, hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, watermark_dim)
        )
        
    def _global_stats(self, x):
        """Extract global mean and std (spatially invariant)."""
        mean = x.mean(dim=(2, 3))  # (B, C)
        std = x.std(dim=(2, 3)) + 1e-6  # (B, C)
        return torch.cat([mean, std], dim=1)  # (B, 2C)
    
    def forward(self, z):
        """
        Args:
            z: (B, C, H, W) latent band
        Returns:
            w_pred: (B, w_dim)
        """
        # Get global stats from each path
        stats_input = self._global_stats(z)  # 2 * input_channels
        stats_p1 = self._global_stats(self.path1(z))  # 2 * hidden_dim
        stats_p2 = self._global_stats(self.path2(z))  # 2 * hidden_dim
        stats_p3 = self._global_stats(self.path3(z))  # 2 * hidden_dim
        
        # Concatenate all global features
        features = torch.cat([stats_input, stats_p1, stats_p2, stats_p3], dim=1)
        
        # MLP to predict watermark
        return self.mlp(features)


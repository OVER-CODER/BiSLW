import torch
import torch.nn as nn

class WatermarkEncoder(nn.Module):
    """
    Injects watermark into a latent band.
    """
    def __init__(self, input_channels=4, watermark_dim=64, hidden_dim=64):
        super().__init__()
        # "Encoders must be lightweight but expressive"
        # Maps (z, w) -> delta
        # We concatenate z and w (broadcasted)
        
        self.net = nn.Sequential(
            nn.Conv2d(input_channels + watermark_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, input_channels, kernel_size=1)
        )
        
        # Initialize last layer to zero for stability (start with no change)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, z, w, alpha=1.0):
        """
        Args:
            z: (B, C, H, W) latent band
            w: (B, w_dim) watermark vector
            alpha: scalar strength
        Returns:
            z_watermarked: z + alpha * delta
        """
        B, C, H, W = z.shape
        w_expanded = w.view(B, -1, 1, 1).expand(B, -1, H, W)
        
        inp = torch.cat([z, w_expanded], dim=1)
        delta = self.net(inp)
        
        return z + alpha * delta

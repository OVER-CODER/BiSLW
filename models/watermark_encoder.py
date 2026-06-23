import torch
import torch.nn as nn

class WatermarkEncoder(nn.Module):
    """Embeds a watermark message vector into a target frequency band.
    
    Uses a convolutional residual network to project and inject the watermark
    into the latent band with controlled strength alpha.
    
    Args:
        input_channels (int): Number of channels in the latent band.
        watermark_dim (int): Dimension of the watermark vector.
        hidden_dim (int): Number of hidden convolutional channels.
    """
    def __init__(self, input_channels=4, watermark_dim=64, hidden_dim=64):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Conv2d(input_channels + watermark_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, input_channels, kernel_size=1)
        )
        
        # Zero-initialize the final convolution projection layer for training stability
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, z, w, alpha=1.0):
        """Forward pass for watermark injection.
        
        Args:
            z (torch.Tensor): Latent band tensor of shape (B, C, H, W).
            w (torch.Tensor): Watermark vector of shape (B, w_dim).
            alpha (float): Scaling factor controlling injection strength.
            
        Returns:
            torch.Tensor: Watermarked latent band of shape (B, C, H, W).
        """
        B, C, H, W = z.shape
        w_expanded = w.view(B, -1, 1, 1).expand(B, -1, H, W)
        
        inp = torch.cat([z, w_expanded], dim=1)
        delta = self.net(inp)
        
        return z + alpha * delta

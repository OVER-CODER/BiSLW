import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class LatentSplitter(nn.Module):
    """Decomposes latent representations into low-frequency and high-frequency bands.
    
    Supports Discrete Cosine Transform (DCT) and learned channel projection modes.
    
    Args:
        mode (str): Spectral decomposition method ('dct' or 'learned').
        channels (int): Number of input latent channels.
        learned_split_ratio (float): Split ratio for learned mode channels.
    """
    def __init__(self, mode='dct', channels=4, learned_split_ratio=0.5):
        super().__init__()
        self.mode = mode
        self.channels = channels
        
        if mode == 'learned':
            # Project channels to split low-frequency and high-frequency components
            self.splitter = nn.Conv2d(channels, channels * 2, kernel_size=1)
        elif mode == 'dct':
            pass
        else:
            raise ValueError(f"Unknown split mode: {mode}")

    def forward(self, z):
        """Forward pass for spectral decomposition.
        
        Args:
            z (torch.Tensor): Input latent tensor of shape (B, C, H, W).
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - z_low (torch.Tensor): Low-frequency component of shape (B, C, H, W).
                - z_high (torch.Tensor): High-frequency component of shape (B, C, H, W).
        """
        if self.mode == 'dct':
            return self._dct_split(z)
        else:
            return self._learned_split(z)

    def _dct_split(self, z):
        """Splits the latent tensor using 2D DCT-II.
        
        Extracts the top-left quadrant (50% in each spatial dimension) 
        as the low-frequency component encoding global semantics, leaving
        the remaining quadrants as high-frequency textures.
        """
        B, C, H, W = z.shape
        
        # Retrieve orthonormal DCT-II transformation matrices
        dct_h = self._get_dct_matrix(H, z.device)
        dct_w = self._get_dct_matrix(W, z.device)
        
        # Project spatial latent to frequency domain: Z_freq = D_h * Z * D_w^T
        z_freq = torch.einsum('ij,b c j k -> b c i k', dct_h, z)
        z_freq = torch.einsum('b c i k, jk -> b c i j', z_freq, dct_w)
        
        # Apply binary frequency quadrant mask
        mask = torch.zeros_like(z_freq)
        h_cut, w_cut = H // 2, W // 2
        mask[:, :, :h_cut, :w_cut] = 1.0
        
        z_freq_low = z_freq * mask
        z_freq_high = z_freq * (1.0 - mask)
        
        return z_freq_low, z_freq_high

    def _learned_split(self, z):
        """Splits the latent tensor using a learned 1x1 convolution."""
        features = self.splitter(z)
        z_low, z_high = torch.chunk(features, 2, dim=1)
        return z_low, z_high

    def _get_dct_matrix(self, N, device):
        """Generates the orthonormal 1D DCT-II matrix of size N."""
        n = torch.arange(N, device=device).float()
        k = torch.arange(N, device=device).float()
        dct_mat = torch.cos((math.pi / N) * (n + 0.5) * k.unsqueeze(1))
        dct_mat[0] *= 1.0 / math.sqrt(2)
        dct_mat *= math.sqrt(2 / N)
        return dct_mat

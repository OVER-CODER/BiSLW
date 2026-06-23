import math
import torch
import torch.nn as nn

class LatentRecombiner(nn.Module):
    """Recombines low-frequency and high-frequency components back into a spatial latent.
    
    Inverse operation of LatentSplitter.
    
    Args:
        mode (str): Spectral decomposition method ('dct' or 'learned').
        channels (int): Number of input latent channels.
    """
    def __init__(self, mode='dct', channels=4):
        super().__init__()
        self.mode = mode
        self.channels = channels
        
        if mode == 'learned':
            # Project concatenated features back to original channel space
            self.combiner = nn.Conv2d(channels * 2, channels, kernel_size=1)
        elif mode == 'dct':
            pass

    def forward(self, z_low, z_high):
        """Forward pass for recombination.
        
        Args:
            z_low (torch.Tensor): Low-frequency latent component.
            z_high (torch.Tensor): High-frequency latent component.
            
        Returns:
            torch.Tensor: Recombined spatial latent tensor of shape (B, C, H, W).
        """
        if self.mode == 'dct':
            return self._dct_combine(z_low, z_high)
        else:
            return self._learned_combine(z_low, z_high)

    def _dct_combine(self, z_low, z_high):
        """Reconstructs the spatial latent using 2D Inverse DCT-II.
        
        Summation of z_low and z_high reconstructs the full frequency spectrum,
        which is then projected back to the spatial domain.
        """
        z_freq = z_low + z_high
        B, C, H, W = z_freq.shape
        
        # Retrieve orthonormal DCT matrices
        dct_h = self._get_dct_matrix(H, z_freq.device)
        dct_w = self._get_dct_matrix(W, z_freq.device)
        
        # Project back to spatial domain: Z = D_h^T * Z_freq * D_w
        z = torch.einsum('ij,b c j k -> b c i k', dct_h.t(), z_freq)
        z = torch.einsum('b c i k, jk -> b c i j', z, dct_w.t())
        
        return z

    def _learned_combine(self, z_low, z_high):
        """Recombines components using channel concatenation and 1x1 projection."""
        combined = torch.cat([z_low, z_high], dim=1)
        return self.combiner(combined)

    def _get_dct_matrix(self, N, device):
        """Generates the orthonormal 1D DCT-II matrix of size N."""
        n = torch.arange(N, device=device).float()
        k = torch.arange(N, device=device).float()
        dct_mat = torch.cos((math.pi / N) * (n + 0.5) * k.unsqueeze(1))
        dct_mat[0] *= 1.0 / math.sqrt(2)
        dct_mat *= math.sqrt(2 / N)
        return dct_mat

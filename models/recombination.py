import torch
import torch.nn as nn
import math

class LatentRecombiner(nn.Module):
    """
    Recombines low and high frequency components.
    Inverse of LatentSplitter.
    """
    def __init__(self, mode='dct', channels=4):
        super().__init__()
        self.mode = mode
        self.channels = channels
        
        if mode == 'learned':
            # Inverse of the 1x1 split (which was a projection)
            # We need to map back from 2*channels to channels.
            self.combiner = nn.Conv2d(channels * 2, channels, kernel_size=1)
        elif mode == 'dct':
            pass

    def forward(self, z_low, z_high):
        """
        Args:
            z_low: Low freq component
            z_high: High freq component
        Returns:
            z_full: Recombined latent
        """
        if self.mode == 'dct':
            return self._dct_combine(z_low, z_high)
        else:
            return self._learned_combine(z_low, z_high)

    def _dct_combine(self, z_low, z_high):
        # z_low and z_high are in frequency domain and masked (or should be summed if they are disjoint masks)
        # Since we masked them, z_low + z_high reconstructs the full frequency spectrum.
        z_freq = z_low + z_high
        
        B, C, H, W = z_freq.shape
        
        # Inverse DCT
        # Z = D_h^T * Z_freq * D_w
        
        dct_h = self._get_dct_matrix(H, z_freq.device)
        dct_w = self._get_dct_matrix(W, z_freq.device)
        
        # Transpose DCT matrices for inverse
        idct_h = dct_h.t()
        idct_w = dct_w.t()
        
        z = torch.einsum('ij,b c j k -> b c i k', idct_h, z_freq)
        z = torch.einsum('b c i k, jk -> b c i j', z, idct_w)
        
        return z

    def _learned_combine(self, z_low, z_high):
        # Concatenate and project back
        combined = torch.cat([z_low, z_high], dim=1)
        return self.combiner(combined)

    def _get_dct_matrix(self, N, device):
        # Same as in Splitter, could be shared utility
        n = torch.arange(N, device=device).float()
        k = torch.arange(N, device=device).float()
        dct_mat = torch.cos((math.pi / N) * (n + 0.5) * k.unsqueeze(1))
        dct_mat[0] *= 1.0 / math.sqrt(2)
        dct_mat *= math.sqrt(2 / N)
        return dct_mat

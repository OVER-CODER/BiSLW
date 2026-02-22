import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LatentSplitter(nn.Module):
    """
    Splits latents into low and high frequency components.
    Supports 'dct' and 'learned' modes.
    """
    def __init__(self, mode='dct', channels=4, learned_split_ratio=0.5):
        super().__init__()
        self.mode = mode
        self.channels = channels
        
        if mode == 'learned':
            # 1x1 conv to split channels into low and high features
            # We maintain the same number of channels for both outputs to simplify downstream
            # But in practice, we might want to project to a different space.
            # Here we implement a simple channel split or projection.
            # Requirement: "Learned spectral decomposition (1x1 conv split)"
            # Let's project to 2 * channels and split.
            self.splitter = nn.Conv2d(channels, channels * 2, kernel_size=1)
        elif mode == 'dct':
            pass # DCT is parameter-free
        else:
            raise ValueError(f"Unknown split mode: {mode}")

    def forward(self, z):
        """
        Args:
            z: (B, C, H, W) latent
        Returns:
            z_low: Low frequency component
            z_high: High frequency component
        """
        if self.mode == 'dct':
            return self._dct_split(z)
        else:
            return self._learned_split(z)

    def _dct_split(self, z):
        """
        Splits using DCT. Low freq is the top-left corner of DCT spectrum.
        """
        B, C, H, W = z.shape
        # Simple 2D DCT implementation
        # We can use torch.fft.rfft2 but DCT is requested.
        # For simplicity and differentiability, we can implement a block-based DCT or full DCT.
        # Given "Global semantics" vs "Fine texture", a global DCT makes sense but is heavy.
        # Let's use a simple frequency mask in Fourier domain as a proxy for "spectral decomposition" 
        # if strict DCT is too slow, but let's try to do a real DCT-II.
        
        # Using an orthonormal transform matrix for H and W
        dct_h = self._get_dct_matrix(H, z.device)
        dct_w = self._get_dct_matrix(W, z.device)
        
        # Z_freq = D_h * Z * D_w^T
        z_freq = torch.einsum('ij,b c j k -> b c i k', dct_h, z)
        z_freq = torch.einsum('b c i k, jk -> b c i j', z_freq, dct_w)
        
        # Split: Low freq is the top-left quadrant (or some ratio)
        # Let's take the top-left 50% (by area, so 0.707 by dimension) or just half H/W?
        # Requirement says "Low-frequency component: Encodes global semantics".
        # Let's use a soft mask or hard crop. Hard crop is easier to reason about for "split".
        # But we need to return tensors of same shape or handle shapes.
        # To keep it simple and allow "Recombination", let's return the full frequency tensor masked.
        
        # Actually, "z_low" and "z_high" usually imply we separate the information.
        # Let's mask.
        mask = torch.zeros_like(z_freq)
        h_cut, w_cut = H // 2, W // 2
        mask[:, :, :h_cut, :w_cut] = 1.0
        
        z_freq_low = z_freq * mask
        z_freq_high = z_freq * (1.0 - mask)
        
        # Inverse DCT to get back to spatial domain? 
        # "Treat latent as a signal... decompose it".
        # If we return frequency domain, the watermarking happens in freq domain.
        # "Watermark is injected independently into both bands... and then recombined".
        # Usually easier to inject in the domain where they are separated.
        # So we return the frequency representations.
        
        return z_freq_low, z_freq_high

    def _learned_split(self, z):
        features = self.splitter(z)
        z_low, z_high = torch.chunk(features, 2, dim=1)
        return z_low, z_high

    def _get_dct_matrix(self, N, device):
        # DCT-II matrix
        n = torch.arange(N, device=device).float()
        k = torch.arange(N, device=device).float()
        dct_mat = torch.cos((math.pi / N) * (n + 0.5) * k.unsqueeze(1))
        dct_mat[0] *= 1.0 / math.sqrt(2)
        dct_mat *= math.sqrt(2 / N)
        return dct_mat

"""Watermark training loss functions.

Combines recovery loss, cross-band consistency loss, latent preservation loss,
robustness loss, and perceptual similarity loss (VGG/LPIPS).
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class WatermarkLosses(nn.Module):
    """Container for calculating all individual and combined loss terms.

    Loss components:
    1. Watermark Recovery Loss: MSE between predicted and true watermarks.
    2. Cross-Band Consistency Loss: Agreement between L and H band predictions.
    3. Latent Preservation Loss: MSE between original and watermarked latents.
    4. Robustness Loss: Recovery accuracy under simulated attacks.
    5. Perceptual Loss: VGG feature-level similarity.
    """
    
    def __init__(
        self,
        lambda_w: float = 1.0,
        lambda_cons: float = 1.0,
        lambda_latent: float = 1.0, 
        lambda_robust: float = 1.0,
        lambda_perceptual: float = 0.0,
        device: str = 'cuda'
    ):
        """Initializes the WatermarkLosses module.

        Args:
            lambda_w (float): Weight for watermark recovery loss.
            lambda_cons (float): Weight for cross-band consistency loss.
            lambda_latent (float): Weight for latent space preservation loss.
            lambda_robust (float): Weight for robustness recovery loss.
            lambda_perceptual (float): Weight for VGG perceptual loss.
            device (str): Targeted device for VGG initialization.
        """
        super().__init__()
        self.lambda_w = lambda_w
        self.lambda_cons = lambda_cons
        self.lambda_latent = lambda_latent
        self.lambda_robust = lambda_robust
        self.lambda_perceptual = lambda_perceptual
        
        if lambda_perceptual > 0:
            self._init_perceptual_net(device)
        else:
            self.perceptual_net = None
            
    def _init_perceptual_net(self, device: str):
        """Initializes pre-trained VGG features up to relu3_3 for perceptual loss.

        Args:
            device (str): Device to mount the perceptual network.
        """
        try:
            from torchvision.models import vgg16, VGG16_Weights
            vgg = vgg16(weights=VGG16_Weights.DEFAULT)
            self.perceptual_net = nn.Sequential(*list(vgg.features[:16])).to(device)
            self.perceptual_net.eval()
            for param in self.perceptual_net.parameters():
                param.requires_grad = False
            
            self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        except Exception as e:
            print(f"Warning: Could not initialize perceptual network: {e}")
            self.perceptual_net = None
            
    def _normalize_for_vgg(self, x: torch.Tensor) -> torch.Tensor:
        """Normalizes an image tensor from range [-1, 1] to Imagenet distribution.

        Args:
            x (torch.Tensor): Image tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Normalized image tensor.
        """
        x = (x + 1) / 2
        return (x - self.mean.to(x.device)) / self.std.to(x.device)
        
    def perceptual_loss(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """Computes feature-level MSE loss between two image tensors.

        Args:
            img1 (torch.Tensor): First image tensor of shape (B, C, H, W).
            img2 (torch.Tensor): Second image tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Perceptual loss scalar.
        """
        if self.perceptual_net is None:
            return torch.tensor(0.0, device=img1.device)
            
        img1_norm = self._normalize_for_vgg(img1)
        img2_norm = self._normalize_for_vgg(img2)
        
        with torch.no_grad():
            feat1 = self.perceptual_net(img1_norm)
        feat2 = self.perceptual_net(img2_norm)
        
        return F.mse_loss(feat2, feat1)

    def forward(
        self,
        w_true: torch.Tensor,
        w_pred_L: torch.Tensor,
        w_pred_H: torch.Tensor,
        z_orig: torch.Tensor,
        z_watermarked: torch.Tensor, 
        w_pred_robust_L: Optional[torch.Tensor] = None,
        w_pred_robust_H: Optional[torch.Tensor] = None,
        img_orig: Optional[torch.Tensor] = None,
        img_watermarked: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Computes weighted aggregate loss and extracts breakdown stats dictionary.

        Args:
            w_true (torch.Tensor): Ground truth watermark of shape (B, W_dim).
            w_pred_L (torch.Tensor): Extraction prediction from low frequency band.
            w_pred_H (torch.Tensor): Extraction prediction from high frequency band.
            z_orig (torch.Tensor): Original latent tensor.
            z_watermarked (torch.Tensor): Embedded watermarked latent tensor.
            w_pred_robust_L (Optional[torch.Tensor]): Low band prediction under attack.
            w_pred_robust_H (Optional[torch.Tensor]): High band prediction under attack.
            img_orig (Optional[torch.Tensor]): Decoded original image.
            img_watermarked (Optional[torch.Tensor]): Decoded watermarked image.

        Returns:
            Tuple[torch.Tensor, Dict[str, float]]: Aggregate loss scalar and logging dict.
        """
        l_w = F.mse_loss(w_pred_L, w_true) + F.mse_loss(w_pred_H, w_true)
        l_cons = F.mse_loss(w_pred_L, w_pred_H)
        l_latent = F.mse_loss(z_watermarked, z_orig)
        
        l_robust = torch.tensor(0.0, device=z_orig.device)
        if w_pred_robust_L is not None and w_pred_robust_H is not None:
            l_robust = F.mse_loss(w_pred_robust_L, w_true) + F.mse_loss(w_pred_robust_H, w_true)
            
        l_perceptual = torch.tensor(0.0, device=z_orig.device)
        if self.lambda_perceptual > 0 and img_orig is not None and img_watermarked is not None:
            l_perceptual = self.perceptual_loss(img_orig, img_watermarked)
            
        total_loss = (self.lambda_w * l_w +
                      self.lambda_cons * l_cons +
                      self.lambda_latent * l_latent +
                      self.lambda_robust * l_robust +
                      self.lambda_perceptual * l_perceptual)
                      
        return total_loss, {
            "loss_w": l_w.item(),
            "loss_cons": l_cons.item(),
            "loss_latent": l_latent.item(),
            "loss_robust": l_robust.item() if isinstance(l_robust, torch.Tensor) else float(l_robust),
            "loss_perceptual": l_perceptual.item() if isinstance(l_perceptual, torch.Tensor) else float(l_perceptual)
        }

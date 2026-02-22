import torch
import torch.nn as nn
import torch.nn.functional as F

class WatermarkLosses(nn.Module):
    """
    Container for all loss functions.
    
    Loss components:
    1. Watermark Recovery Loss: MSE between predicted and true watermarks
    2. Cross-Band Consistency Loss: Agreement between L and H band predictions
    3. Latent Preservation Loss: MSE between original and watermarked latents
    4. Robustness Loss: Recovery accuracy after attacks
    5. Perceptual Loss: LPIPS-style perceptual similarity
    """
    def __init__(self, lambda_w=1.0, lambda_cons=1.0, lambda_latent=1.0, 
                 lambda_robust=1.0, lambda_perceptual=0.0, device='cuda'):
        super().__init__()
        self.lambda_w = lambda_w
        self.lambda_cons = lambda_cons
        self.lambda_latent = lambda_latent
        self.lambda_robust = lambda_robust
        self.lambda_perceptual = lambda_perceptual
        
        # Perceptual loss network (VGG features)
        if lambda_perceptual > 0:
            self._init_perceptual_net(device)
        else:
            self.perceptual_net = None
            
    def _init_perceptual_net(self, device):
        """Initialize VGG for perceptual loss."""
        try:
            from torchvision.models import vgg16, VGG16_Weights
            vgg = vgg16(weights=VGG16_Weights.DEFAULT)
            # Use features up to relu3_3
            self.perceptual_net = nn.Sequential(*list(vgg.features[:16])).to(device)
            self.perceptual_net.eval()
            for param in self.perceptual_net.parameters():
                param.requires_grad = False
            
            # Normalization for VGG
            self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        except Exception as e:
            print(f"Warning: Could not initialize perceptual network: {e}")
            self.perceptual_net = None
            
    def _normalize_for_vgg(self, x):
        """Normalize from [-1, 1] to VGG input range."""
        x = (x + 1) / 2  # [-1, 1] -> [0, 1]
        return (x - self.mean.to(x.device)) / self.std.to(x.device)
        
    def perceptual_loss(self, img1, img2):
        """Compute perceptual loss between two images."""
        if self.perceptual_net is None:
            return torch.tensor(0.0, device=img1.device)
            
        # Normalize
        img1_norm = self._normalize_for_vgg(img1)
        img2_norm = self._normalize_for_vgg(img2)
        
        # Extract features
        with torch.no_grad():
            feat1 = self.perceptual_net(img1_norm)
        feat2 = self.perceptual_net(img2_norm)
        
        return F.mse_loss(feat2, feat1)

    def forward(self, w_true, w_pred_L, w_pred_H, z_orig, z_watermarked, 
                w_pred_robust_L=None, w_pred_robust_H=None,
                img_orig=None, img_watermarked=None):
        """
        Compute combined loss.
        
        Args:
            w_true: (B, w_dim) true watermark
            w_pred_L: (B, w_dim) prediction from low freq
            w_pred_H: (B, w_dim) prediction from high freq
            z_orig: (B, C, H, W) original latent
            z_watermarked: (B, C, H, W) watermarked latent
            w_pred_robust_L: (B, w_dim) prediction after attack (low)
            w_pred_robust_H: (B, w_dim) prediction after attack (high)
            img_orig: (B, C, H, W) original image for perceptual loss
            img_watermarked: (B, C, H, W) watermarked image for perceptual loss
        """
        # 1. Watermark Recovery Loss
        l_w = F.mse_loss(w_pred_L, w_true) + F.mse_loss(w_pred_H, w_true)
        
        # 2. Cross-Band Consistency Loss
        l_cons = F.mse_loss(w_pred_L, w_pred_H)
        
        # 3. Latent Preservation Loss
        l_latent = F.mse_loss(z_watermarked, z_orig)
        
        # 4. Robustness Loss
        l_robust = torch.tensor(0.0, device=z_orig.device)
        if w_pred_robust_L is not None and w_pred_robust_H is not None:
            l_robust = F.mse_loss(w_pred_robust_L, w_true) + F.mse_loss(w_pred_robust_H, w_true)
            
        # 5. Perceptual Loss
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
            "loss_robust": l_robust.item() if isinstance(l_robust, torch.Tensor) else l_robust,
            "loss_perceptual": l_perceptual.item() if isinstance(l_perceptual, torch.Tensor) else l_perceptual
        }

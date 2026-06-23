import math
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class MetricResult:
    """Container for metric evaluation results with stats."""
    mean: float
    std: float
    values: List[float]
    name: str
    
    def __repr__(self):
        return f"{self.name}: {self.mean:.4f} ± {self.std:.4f}"


class PSNR(nn.Module):
    """Peak Signal-to-Noise Ratio (PSNR) calculation.
    
    Target: 40 dB for imperceptible watermarking.
    """
    def __init__(self, max_val: float = 1.0, data_range: float = 2.0):
        super().__init__()
        self.max_val = max_val
        self.data_range = data_range
        
    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        mse = torch.mean((img1 - img2) ** 2, dim=[1, 2, 3])
        mse = torch.clamp(mse, min=1e-10)
        return 10 * torch.log10((self.data_range ** 2) / mse)


class SSIM(nn.Module):
    """Structural Similarity Index Measure (SSIM) calculation.
    
    Target: 0.91+ for high-quality watermarking.
    """
    def __init__(self, window_size: int = 11, sigma: float = 1.5, data_range: float = 2.0,
                 K1: float = 0.01, K2: float = 0.03, channel: int = 3):
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.data_range = data_range
        self.C1 = (K1 * data_range) ** 2
        self.C2 = (K2 * data_range) ** 2
        self.C3 = self.C2 / 2
        self.register_buffer('window', self._create_window(window_size, sigma, channel))
        
    def _create_window(self, window_size: int, sigma: float, channel: int) -> torch.Tensor:
        coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
        gauss = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        gauss = gauss / gauss.sum()
        window = (gauss.unsqueeze(1) @ gauss.unsqueeze(0)).unsqueeze(0).unsqueeze(0)
        return window.expand(channel, 1, window_size, window_size).contiguous()
        
    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        B, C, H, W = img1.shape
        window = self.window.to(img1.device)
        
        mu1 = F.conv2d(img1, window, padding=self.window_size//2, groups=C)
        mu2 = F.conv2d(img2, window, padding=self.window_size//2, groups=C)
        
        mu1_sq, mu2_sq, mu1_mu2 = mu1**2, mu2**2, mu1 * mu2
        
        sigma1_sq = torch.clamp(F.conv2d(img1 * img1, window, padding=self.window_size//2, groups=C) - mu1_sq, min=0)
        sigma2_sq = torch.clamp(F.conv2d(img2 * img2, window, padding=self.window_size//2, groups=C) - mu2_sq, min=0)
        sigma12 = F.conv2d(img1 * img2, window, padding=self.window_size//2, groups=C) - mu1_mu2
        
        numerator = (2 * mu1_mu2 + self.C1) * (2 * sigma12 + self.C2)
        denominator = (mu1_sq + mu2_sq + self.C1) * (sigma1_sq + sigma2_sq + self.C2)
        
        return (numerator / (denominator + 1e-8)).mean(dim=[1, 2, 3])


class LPIPS(nn.Module):
    """Learned Perceptual Image Patch Similarity (LPIPS) using pre-trained VGG-16."""
    def __init__(self, net: str = 'vgg', pretrained: bool = True):
        super().__init__()
        self.net = net
        if net == 'vgg':
            from torchvision.models import vgg16, VGG16_Weights
            vgg = vgg16(weights=VGG16_Weights.DEFAULT if pretrained else None)
            vgg.eval()
            for param in vgg.parameters():
                param.requires_grad = False
            
            self.layers = nn.ModuleList([
                vgg.features[:4],
                vgg.features[4:9],
                vgg.features[9:16],
                vgg.features[16:23],
                vgg.features[23:30]
            ])
            self.register_buffer('weights', torch.ones(5) / 5)
        else:
            raise NotImplementedError(f"Network {net} not implemented")
            
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        
    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        x = (x + 1) / 2
        return (x - self.mean.to(x.device)) / self.std.to(x.device)
        
    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        x1 = self._normalize(img1)
        x2 = self._normalize(img2)
        
        lpips_vals = []
        feat1, feat2 = x1, x2
        for i, layer in enumerate(self.layers):
            feat1 = layer(feat1)
            feat2 = layer(feat2)
            
            diff = (F.normalize(feat1, dim=1) - F.normalize(feat2, dim=1)) ** 2
            lpips_vals.append(self.weights[i] * diff.mean(dim=[2, 3]).mean(dim=1))
            
        return torch.stack(lpips_vals, dim=0).sum(dim=0)


class FID(nn.Module):
    """Fréchet Inception Distance (FID) calculation using Inception-v3."""
    def __init__(self, dims: int = 2048):
        super().__init__()
        self.dims = dims
        from torchvision.models import inception_v3, Inception_V3_Weights
        self.inception = inception_v3(weights=Inception_V3_Weights.DEFAULT, transform_input=False)
        self.inception.eval()
        self.inception.fc = nn.Identity()
        for param in self.inception.parameters():
            param.requires_grad = False
            
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
            
    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        x = (x + 1) / 2
        return (x - self.mean.to(x.device)) / self.std.to(x.device)
        
    @torch.no_grad()
    def compute_features(self, images: torch.Tensor) -> torch.Tensor:
        images = F.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)
        images = self._normalize(images)
        return self.inception(images)
        
    def compute_statistics(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu = features.mean(dim=0)
        features_centered = features - mu.unsqueeze(0)
        sigma = (features_centered.T @ features_centered) / (features.shape[0] - 1)
        return mu, sigma
        
    def compute_fid(self, mu1: torch.Tensor, sigma1: torch.Tensor, mu2: torch.Tensor, sigma2: torch.Tensor) -> float:
        mu1 = mu1.cpu().numpy()
        mu2 = mu2.cpu().numpy()
        sigma1 = sigma1.cpu().numpy()
        sigma2 = sigma2.cpu().numpy()
        
        diff = mu1 - mu2
        from scipy.linalg import sqrtm
        covmean = sqrtm(sigma1 @ sigma2)
        if np.iscomplexobj(covmean):
            covmean = covmean.real
            
        return float(diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean))
        
    @torch.no_grad()
    def forward(self, real_images: torch.Tensor, generated_images: torch.Tensor, batch_size: int = 32) -> float:
        real_features, gen_features = [], []
        for i in range(0, len(real_images), batch_size):
            real_features.append(self.compute_features(real_images[i:i+batch_size]))
            gen_features.append(self.compute_features(generated_images[i:i+batch_size]))
            
        real_features = torch.cat(real_features, dim=0)
        gen_features = torch.cat(gen_features, dim=0)
        
        mu1, sigma1 = self.compute_statistics(real_features)
        mu2, sigma2 = self.compute_statistics(gen_features)
        
        return self.compute_fid(mu1, sigma1, mu2, sigma2)


class SIFID(nn.Module):
    """Single Image Fréchet Inception Distance (SIFID)."""
    def __init__(self, dims: int = 2048):
        super().__init__()
        self.dims = dims
        from torchvision.models import inception_v3, Inception_V3_Weights
        self.inception = inception_v3(weights=Inception_V3_Weights.DEFAULT, transform_input=False)
        self.inception.eval()
        self.inception.fc = nn.Identity()
        for param in self.inception.parameters():
            param.requires_grad = False
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        x = (x + 1) / 2
        return (x - self.mean.to(x.device)) / self.std.to(x.device)

    @torch.no_grad()
    def compute_features(self, image: torch.Tensor) -> torch.Tensor:
        image = F.interpolate(image, size=(299, 299), mode='bilinear', align_corners=False)
        image = self._normalize(image)
        return self.inception(image)

    @torch.no_grad()
    def forward(self, real_image: torch.Tensor, generated_image: torch.Tensor) -> float:
        real_feat = self.compute_features(real_image.unsqueeze(0))
        gen_feat = self.compute_features(generated_image.unsqueeze(0))
        diff = real_feat.squeeze(0) - gen_feat.squeeze(0)
        return float(diff @ diff)


class ImageQualityMetrics(nn.Module):
    """Unified manager for evaluating multiple image quality metrics in batches."""
    def __init__(self, device: torch.device = None):
        super().__init__()
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.psnr = PSNR(data_range=2.0)
        self.ssim = SSIM(data_range=2.0)
        self.lpips = None
        self.fid = None
        self.sifid = None
        
    def _ensure_lpips(self):
        if self.lpips is None:
            self.lpips = LPIPS().to(self.device)
            
    def _ensure_fid(self):
        if self.fid is None:
            self.fid = FID().to(self.device)

    def _ensure_sifid(self):
        if self.sifid is None:
            self.sifid = SIFID().to(self.device)
            
    @torch.no_grad()
    def compute_psnr(self, original: torch.Tensor, watermarked: torch.Tensor) -> MetricResult:
        values = self.psnr(original, watermarked)
        return MetricResult(
            mean=float(values.mean()),
            std=float(values.std()),
            values=values.cpu().tolist(),
            name="PSNR (dB)"
        )
        
    @torch.no_grad()
    def compute_ssim(self, original: torch.Tensor, watermarked: torch.Tensor) -> MetricResult:
        values = self.ssim.to(original.device)(original, watermarked)
        return MetricResult(
            mean=float(values.mean()),
            std=float(values.std()),
            values=values.cpu().tolist(),
            name="SSIM"
        )
        
    @torch.no_grad()
    def compute_lpips(self, original: torch.Tensor, watermarked: torch.Tensor) -> MetricResult:
        self._ensure_lpips()
        values = self.lpips(original.to(self.device), watermarked.to(self.device))
        return MetricResult(
            mean=float(values.mean()),
            std=float(values.std()),
            values=values.cpu().tolist(),
            name="LPIPS"
        )

    @torch.no_grad()
    def compute_sifid(self, original: torch.Tensor, watermarked: torch.Tensor) -> MetricResult:
        self._ensure_sifid()
        values = []
        for i in range(original.shape[0]):
            val = self.sifid(original[i], watermarked[i])
            values.append(val)
        values = torch.tensor(values)
        return MetricResult(
            mean=float(values.mean()),
            std=float(values.std()),
            values=values.cpu().tolist(),
            name="SIFID"
        )
        
    @torch.no_grad()
    def compute_fid(self, original: torch.Tensor, watermarked: torch.Tensor, batch_size: int = 32) -> float:
        self._ensure_fid()
        return self.fid(original.to(self.device), watermarked.to(self.device), batch_size=batch_size)
        
    @torch.no_grad()
    def evaluate(self, original: torch.Tensor, watermarked: torch.Tensor, compute_fid: bool = True, batch_size: int = 32) -> Dict[str, any]:
        original = original.to(self.device)
        watermarked = watermarked.to(self.device)
        
        results = {
            'psnr': self.compute_psnr(original, watermarked),
            'ssim': self.compute_ssim(original, watermarked),
            'lpips': self.compute_lpips(original, watermarked)
        }
        
        if compute_fid and len(original) >= 10:
            results['fid'] = self.compute_fid(original, watermarked, batch_size)
        else:
            results['fid'] = None
            
        return results
        
    def print_results(self, results: Dict[str, any]):
        print("\n" + "=" * 50)
        print("Image Quality Metrics")
        print("=" * 50)
        for k in ['psnr', 'ssim', 'lpips']:
            if k in results and results[k] is not None:
                print(f"  {results[k]}")
        if 'fid' in results and results['fid'] is not None:
            print(f"  FID: {results['fid']:.4f}")
        print("=" * 50)

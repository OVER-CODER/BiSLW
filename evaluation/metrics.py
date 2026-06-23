"""
Image Quality Metrics for Watermark Imperceptibility Evaluation.

Implements:
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)
- LPIPS (Learned Perceptual Image Patch Similarity)
- FID (Fréchet Inception Distance)

All metrics support batched evaluation with mean and standard deviation reporting.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, List
import math
from dataclasses import dataclass


@dataclass
class MetricResult:
    """Container for metric results with statistics."""
    mean: float
    std: float
    values: List[float]
    name: str
    
    def __repr__(self):
        return f"{self.name}: {self.mean:.4f} ± {self.std:.4f}"


class PSNR(nn.Module):
    """
    Peak Signal-to-Noise Ratio.
    
    Formula:
        PSNR = 10 * log10(MAX^2 / MSE)
        
    where:
        - MAX is the maximum possible pixel value (1.0 for normalized images)
        - MSE is the Mean Squared Error between original and watermarked images
        
    Higher PSNR indicates better quality (less distortion).
    Target: 40 dB for imperceptible watermarking.
    """
    def __init__(self, max_val: float = 1.0, data_range: float = 2.0):
        """
        Args:
            max_val: Maximum pixel value (1.0 for [0,1], 255 for [0,255])
            data_range: Range of pixel values (2.0 for [-1,1], 1.0 for [0,1])
        """
        super().__init__()
        self.max_val = max_val
        self.data_range = data_range
        
    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """
        Compute PSNR between two images.
        
        Args:
            img1: Original images (B, C, H, W) in range [-1, 1] or [0, 1]
            img2: Watermarked images (B, C, H, W) in range [-1, 1] or [0, 1]
            
        Returns:
            psnr: (B,) tensor of PSNR values in dB
        """
        # Compute MSE per image
        mse = torch.mean((img1 - img2) ** 2, dim=[1, 2, 3])
        
        # Avoid log(0) by clamping MSE
        mse = torch.clamp(mse, min=1e-10)
        
        # PSNR formula with data_range
        psnr = 10 * torch.log10((self.data_range ** 2) / mse)
        
        return psnr


class SSIM(nn.Module):
    """
    Structural Similarity Index Measure.
    
    Formula:
        SSIM(x, y) = [l(x,y)]^α * [c(x,y)]^β * [s(x,y)]^γ
        
    where:
        l(x,y) = (2*μx*μy + C1) / (μx² + μy² + C1)  [luminance]
        c(x,y) = (2*σx*σy + C2) / (σx² + σy² + C2)  [contrast]
        s(x,y) = (σxy + C3) / (σx*σy + C3)          [structure]
        
        C1 = (K1*L)², C2 = (K2*L)², C3 = C2/2
        L = dynamic range, K1 = 0.01, K2 = 0.03
        α = β = γ = 1 (default)
    
    Higher SSIM (closer to 1.0) indicates better structural preservation.
    Target: 0.91+ for high-quality watermarking.
    """
    def __init__(
        self,
        window_size: int = 11,
        sigma: float = 1.5,
        data_range: float = 2.0,
        K1: float = 0.01,
        K2: float = 0.03,
        channel: int = 3
    ):
        """
        Args:
            window_size: Size of the Gaussian window
            sigma: Standard deviation of Gaussian window
            data_range: Dynamic range of pixel values
            K1, K2: Stability constants
            channel: Number of color channels
        """
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.data_range = data_range
        self.C1 = (K1 * data_range) ** 2
        self.C2 = (K2 * data_range) ** 2
        self.C3 = self.C2 / 2
        
        # Create Gaussian window
        self.register_buffer('window', self._create_window(window_size, sigma, channel))
        
    def _create_window(self, window_size: int, sigma: float, channel: int) -> torch.Tensor:
        """Create a Gaussian window for SSIM computation."""
        # 1D Gaussian
        coords = torch.arange(window_size, dtype=torch.float32) - window_size // 2
        gauss = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        gauss = gauss / gauss.sum()
        
        # 2D Gaussian
        window = gauss.unsqueeze(1) @ gauss.unsqueeze(0)
        window = window.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        
        # Repeat for all channels
        window = window.expand(channel, 1, window_size, window_size).contiguous()
        
        return window
        
    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """
        Compute SSIM between two images.
        
        Args:
            img1: Original images (B, C, H, W)
            img2: Watermarked images (B, C, H, W)
            
        Returns:
            ssim: (B,) tensor of SSIM values
        """
        B, C, H, W = img1.shape
        
        # Ensure window is on same device
        window = self.window.to(img1.device)
        
        # Compute means with Gaussian weighting
        mu1 = F.conv2d(img1, window, padding=self.window_size//2, groups=C)
        mu2 = F.conv2d(img2, window, padding=self.window_size//2, groups=C)
        
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        
        # Compute variances and covariance
        sigma1_sq = F.conv2d(img1 * img1, window, padding=self.window_size//2, groups=C) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=self.window_size//2, groups=C) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=self.window_size//2, groups=C) - mu1_mu2
        
        # Clamp variances to be non-negative
        sigma1_sq = torch.clamp(sigma1_sq, min=0)
        sigma2_sq = torch.clamp(sigma2_sq, min=0)
        
        # SSIM formula
        numerator = (2 * mu1_mu2 + self.C1) * (2 * sigma12 + self.C2)
        denominator = (mu1_sq + mu2_sq + self.C1) * (sigma1_sq + sigma2_sq + self.C2)
        
        ssim_map = numerator / (denominator + 1e-8)
        
        # Mean SSIM per image
        ssim = ssim_map.mean(dim=[1, 2, 3])
        
        return ssim


class LPIPS(nn.Module):
    """
    Learned Perceptual Image Patch Similarity.
    
    Uses pre-trained VGG-16 features to compute perceptual similarity.
    
    Formula:
        LPIPS(x, y) = Σ_l w_l * ||φ_l(x) - φ_l(y)||²
        
    where:
        - φ_l is the feature extractor at layer l
        - w_l are learned weights per layer
    
    Lower LPIPS indicates better perceptual quality.
    """
    def __init__(self, net: str = 'vgg', pretrained: bool = True):
        """
        Args:
            net: Network architecture ('vgg', 'alex', 'squeeze')
            pretrained: Use pretrained weights
        """
        super().__init__()
        self.net = net
        
        # Use VGG-16 as default
        if net == 'vgg':
            from torchvision.models import vgg16, VGG16_Weights
            vgg = vgg16(weights=VGG16_Weights.DEFAULT if pretrained else None)
            vgg.eval()
            for param in vgg.parameters():
                param.requires_grad = False
            
            # Extract features at specific layers
            self.layers = nn.ModuleList([
                vgg.features[:4],   # relu1_2
                vgg.features[4:9],  # relu2_2
                vgg.features[9:16], # relu3_3
                vgg.features[16:23], # relu4_3
                vgg.features[23:30]  # relu5_3
            ])
            
            # Learned weights (simplification: use equal weights)
            self.register_buffer('weights', torch.ones(5) / 5)
        else:
            raise NotImplementedError(f"Network {net} not implemented")
            
        # ImageNet normalization
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        
    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize from [-1, 1] to ImageNet stats."""
        # First convert from [-1, 1] to [0, 1]
        x = (x + 1) / 2
        # Then normalize with ImageNet stats
        return (x - self.mean.to(x.device)) / self.std.to(x.device)
        
    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """
        Compute LPIPS between two images.
        
        Args:
            img1: Original images (B, C, H, W) in range [-1, 1]
            img2: Watermarked images (B, C, H, W) in range [-1, 1]
            
        Returns:
            lpips: (B,) tensor of LPIPS values
        """
        # Normalize
        x1 = self._normalize(img1)
        x2 = self._normalize(img2)
        
        lpips_vals = []
        
        # Extract features at each layer
        feat1, feat2 = x1, x2
        for i, layer in enumerate(self.layers):
            feat1 = layer(feat1)
            feat2 = layer(feat2)
            
            # Normalize features
            f1_norm = F.normalize(feat1, dim=1)
            f2_norm = F.normalize(feat2, dim=1)
            
            # Compute squared difference
            diff = (f1_norm - f2_norm) ** 2
            
            # Mean over spatial dimensions, weighted sum over channels
            diff = diff.mean(dim=[2, 3]).mean(dim=1)  # (B,)
            
            lpips_vals.append(self.weights[i] * diff)
            
        # Sum over layers
        lpips = torch.stack(lpips_vals, dim=0).sum(dim=0)
        
        return lpips


class FID(nn.Module):
    """
    Fréchet Inception Distance.
    
    Measures the distance between feature distributions of real and generated images.
    
    Formula:
        FID = ||μ_r - μ_g||² + Tr(Σ_r + Σ_g - 2*(Σ_r*Σ_g)^(1/2))
        
    where:
        - μ_r, μ_g are mean feature vectors of real and generated images
        - Σ_r, Σ_g are covariance matrices of features
    
    Lower FID indicates better quality (more similar to real images).
    """
    def __init__(self, dims: int = 2048):
        """
        Args:
            dims: Feature dimensions from Inception-v3
        """
        super().__init__()
        self.dims = dims
        
        # Load Inception-v3
        from torchvision.models import inception_v3, Inception_V3_Weights
        self.inception = inception_v3(weights=Inception_V3_Weights.DEFAULT, transform_input=False)
        self.inception.eval()
        
        # Remove classification head - we want features before final FC
        self.inception.fc = nn.Identity()
        
        for param in self.inception.parameters():
            param.requires_grad = False
            
        # ImageNet normalization
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
            
    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize from [-1, 1] to ImageNet stats."""
        x = (x + 1) / 2
        return (x - self.mean.to(x.device)) / self.std.to(x.device)
        
    @torch.no_grad()
    def compute_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract Inception features from images.
        
        Args:
            images: (B, C, H, W) in range [-1, 1]
            
        Returns:
            features: (B, dims)
        """
        # Resize to 299x299 for Inception
        images = F.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)
        images = self._normalize(images)
        
        # Get features
        features = self.inception(images)
        
        return features
        
    def compute_statistics(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute mean and covariance of features.
        
        Args:
            features: (N, dims)
            
        Returns:
            mu: (dims,)
            sigma: (dims, dims)
        """
        mu = features.mean(dim=0)
        
        # Centered features
        features_centered = features - mu.unsqueeze(0)
        
        # Covariance matrix
        sigma = (features_centered.T @ features_centered) / (features.shape[0] - 1)
        
        return mu, sigma
        
    def compute_fid(
        self,
        mu1: torch.Tensor,
        sigma1: torch.Tensor,
        mu2: torch.Tensor,
        sigma2: torch.Tensor,
        eps: float = 1e-6
    ) -> float:
        """
        Compute FID from precomputed statistics.
        
        Args:
            mu1, sigma1: Statistics of real images
            mu2, sigma2: Statistics of generated images
            eps: Small constant for numerical stability
            
        Returns:
            fid: FID score
        """
        # Convert to numpy for matrix square root
        mu1 = mu1.cpu().numpy()
        mu2 = mu2.cpu().numpy()
        sigma1 = sigma1.cpu().numpy()
        sigma2 = sigma2.cpu().numpy()
        
        # Compute squared difference of means
        diff = mu1 - mu2
        
        # Product of covariances
        covmean = self._sqrtm(sigma1 @ sigma2)
        
        # Handle numerical errors
        if np.iscomplexobj(covmean):
            covmean = covmean.real
            
        # FID formula
        fid = diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean)
        
        return float(fid)
        
    def _sqrtm(self, A: np.ndarray) -> np.ndarray:
        """Compute matrix square root."""
        from scipy.linalg import sqrtm
        return sqrtm(A)
        
    @torch.no_grad()
    def forward(
        self,
        real_images: torch.Tensor,
        generated_images: torch.Tensor,
        batch_size: int = 32
    ) -> float:
        """
        Compute FID between two sets of images.
        
        Args:
            real_images: (N, C, H, W) original images
            generated_images: (N, C, H, W) watermarked images
            batch_size: Batch size for feature extraction
            
        Returns:
            fid: FID score
        """
        # Extract features in batches
        real_features = []
        gen_features = []
        
        for i in range(0, len(real_images), batch_size):
            batch_real = real_images[i:i+batch_size]
            batch_gen = generated_images[i:i+batch_size]
            
            real_features.append(self.compute_features(batch_real))
            gen_features.append(self.compute_features(batch_gen))
            
        real_features = torch.cat(real_features, dim=0)
        gen_features = torch.cat(gen_features, dim=0)
        
        # Compute statistics
        mu1, sigma1 = self.compute_statistics(real_features)
        mu2, sigma2 = self.compute_statistics(gen_features)
        
        # Compute FID
        return self.compute_fid(mu1, sigma1, mu2, sigma2)


class ImageQualityMetrics(nn.Module):
    """
    Unified interface for all image quality metrics.
    
    Supports batched evaluation with mean and standard deviation reporting.
    """
    def __init__(self, device: torch.device = None):
        """
        Args:
            device: Device to run metrics on
        """
        super().__init__()
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.psnr = PSNR(data_range=2.0)
        self.ssim = SSIM(data_range=2.0)
        self.lpips = None  # Lazy load
        self.fid = None    # Lazy load
        
    def _ensure_lpips(self):
        if self.lpips is None:
            self.lpips = LPIPS().to(self.device)
            
    def _ensure_fid(self):
        if self.fid is None:
            self.fid = FID().to(self.device)

    def _ensure_sifid(self):
        if not hasattr(self, 'sifid') or self.sifid is None:
            self.sifid = SIFID().to(self.device)
            class SIFID(nn.Module):
                """
                Single Image Fréchet Inception Distance (SIFID).
                Measures the distance between feature distributions of a single real and generated image.
                """
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
                    features = self.inception(image)
                    return features

                @torch.no_grad()
                def forward(self, real_image: torch.Tensor, generated_image: torch.Tensor) -> float:
                    real_feat = self.compute_features(real_image.unsqueeze(0))
                    gen_feat = self.compute_features(generated_image.unsqueeze(0))
                    mu1 = real_feat.squeeze(0)
                    mu2 = gen_feat.squeeze(0)
                    diff = mu1 - mu2
                    return float(diff @ diff)
                @torch.no_grad()
                def compute_sifid(self, original: torch.Tensor, watermarked: torch.Tensor) -> MetricResult:
                    self._ensure_sifid()
                    values = []
                    for i in range(original.shape[0]):
                        val = self.sifid(original[i], watermarked[i])
                        values.append(val)
                    values = torch.tensor(values)
                    values_list = values.cpu().tolist()
                    return MetricResult(
                        mean=float(values.mean()),
                        std=float(values.std()),
                        values=values_list,
                        name="SIFID"
                    )
            
    @torch.no_grad()
    def compute_psnr(
        self,
        original: torch.Tensor,
        watermarked: torch.Tensor
    ) -> MetricResult:
        """
        Compute PSNR between original and watermarked images.
        
        Args:
            original: (B, C, H, W) original images
            watermarked: (B, C, H, W) watermarked images
            
        Returns:
            MetricResult with mean, std, and all values
        """
        values = self.psnr(original, watermarked)
        values_list = values.cpu().tolist()
        
        return MetricResult(
            mean=float(values.mean()),
            std=float(values.std()),
            values=values_list,
            name="PSNR (dB)"
        )
        
    @torch.no_grad()
    def compute_ssim(
        self,
        original: torch.Tensor,
        watermarked: torch.Tensor
    ) -> MetricResult:
        """
        Compute SSIM between original and watermarked images.
        
        Args:
            original: (B, C, H, W) original images
            watermarked: (B, C, H, W) watermarked images
            
        Returns:
            MetricResult with mean, std, and all values
        """
        values = self.ssim.to(original.device)(original, watermarked)
        values_list = values.cpu().tolist()
        
        return MetricResult(
            mean=float(values.mean()),
            std=float(values.std()),
            values=values_list,
            name="SSIM"
        )
        
    @torch.no_grad()
    def compute_lpips(
        self,
        original: torch.Tensor,
        watermarked: torch.Tensor
    ) -> MetricResult:
        """
        Compute LPIPS between original and watermarked images.
        
        Args:
            original: (B, C, H, W) original images
            watermarked: (B, C, H, W) watermarked images
            
        Returns:
            MetricResult with mean, std, and all values
        """
        self._ensure_lpips()
        values = self.lpips(original.to(self.device), watermarked.to(self.device))
        values_list = values.cpu().tolist()
        
        return MetricResult(
            mean=float(values.mean()),
            std=float(values.std()),
            values=values_list,
            name="LPIPS"
        )
        
    @torch.no_grad()
    def compute_fid(
        self,
        original: torch.Tensor,
        watermarked: torch.Tensor,
        batch_size: int = 32
    ) -> float:
        """
        Compute FID between original and watermarked image sets.
        
        Args:
            original: (N, C, H, W) original images
            watermarked: (N, C, H, W) watermarked images
            batch_size: Batch size for feature extraction
            
        Returns:
            FID score (single value)
        """
        self._ensure_fid()
        return self.fid(
            original.to(self.device),
            watermarked.to(self.device),
            batch_size=batch_size
        )
        
    @torch.no_grad()
    def evaluate(
        self,
        original: torch.Tensor,
        watermarked: torch.Tensor,
        compute_fid: bool = True,
        batch_size: int = 32
    ) -> Dict[str, any]:
        """
        Run all metrics on a batch of images.
        
        Args:
            original: (B, C, H, W) original images in range [-1, 1]
            watermarked: (B, C, H, W) watermarked images in range [-1, 1]
            compute_fid: Whether to compute FID (slow for small batches)
            batch_size: Batch size for FID computation
            
        Returns:
            Dictionary with all metric results
        """
        results = {}
        
        # Move to device
        original = original.to(self.device)
        watermarked = watermarked.to(self.device)
        
        # Compute metrics
        results['psnr'] = self.compute_psnr(original, watermarked)
        results['ssim'] = self.compute_ssim(original, watermarked)
        results['lpips'] = self.compute_lpips(original, watermarked)
        
        if compute_fid and len(original) >= 10:  # FID needs sufficient samples
            results['fid'] = self.compute_fid(original, watermarked, batch_size)
        else:
            results['fid'] = None
            
        return results
        
    def print_results(self, results: Dict[str, any]):
        """Print formatted results."""
        print("\n" + "=" * 50)
        print("Image Quality Metrics")
        print("=" * 50)
        
        if 'psnr' in results and results['psnr'] is not None:
            print(f"  {results['psnr']}")
            
        if 'ssim' in results and results['ssim'] is not None:
            print(f"  {results['ssim']}")
            
        if 'lpips' in results and results['lpips'] is not None:
            print(f"  {results['lpips']}")
            
        if 'fid' in results and results['fid'] is not None:
            print(f"  FID: {results['fid']:.4f}")
            
        print("=" * 50)

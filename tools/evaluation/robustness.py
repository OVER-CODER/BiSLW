"""Watermark Robustness Evaluation Module.

Evaluates watermark robustness under various attacks including JPEG compression,
Gaussian noise, Gaussian blur, resizing, random cropping, and rotation.
Computes bit-level, detection-level, and ROC/AUC metrics.
"""

import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class AttackResult:
    """Results container for a single attack evaluation."""
    attack_name: str
    attack_params: Dict
    bit_accuracy: float
    detection_accuracy: float
    ber: float
    fpr: float
    tpr: float
    auc_score: float
    roc_data: Dict = field(default_factory=dict)
    
    def __repr__(self):
        return (f"{self.attack_name} ({self.attack_params}): "
                f"BitAcc={self.bit_accuracy:.4f}, DetAcc={self.detection_accuracy:.4f}, "
                f"BER={self.ber:.4f}, AUC={self.auc_score:.4f}")


class Attack(nn.Module):
    """Base class for image-space watermark attacks."""
    
    def __init__(self, name: str):
        """Initializes the Attack base class.

        Args:
            name (str): The name of the attack.
        """
        super().__init__()
        self.name = name
        
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Applies the attack transformation on the image tensor.

        Args:
            images (torch.Tensor): Input images of shape (B, C, H, W).

        Returns:
            torch.Tensor: Attacked images of shape (B, C, H, W).
        """
        raise NotImplementedError


class JPEGCompression(Attack):
    """Differentiable JPEG compression approximation using DCT-based quantization."""
    
    def __init__(self, quality: int = 50):
        """Initializes the JPEGCompression attack.

        Args:
            quality (int): Simulated quality factor (1-100).
        """
        super().__init__(f"JPEG-Q{quality}")
        self.quality = quality
        self.params = {"quality": quality}
        self.scale = (1 - quality / 100) * 0.5 + 0.1
        
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Applies simulated JPEG compression to images.

        Args:
            images (torch.Tensor): Input images of shape (B, C, H, W).

        Returns:
            torch.Tensor: Attacked images.
        """
        B, C, H, W = images.shape
        block_size = 8
        
        pad_h = (block_size - H % block_size) % block_size
        pad_w = (block_size - W % block_size) % block_size
        images_padded = F.pad(images, (0, pad_w, 0, pad_h), mode='reflect')
        
        H_pad, W_pad = images_padded.shape[2:]
        
        blocks = images_padded.unfold(2, block_size, block_size).unfold(3, block_size, block_size)
        blocks = blocks.contiguous()
        
        noise = torch.randn_like(blocks) * self.scale
        blocks_noisy = blocks + noise
        
        scale_factor = max(0.5, self.quality / 100)
        h_small = max(1, int(H_pad * scale_factor))
        w_small = max(1, int(W_pad * scale_factor))
        
        images_down = F.interpolate(images_padded, size=(h_small, w_small), mode='bilinear', align_corners=False)
        images_up = F.interpolate(images_down, size=(H_pad, W_pad), mode='bilinear', align_corners=False)
        
        blend_factor = self.quality / 100
        images_compressed = blend_factor * images_padded + (1 - blend_factor) * images_up
        
        images_compressed = images_compressed[:, :, :H, :W]
        return images_compressed


class GaussianNoise(Attack):
    """Adds zero-mean Gaussian noise to images."""
    
    def __init__(self, sigma: float = 0.05):
        """Initializes the GaussianNoise attack.

        Args:
            sigma (float): Standard deviation of the noise.
        """
        super().__init__(f"Noise-σ{sigma}")
        self.sigma = sigma
        self.params = {"sigma": sigma}
        
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Applies Gaussian noise to images.

        Args:
            images (torch.Tensor): Input images.

        Returns:
            torch.Tensor: Noisy images clipped to [-1, 1].
        """
        noise = torch.randn_like(images) * self.sigma
        return torch.clamp(images + noise, -1, 1)


class GaussianBlur(Attack):
    """Applies a 2D Gaussian blur filter to images."""
    
    def __init__(self, kernel_size: int = 5, sigma: Optional[float] = None):
        """Initializes the GaussianBlur attack.

        Args:
            kernel_size (int): Size of the blur kernel (must be odd).
            sigma (Optional[float]): Blur standard deviation.
        """
        super().__init__(f"Blur-K{kernel_size}")
        self.kernel_size = kernel_size
        self.sigma = sigma if sigma else kernel_size / 3
        self.params = {"kernel_size": kernel_size, "sigma": self.sigma}
        
        self.register_buffer('kernel', self._create_kernel())
        
    def _create_kernel(self) -> torch.Tensor:
        """Creates a 2D Gaussian kernel tensor.

        Returns:
            torch.Tensor: Normalized Gaussian kernel of shape (1, 1, K, K).
        """
        coords = torch.arange(self.kernel_size, dtype=torch.float32) - self.kernel_size // 2
        xx, yy = torch.meshgrid(coords, coords, indexing='ij')
        kernel = torch.exp(-(xx**2 + yy**2) / (2 * self.sigma**2))
        kernel = kernel / kernel.sum()
        return kernel.unsqueeze(0).unsqueeze(0)
        
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Blurs the input images.

        Args:
            images (torch.Tensor): Input images.

        Returns:
            torch.Tensor: Blurred images.
        """
        B, C, H, W = images.shape
        kernel = self.kernel.to(images.device)
        kernel_expanded = kernel.expand(C, 1, -1, -1)
        
        padding = self.kernel_size // 2
        blurred = F.conv2d(images, kernel_expanded, padding=padding, groups=C)
        return blurred


class Resize(Attack):
    """Resizes images to a smaller dimension, then interpolates back."""
    
    def __init__(self, scale: float = 0.5):
        """Initializes the Resize attack.

        Args:
            scale (float): Rescaling factor.
        """
        super().__init__(f"Resize-{scale}x")
        self.scale = scale
        self.params = {"scale": scale}
        
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Applies resize downsampling and upsampling.

        Args:
            images (torch.Tensor): Input images.

        Returns:
            torch.Tensor: Resized and upscaled images.
        """
        B, C, H, W = images.shape
        
        h_small = max(1, int(H * self.scale))
        w_small = max(1, int(W * self.scale))
        
        images_down = F.interpolate(images, size=(h_small, w_small), mode='bilinear', align_corners=False)
        images_up = F.interpolate(images_down, size=(H, W), mode='bilinear', align_corners=False)
        return images_up


class RandomCrop(Attack):
    """Crops a random subgrid from the image and scales it back."""
    
    def __init__(self, crop_ratio: float = 0.1):
        """Initializes the RandomCrop attack.

        Args:
            crop_ratio (float): Ratio of image edges to crop (0 to 0.5).
        """
        super().__init__(f"Crop-{int(crop_ratio*100)}%")
        self.crop_ratio = crop_ratio
        self.params = {"crop_ratio": crop_ratio}
        
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Applies random cropping.

        Args:
            images (torch.Tensor): Input images.

        Returns:
            torch.Tensor: Cropped and resized images.
        """
        B, C, H, W = images.shape
        
        crop_h = int(H * self.crop_ratio)
        crop_w = int(W * self.crop_ratio)
        
        top = torch.randint(0, crop_h + 1, (1,)).item() if crop_h > 0 else 0
        left = torch.randint(0, crop_w + 1, (1,)).item() if crop_w > 0 else 0
        
        cropped = images[:, :, top:H-crop_h+top, left:W-crop_w+left]
        resized = F.interpolate(cropped, size=(H, W), mode='bilinear', align_corners=False)
        return resized


class Rotation(Attack):
    """Rotates images by a set degree with bilinear sampling and reflection padding."""
    
    def __init__(self, angle: float = 5.0):
        """Initializes the Rotation attack.

        Args:
            angle (float): Rotation angle in degrees.
        """
        super().__init__(f"Rotate-{angle}°")
        self.angle = angle
        self.params = {"angle": angle}
        
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Applies rotation.

        Args:
            images (torch.Tensor): Input images.

        Returns:
            torch.Tensor: Rotated images.
        """
        B, C, H, W = images.shape
        angle_rad = self.angle * np.pi / 180
        
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        
        theta = torch.tensor([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0]
        ], dtype=images.dtype, device=images.device).unsqueeze(0).expand(B, -1, -1)
        
        grid = F.affine_grid(theta, images.size(), align_corners=False)
        rotated = F.grid_sample(images, grid, mode='bilinear', padding_mode='reflection', align_corners=False)
        return rotated


class CombinedAttack(Attack):
    """Chains multiple attacks sequentially."""
    
    def __init__(self, attacks: List[Attack]):
        """Initializes the CombinedAttack.

        Args:
            attacks (List[Attack]): Ordered list of attacks to apply.
        """
        names = [a.name for a in attacks]
        super().__init__(f"Combined({','.join(names)})")
        self.attacks = nn.ModuleList(attacks)
        self.params = {"attacks": names}
        
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Applies chained attacks sequentially.

        Args:
            images (torch.Tensor): Input images.

        Returns:
            torch.Tensor: Combined attacked images.
        """
        for attack in self.attacks:
            images = attack(images)
        return images


class RobustnessEvaluator:
    """Robustness evaluation framework for measuring watermark integrity under attacks."""
    
    def __init__(
        self,
        device: Optional[torch.device] = None,
        output_dir: str = "results/robustness"
    ):
        """Initializes the RobustnessEvaluator.

        Args:
            device: Device for computation (e.g., CPU, CUDA, MPS).
            output_dir: Directory for storing evaluation plots and logs.
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.attacks = self._create_attack_battery()
        
    def _create_attack_battery(self) -> Dict[str, List[Attack]]:
        """Defines the default evaluation suite of attacks.

        Returns:
            Dict[str, List[Attack]]: A dictionary grouping attack types.
        """
        attacks = {
            "JPEG Compression": [
                JPEGCompression(quality=90),
                JPEGCompression(quality=70),
                JPEGCompression(quality=50),
                JPEGCompression(quality=30),
            ],
            "Gaussian Noise": [
                GaussianNoise(sigma=0.01),
                GaussianNoise(sigma=0.05),
                GaussianNoise(sigma=0.1),
            ],
            "Gaussian Blur": [
                GaussianBlur(kernel_size=3),
                GaussianBlur(kernel_size=5),
                GaussianBlur(kernel_size=7),
            ],
            "Resize": [
                Resize(scale=0.5),
                Resize(scale=0.75),
                Resize(scale=1.5),
            ],
            "Random Crop": [
                RandomCrop(crop_ratio=0.10),
                RandomCrop(crop_ratio=0.25),
            ],
            "Rotation": [
                Rotation(angle=5.0),
                Rotation(angle=-5.0),
                Rotation(angle=10.0),
                Rotation(angle=-10.0),
            ],
        }
        return attacks
        
    def compute_bit_accuracy(
        self,
        watermark_true: torch.Tensor,
        watermark_pred: torch.Tensor
    ) -> float:
        """Computes the bit-level recovery accuracy.

        Args:
            watermark_true (torch.Tensor): True watermark of shape (B, W_dim).
            watermark_pred (torch.Tensor): Extracted watermark predictions.

        Returns:
            float: Bit accuracy value in range [0, 1].
        """
        bits_true = (watermark_true > 0).float()
        bits_pred = (watermark_pred > 0).float()
        accuracy = (bits_true == bits_pred).float().mean().item()
        return accuracy
        
    def compute_ber(
        self,
        watermark_true: torch.Tensor,
        watermark_pred: torch.Tensor
    ) -> float:
        """Computes the Bit Error Rate (BER).

        Args:
            watermark_true (torch.Tensor): True watermark of shape (B, W_dim).
            watermark_pred (torch.Tensor): Extracted watermark predictions.

        Returns:
            float: BER value in range [0, 1].
        """
        bits_true = (watermark_true > 0).float()
        bits_pred = (watermark_pred > 0).float()
        ber = (bits_true != bits_pred).float().mean().item()
        return ber
        
    def compute_detection_metrics(
        self,
        watermark_true: torch.Tensor,
        watermark_pred: torch.Tensor,
        threshold: float = 0.5
    ) -> Tuple[float, float, float, float, Dict]:
        """Calculates TPR, FPR, detection accuracy, and ROC/AUC metrics.

        Args:
            watermark_true (torch.Tensor): True watermark of shape (B, W_dim).
            watermark_pred (torch.Tensor): Extracted watermark predictions.
            threshold (float): Similarity threshold for detection.

        Returns:
            Tuple[float, float, float, float, Dict]:
                - detection_acc
                - fpr
                - tpr
                - auc_score
                - roc_data (fpr_array, tpr_array, thresholds)
        """
        watermark_true_norm = F.normalize(watermark_true, dim=1)
        watermark_pred_norm = F.normalize(watermark_pred, dim=1)
        
        similarity = (watermark_true_norm * watermark_pred_norm).sum(dim=1)
        
        B = watermark_true.shape[0]
        random_watermarks = torch.randn_like(watermark_true)
        random_watermarks_norm = F.normalize(random_watermarks, dim=1)
        
        similarity_random = (watermark_true_norm * random_watermarks_norm).sum(dim=1)
        
        labels = torch.cat([torch.ones(B), torch.zeros(B)])
        scores = torch.cat([similarity, similarity_random])
        
        labels_np = labels.cpu().numpy()
        scores_np = scores.cpu().numpy()
        
        fpr_array, tpr_array, thresholds = roc_curve(labels_np, scores_np)
        auc_score = auc(fpr_array, tpr_array)
        
        detected = (scores > threshold).float()
        
        tp = (detected[:B] == 1).sum().item()
        fn = (detected[:B] == 0).sum().item()
        fp = (detected[B:] == 1).sum().item()
        tn = (detected[B:] == 0).sum().item()
        
        tpr = tp / (tp + fn + 1e-8)
        fpr = fp / (fp + tn + 1e-8)
        detection_acc = (tp + tn) / (tp + tn + fp + fn)
        
        roc_data = {
            "fpr_array": fpr_array,
            "tpr_array": tpr_array,
            "thresholds": thresholds
        }
        
        return detection_acc, fpr, tpr, auc_score, roc_data
        
    @torch.no_grad()
    def evaluate_attack(
        self,
        attack: Attack,
        images_watermarked: torch.Tensor,
        watermark_true: torch.Tensor,
        extract_watermark_fn: Callable,
        vae_encode_fn: Optional[Callable] = None,
        vae_decode_fn: Optional[Callable] = None
    ) -> AttackResult:
        """Evaluates watermark robustness under a single attack.

        Args:
            attack (Attack): Attack instance to apply.
            images_watermarked (torch.Tensor): Watermarked images.
            watermark_true (torch.Tensor): True watermark bits.
            extract_watermark_fn (Callable): Extraction function.
            vae_encode_fn (Optional[Callable]): Optional encoder to project back to latent space.
            vae_decode_fn (Optional[Callable]): Optional decoder.

        Returns:
            AttackResult: Performance metrics under the specified attack.
        """
        images_watermarked = images_watermarked.to(self.device)
        watermark_true = watermark_true.to(self.device)
        attack = attack.to(self.device)
        
        images_attacked = attack(images_watermarked)
        
        if vae_encode_fn is not None:
            z_attacked = vae_encode_fn(images_attacked)
            watermark_pred = extract_watermark_fn(z_attacked)
        else:
            watermark_pred = extract_watermark_fn(images_attacked)
            
        bit_accuracy = self.compute_bit_accuracy(watermark_true, watermark_pred)
        ber = self.compute_ber(watermark_true, watermark_pred)
        detection_acc, fpr, tpr, auc_score, roc_data = self.compute_detection_metrics(
            watermark_true, watermark_pred
        )
        
        return AttackResult(
            attack_name=attack.name,
            attack_params=attack.params,
            bit_accuracy=bit_accuracy,
            detection_accuracy=detection_acc,
            ber=ber,
            fpr=fpr,
            tpr=tpr,
            auc_score=auc_score,
            roc_data=roc_data
        )
        
    @torch.no_grad()
    def evaluate_all_attacks(
        self,
        images_watermarked: torch.Tensor,
        watermark_true: torch.Tensor,
        extract_watermark_fn: Callable,
        vae_encode_fn: Optional[Callable] = None,
        vae_decode_fn: Optional[Callable] = None
    ) -> Dict[str, List[AttackResult]]:
        """Runs the complete battery of attacks on watermarked images.

        Args:
            images_watermarked (torch.Tensor): Watermarked images.
            watermark_true (torch.Tensor): True watermark bits.
            extract_watermark_fn (Callable): Extraction function.
            vae_encode_fn (Optional[Callable]): Optional VAE encoder.
            vae_decode_fn (Optional[Callable]): Optional VAE decoder.

        Returns:
            Dict[str, List[AttackResult]]: Mapping of attack categories to results.
        """
        results = {}
        
        for category, attacks in self.attacks.items():
            print(f"\nEvaluating {category}...")
            results[category] = []
            
            for attack in attacks:
                result = self.evaluate_attack(
                    attack=attack,
                    images_watermarked=images_watermarked,
                    watermark_true=watermark_true,
                    extract_watermark_fn=extract_watermark_fn,
                    vae_encode_fn=vae_encode_fn,
                    vae_decode_fn=vae_decode_fn
                )
                results[category].append(result)
                print(f"  {result}")
                
        return results
        
    def plot_roc_curves(
        self,
        results: Dict[str, List[AttackResult]],
        save_path: Optional[str] = None
    ):
        """Generates ROC curves showing detection AUC under various attacks.

        Args:
            results (Dict[str, List[AttackResult]]): Results mapping from evaluate_all_attacks.
            save_path (Optional[str]): Output file path.
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, (category, attack_results) in enumerate(results.items()):
            if idx >= len(axes):
                break
                
            ax = axes[idx]
            
            for result in attack_results:
                if result.roc_data:
                    ax.plot(
                        result.roc_data["fpr_array"],
                        result.roc_data["tpr_array"],
                        label=f"{result.attack_name} (AUC={result.auc_score:.3f})"
                    )
                    
            ax.plot([0, 1], [0, 1], 'k--', label='Random')
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title(category)
            ax.legend(loc='lower right', fontsize=8)
            ax.grid(True, alpha=0.3)
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"ROC curves saved to {save_path}")
        else:
            plt.savefig(os.path.join(self.output_dir, "roc_curves.png"), dpi=150, bbox_inches='tight')
            
        plt.close()
        
    def plot_attack_severity(
        self,
        results: Dict[str, List[AttackResult]],
        metric: str = "bit_accuracy",
        save_path: Optional[str] = None
    ):
        """Plots watermark recovery performance against attack severity.

        Args:
            results (Dict[str, List[AttackResult]]): Results mapping.
            metric (str): The metric variable to plot ('bit_accuracy', 'ber', 'auc_score').
            save_path (Optional[str]): Output file path.
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for idx, (category, attack_results) in enumerate(results.items()):
            if idx >= len(axes):
                break
                
            ax = axes[idx]
            params = []
            metrics = []
            labels = []
            
            for result in attack_results:
                if 'quality' in result.attack_params:
                    params.append(result.attack_params['quality'])
                elif 'sigma' in result.attack_params:
                    params.append(result.attack_params['sigma'])
                elif 'kernel_size' in result.attack_params:
                    params.append(result.attack_params['kernel_size'])
                elif 'scale' in result.attack_params:
                    params.append(result.attack_params['scale'])
                elif 'crop_ratio' in result.attack_params:
                    params.append(result.attack_params['crop_ratio'] * 100)
                elif 'angle' in result.attack_params:
                    params.append(abs(result.attack_params['angle']))
                else:
                    params.append(len(params))
                    
                metrics.append(getattr(result, metric))
                labels.append(result.attack_name)
                
            ax.bar(range(len(params)), metrics, tick_label=labels)
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_title(category)
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3, axis='y')
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.savefig(
                os.path.join(self.output_dir, f"attack_severity_{metric}.png"), 
                dpi=150, 
                bbox_inches='tight'
            )
            
        plt.close()
        
    def generate_report(
        self,
        results: Dict[str, List[AttackResult]],
        save_path: Optional[str] = None
    ) -> str:
        """Compiles and saves a comprehensive robustness evaluation report.

        Args:
            results (Dict[str, List[AttackResult]]): Results from evaluate_all_attacks.
            save_path (Optional[str]): Output file path.

        Returns:
            str: Compiled report contents.
        """
        lines = []
        lines.append("=" * 80)
        lines.append("WATERMARK ROBUSTNESS EVALUATION REPORT")
        lines.append("=" * 80)
        lines.append("")
        
        for category, attack_results in results.items():
            lines.append(f"\n{category}")
            lines.append("-" * 40)
            lines.append(f"{'Attack':<20} {'BitAcc':>8} {'DetAcc':>8} {'BER':>8} {'AUC':>8}")
            lines.append("-" * 40)
            
            for result in attack_results:
                lines.append(
                    f"{result.attack_name:<20} "
                    f"{result.bit_accuracy:>8.4f} "
                    f"{result.detection_accuracy:>8.4f} "
                    f"{result.ber:>8.4f} "
                    f"{result.auc_score:>8.4f}"
                )
                
        lines.append("\n" + "=" * 80)
        
        all_results = [r for results_list in results.values() for r in results_list]
        avg_bit_acc = np.mean([r.bit_accuracy for r in all_results])
        avg_auc = np.mean([r.auc_score for r in all_results])
        avg_ber = np.mean([r.ber for r in all_results])
        
        lines.append("\nSUMMARY")
        lines.append(f"Average Bit Accuracy: {avg_bit_acc:.4f}")
        lines.append(f"Average AUC: {avg_auc:.4f}")
        lines.append(f"Average BER: {avg_ber:.4f}")
        lines.append("=" * 80)
        
        report = "\n".join(lines)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
            print(f"Report saved to {save_path}")
        else:
            report_path = os.path.join(self.output_dir, "robustness_report.txt")
            with open(report_path, 'w') as f:
                f.write(report)
                
        return report

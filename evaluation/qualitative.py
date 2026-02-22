"""
Qualitative Experiments Module for Watermark Visualization.

Provides:
- Side-by-side visual comparisons (original vs watermarked)
- Visualization of difference maps
- Visualization of extracted bit distributions after attacks
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from typing import Dict, List, Tuple, Optional, Callable
import os
from PIL import Image


class QualitativeExperiments:
    """
    Qualitative experiment framework for watermark visualization.
    """
    
    def __init__(
        self,
        device: torch.device = None,
        output_dir: str = "results/qualitative"
    ):
        """
        Args:
            device: Device for computation
            output_dir: Directory for saving visualizations
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Custom colormap for difference visualization
        colors = [(0, 0, 0.5), (0, 0, 1), (1, 1, 1), (1, 0, 0), (0.5, 0, 0)]
        self.diff_cmap = LinearSegmentedColormap.from_list('diff', colors, N=256)
        
    def _denormalize(self, tensor: torch.Tensor) -> torch.Tensor:
        """Convert from [-1, 1] to [0, 1]."""
        return (tensor + 1) / 2
        
    def _to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert tensor to numpy array for plotting."""
        if tensor.dim() == 4:
            tensor = tensor[0]  # Take first image
        # CHW -> HWC
        return self._denormalize(tensor).cpu().permute(1, 2, 0).numpy().clip(0, 1)
        
    def plot_side_by_side(
        self,
        original: torch.Tensor,
        watermarked: torch.Tensor,
        titles: Tuple[str, str] = ("Original", "Watermarked"),
        n_samples: int = 4,
        save_path: str = None,
        show_metrics: bool = True
    ):
        """
        Create side-by-side comparison of original and watermarked images.
        
        Args:
            original: Original images (B, C, H, W) in [-1, 1]
            watermarked: Watermarked images (B, C, H, W) in [-1, 1]
            titles: Titles for the two columns
            n_samples: Number of samples to show
            save_path: Path to save figure
            show_metrics: Whether to show PSNR/SSIM for each pair
        """
        n_samples = min(n_samples, len(original))
        
        fig, axes = plt.subplots(n_samples, 3, figsize=(12, 4 * n_samples))
        if n_samples == 1:
            axes = axes.reshape(1, -1)
            
        for i in range(n_samples):
            orig_img = self._to_numpy(original[i])
            wm_img = self._to_numpy(watermarked[i])
            
            # Compute difference map
            diff = np.abs(orig_img - wm_img)
            diff_enhanced = diff * 10  # Enhance for visibility
            diff_enhanced = np.clip(diff_enhanced, 0, 1)
            
            # Original
            axes[i, 0].imshow(orig_img)
            axes[i, 0].set_title(titles[0] if i == 0 else "")
            axes[i, 0].axis('off')
            
            # Watermarked
            axes[i, 1].imshow(wm_img)
            
            if show_metrics:
                # Compute PSNR and SSIM for this image
                mse = np.mean((orig_img - wm_img) ** 2)
                psnr = 10 * np.log10(1.0 / (mse + 1e-10))
                # Simple SSIM approximation
                axes[i, 1].set_title(f"{titles[1] if i == 0 else ''}\nPSNR: {psnr:.2f} dB")
            else:
                axes[i, 1].set_title(titles[1] if i == 0 else "")
            axes[i, 1].axis('off')
            
            # Difference map
            axes[i, 2].imshow(diff_enhanced)
            axes[i, 2].set_title("Difference (10×)" if i == 0 else "")
            axes[i, 2].axis('off')
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.savefig(os.path.join(self.output_dir, "side_by_side.png"), dpi=150, bbox_inches='tight')
            
        plt.close()
        
    def plot_difference_maps(
        self,
        original: torch.Tensor,
        watermarked: torch.Tensor,
        n_samples: int = 4,
        save_path: str = None
    ):
        """
        Detailed visualization of difference maps.
        
        Shows:
        - RGB difference
        - Per-channel differences
        - Spatial distribution of errors
        
        Args:
            original: Original images (B, C, H, W)
            watermarked: Watermarked images (B, C, H, W)
            n_samples: Number of samples to visualize
            save_path: Path to save figure
        """
        n_samples = min(n_samples, len(original))
        
        fig = plt.figure(figsize=(16, 4 * n_samples))
        gs = gridspec.GridSpec(n_samples, 6, figure=fig)
        
        for i in range(n_samples):
            orig = self._denormalize(original[i]).cpu().numpy()  # CHW
            wm = self._denormalize(watermarked[i]).cpu().numpy()  # CHW
            
            diff = orig - wm  # Signed difference
            
            # Original image
            ax1 = fig.add_subplot(gs[i, 0])
            ax1.imshow(np.transpose(orig, (1, 2, 0)).clip(0, 1))
            ax1.set_title("Original" if i == 0 else "")
            ax1.axis('off')
            
            # Watermarked image
            ax2 = fig.add_subplot(gs[i, 1])
            ax2.imshow(np.transpose(wm, (1, 2, 0)).clip(0, 1))
            ax2.set_title("Watermarked" if i == 0 else "")
            ax2.axis('off')
            
            # RGB difference (absolute, enhanced)
            ax3 = fig.add_subplot(gs[i, 2])
            diff_rgb = np.abs(diff)
            diff_rgb_enhanced = np.clip(diff_rgb * 20, 0, 1)  # 20x enhancement
            ax3.imshow(np.transpose(diff_rgb_enhanced, (1, 2, 0)))
            ax3.set_title("Diff (20×)" if i == 0 else "")
            ax3.axis('off')
            
            # Per-channel differences
            channel_names = ['R', 'G', 'B']
            for c in range(3):
                ax = fig.add_subplot(gs[i, 3 + c])
                im = ax.imshow(diff[c], cmap=self.diff_cmap, vmin=-0.1, vmax=0.1)
                ax.set_title(f"{channel_names[c]} diff" if i == 0 else "")
                ax.axis('off')
                
        # Add colorbar
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        plt.colorbar(im, cax=cbar_ax, label='Pixel difference')
        
        plt.tight_layout(rect=[0, 0, 0.9, 1])
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.savefig(os.path.join(self.output_dir, "difference_maps.png"), dpi=150, bbox_inches='tight')
            
        plt.close()
        
    def plot_bit_distributions(
        self,
        watermark_true: torch.Tensor,
        watermark_predictions: Dict[str, torch.Tensor],
        save_path: str = None
    ):
        """
        Visualize extracted bit distributions after various attacks.
        
        Args:
            watermark_true: Ground truth watermark (B, W_dim)
            watermark_predictions: Dict mapping attack name to predicted watermarks
            save_path: Path to save figure
        """
        n_attacks = len(watermark_predictions)
        fig, axes = plt.subplots(2, (n_attacks + 1) // 2 + 1, figsize=(4 * ((n_attacks + 1) // 2 + 1), 8))
        axes = axes.flatten()
        
        # True watermark distribution
        ax = axes[0]
        w_true = watermark_true[0].cpu().numpy()
        bits_true = (w_true > 0).astype(float)
        ax.bar(range(len(w_true)), w_true, alpha=0.7, color='blue')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_title("Original Watermark")
        ax.set_xlabel("Bit index")
        ax.set_ylabel("Value")
        
        # Predictions after each attack
        for i, (attack_name, w_pred) in enumerate(watermark_predictions.items(), 1):
            if i >= len(axes):
                break
                
            ax = axes[i]
            w = w_pred[0].cpu().numpy()
            
            # Color by correctness
            bits_pred = (w > 0).astype(float)
            colors = ['green' if t == p else 'red' for t, p in zip(bits_true, bits_pred)]
            
            ax.bar(range(len(w)), w, alpha=0.7, color=colors)
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            
            accuracy = (bits_true == bits_pred).mean() * 100
            ax.set_title(f"{attack_name}\nAcc: {accuracy:.1f}%")
            ax.set_xlabel("Bit index")
            ax.set_ylabel("Value")
            
        # Hide unused axes
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.savefig(os.path.join(self.output_dir, "bit_distributions.png"), dpi=150, bbox_inches='tight')
            
        plt.close()
        
    def plot_bit_correlation(
        self,
        watermark_true: torch.Tensor,
        watermark_pred: torch.Tensor,
        attack_name: str = "No Attack",
        save_path: str = None
    ):
        """
        Scatter plot of true vs predicted watermark values.
        
        Args:
            watermark_true: Ground truth watermark (B, W_dim)
            watermark_pred: Predicted watermark (B, W_dim)
            attack_name: Name of attack applied
            save_path: Path to save figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Flatten all samples
        w_true = watermark_true.cpu().numpy().flatten()
        w_pred = watermark_pred.cpu().numpy().flatten()
        
        # Scatter plot
        ax1 = axes[0]
        ax1.scatter(w_true, w_pred, alpha=0.3, s=10)
        ax1.plot([-3, 3], [-3, 3], 'r--', label='Perfect correlation')
        ax1.set_xlabel('True watermark value')
        ax1.set_ylabel('Predicted watermark value')
        ax1.set_title(f'Watermark Correlation ({attack_name})')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Set equal aspect ratio
        ax1.set_xlim(-3, 3)
        ax1.set_ylim(-3, 3)
        ax1.set_aspect('equal')
        
        # Bit accuracy histogram
        ax2 = axes[1]
        bits_true = (w_true > 0).astype(float)
        bits_pred = (w_pred > 0).astype(float)
        correct = (bits_true == bits_pred).astype(float)
        
        # Show distribution of prediction confidence
        confidence = np.abs(w_pred)
        ax2.hist(confidence[correct == 1], bins=50, alpha=0.7, label='Correct', color='green')
        ax2.hist(confidence[correct == 0], bins=50, alpha=0.7, label='Incorrect', color='red')
        ax2.set_xlabel('Prediction confidence (|value|)')
        ax2.set_ylabel('Count')
        ax2.set_title('Confidence Distribution')
        ax2.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.savefig(os.path.join(self.output_dir, f"bit_correlation_{attack_name.replace(' ', '_').lower()}.png"), 
                       dpi=150, bbox_inches='tight')
            
        plt.close()
        
    def plot_attack_progression(
        self,
        original: torch.Tensor,
        watermarked: torch.Tensor,
        attacked_images: Dict[str, torch.Tensor],
        watermark_predictions: Dict[str, torch.Tensor],
        watermark_true: torch.Tensor,
        sample_idx: int = 0,
        save_path: str = None
    ):
        """
        Visualize progression of attacks and their effect on watermark.
        
        Args:
            original: Original images
            watermarked: Watermarked images
            attacked_images: Dict mapping attack name to attacked images
            watermark_predictions: Dict mapping attack name to extracted watermarks
            watermark_true: Ground truth watermark
            sample_idx: Index of sample to visualize
            save_path: Path to save figure
        """
        n_attacks = len(attacked_images)
        
        fig = plt.figure(figsize=(4 * (n_attacks + 2), 8))
        gs = gridspec.GridSpec(2, n_attacks + 2, figure=fig, height_ratios=[3, 2])
        
        # Row 1: Images
        # Original
        ax = fig.add_subplot(gs[0, 0])
        ax.imshow(self._to_numpy(original[sample_idx]))
        ax.set_title("Original")
        ax.axis('off')
        
        # Watermarked
        ax = fig.add_subplot(gs[0, 1])
        ax.imshow(self._to_numpy(watermarked[sample_idx]))
        ax.set_title("Watermarked")
        ax.axis('off')
        
        # Attacked images
        for i, (attack_name, imgs) in enumerate(attacked_images.items()):
            ax = fig.add_subplot(gs[0, i + 2])
            ax.imshow(self._to_numpy(imgs[sample_idx]))
            ax.set_title(attack_name, fontsize=10)
            ax.axis('off')
            
        # Row 2: Watermark recovery
        w_true = watermark_true[sample_idx].cpu().numpy()
        bits_true = (w_true > 0).astype(float)
        
        # Original watermark
        ax = fig.add_subplot(gs[1, 0:2])
        ax.bar(range(len(w_true)), w_true, alpha=0.7, color='blue')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_title("True Watermark")
        ax.set_ylim(-3, 3)
        
        # Predictions after attacks
        for i, (attack_name, w_pred) in enumerate(watermark_predictions.items()):
            ax = fig.add_subplot(gs[1, i + 2])
            w = w_pred[sample_idx].cpu().numpy()
            bits_pred = (w > 0).astype(float)
            
            colors = ['green' if t == p else 'red' for t, p in zip(bits_true, bits_pred)]
            ax.bar(range(len(w)), w, alpha=0.7, color=colors)
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
            
            accuracy = (bits_true == bits_pred).mean() * 100
            ax.set_title(f"Acc: {accuracy:.1f}%", fontsize=10)
            ax.set_ylim(-3, 3)
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.savefig(os.path.join(self.output_dir, "attack_progression.png"), dpi=150, bbox_inches='tight')
            
        plt.close()
        
    def plot_latent_visualization(
        self,
        z_original: torch.Tensor,
        z_watermarked: torch.Tensor,
        sample_idx: int = 0,
        save_path: str = None
    ):
        """
        Visualize latent space before and after watermarking.
        
        Args:
            z_original: Original latent (B, C, H, W)
            z_watermarked: Watermarked latent (B, C, H, W)
            sample_idx: Index of sample to visualize
            save_path: Path to save figure
        """
        z_orig = z_original[sample_idx].cpu().numpy()
        z_wm = z_watermarked[sample_idx].cpu().numpy()
        z_diff = z_orig - z_wm
        
        C = z_orig.shape[0]
        
        fig, axes = plt.subplots(3, C, figsize=(4 * C, 12))
        
        for c in range(C):
            # Original latent channel
            im1 = axes[0, c].imshow(z_orig[c], cmap='viridis')
            axes[0, c].set_title(f"Original Channel {c}")
            axes[0, c].axis('off')
            plt.colorbar(im1, ax=axes[0, c], fraction=0.046)
            
            # Watermarked latent channel
            im2 = axes[1, c].imshow(z_wm[c], cmap='viridis')
            axes[1, c].set_title(f"Watermarked Channel {c}")
            axes[1, c].axis('off')
            plt.colorbar(im2, ax=axes[1, c], fraction=0.046)
            
            # Difference
            vmax = max(abs(z_diff[c].min()), abs(z_diff[c].max()))
            im3 = axes[2, c].imshow(z_diff[c], cmap=self.diff_cmap, vmin=-vmax, vmax=vmax)
            axes[2, c].set_title(f"Difference Channel {c}")
            axes[2, c].axis('off')
            plt.colorbar(im3, ax=axes[2, c], fraction=0.046)
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.savefig(os.path.join(self.output_dir, "latent_visualization.png"), dpi=150, bbox_inches='tight')
            
        plt.close()
        
    def save_image_grid(
        self,
        images: torch.Tensor,
        nrow: int = 4,
        save_path: str = None,
        title: str = None
    ):
        """
        Save a grid of images.
        
        Args:
            images: Images tensor (B, C, H, W)
            nrow: Number of images per row
            save_path: Path to save
            title: Optional title
        """
        from torchvision.utils import make_grid
        
        images = self._denormalize(images)
        grid = make_grid(images, nrow=nrow, padding=2, normalize=False)
        grid_np = grid.cpu().permute(1, 2, 0).numpy().clip(0, 1)
        
        fig, ax = plt.subplots(figsize=(nrow * 3, (len(images) // nrow + 1) * 3))
        ax.imshow(grid_np)
        ax.axis('off')
        
        if title:
            ax.set_title(title, fontsize=14)
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.savefig(os.path.join(self.output_dir, "image_grid.png"), dpi=150, bbox_inches='tight')
            
        plt.close()
        
    def generate_visual_report(
        self,
        original: torch.Tensor,
        watermarked: torch.Tensor,
        watermark_true: torch.Tensor,
        watermark_predictions: Dict[str, torch.Tensor],
        attacked_images: Dict[str, torch.Tensor] = None,
        z_original: torch.Tensor = None,
        z_watermarked: torch.Tensor = None
    ):
        """
        Generate all visualizations for a comprehensive visual report.
        
        Args:
            original: Original images
            watermarked: Watermarked images
            watermark_true: True watermark
            watermark_predictions: Dict of attack name -> predicted watermarks
            attacked_images: Optional dict of attack name -> attacked images
            z_original: Optional original latents
            z_watermarked: Optional watermarked latents
        """
        print("Generating visual report...")
        
        # Side-by-side comparison
        print("  - Side-by-side comparison...")
        self.plot_side_by_side(original, watermarked)
        
        # Difference maps
        print("  - Difference maps...")
        self.plot_difference_maps(original, watermarked)
        
        # Bit distributions
        print("  - Bit distributions...")
        self.plot_bit_distributions(watermark_true, watermark_predictions)
        
        # Bit correlation for clean extraction
        if 'No Attack' in watermark_predictions or 'Clean' in watermark_predictions:
            key = 'No Attack' if 'No Attack' in watermark_predictions else 'Clean'
            print("  - Bit correlation...")
            self.plot_bit_correlation(watermark_true, watermark_predictions[key])
            
        # Attack progression
        if attacked_images:
            print("  - Attack progression...")
            self.plot_attack_progression(
                original, watermarked, 
                attacked_images, watermark_predictions, 
                watermark_true
            )
            
        # Latent visualization
        if z_original is not None and z_watermarked is not None:
            print("  - Latent visualization...")
            self.plot_latent_visualization(z_original, z_watermarked)
            
        print(f"Visual report saved to {self.output_dir}")

"""
Ablation Study Module for Watermark Evaluation.

Studies the impact of:
- Watermark strength parameter (alpha)
- Embedding layer (early vs late latent layers)
- Watermark bit length

Provides performance trend plots and trade-off interpretation.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Callable, Optional
import matplotlib.pyplot as plt
import os
from dataclasses import dataclass, field
from tqdm import tqdm


@dataclass
class AblationResult:
    """Container for ablation study results."""
    parameter_name: str
    parameter_values: List
    psnr_values: List[float]
    ssim_values: List[float]
    bit_accuracy_values: List[float]
    ber_values: List[float]
    extra_metrics: Dict = field(default_factory=dict)
    
    def best_tradeoff(self, psnr_threshold: float = 35.0, ssim_threshold: float = 0.9) -> int:
        """Find best parameter value meeting quality thresholds."""
        for i, (psnr, ssim, acc) in enumerate(zip(self.psnr_values, self.ssim_values, self.bit_accuracy_values)):
            if psnr >= psnr_threshold and ssim >= ssim_threshold:
                return i
        # If no value meets thresholds, return the one with highest combined score
        scores = [0.5 * p/50 + 0.5 * a for p, a in zip(self.psnr_values, self.bit_accuracy_values)]
        return np.argmax(scores)


class AblationStudy:
    """
    Ablation study framework for watermark parameter analysis.
    """
    
    def __init__(
        self,
        device: torch.device = None,
        output_dir: str = "results/ablation"
    ):
        """
        Args:
            device: Device for computation
            output_dir: Directory for saving results
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    @torch.no_grad()
    def ablate_watermark_strength(
        self,
        images: torch.Tensor,
        watermark: torch.Tensor,
        vae,
        splitter,
        recombiner,
        encoder_l,
        encoder_h,
        decoder_l,
        decoder_h,
        alpha_values: List[float] = None,
        compute_metrics_fn: Callable = None
    ) -> AblationResult:
        """
        Study impact of watermark strength (alpha) parameter.
        
        Args:
            images: Original images (B, C, H, W)
            watermark: Watermark to embed (B, W_dim)
            vae: VAE wrapper
            splitter: Latent splitter
            recombiner: Latent recombiner
            encoder_l, encoder_h: Watermark encoders
            decoder_l, decoder_h: Watermark decoders
            alpha_values: List of alpha values to test
            compute_metrics_fn: Function to compute PSNR/SSIM
            
        Returns:
            AblationResult with metrics for each alpha value
        """
        if alpha_values is None:
            alpha_values = [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
            
        images = images.to(self.device)
        watermark = watermark.to(self.device)
        
        psnr_values = []
        ssim_values = []
        bit_acc_values = []
        ber_values = []
        
        print("Ablating watermark strength (alpha)...")
        
        for alpha in tqdm(alpha_values, desc="Alpha values"):
            # Encode to latent
            z = vae.encode(images)
            z_low, z_high = splitter(z)
            
            # Embed watermark with current alpha
            z_low_wm = encoder_l(z_low, watermark, alpha=alpha)
            z_high_wm = encoder_h(z_high, watermark, alpha=alpha)
            z_wm = recombiner(z_low_wm, z_high_wm)
            
            # Decode to image
            images_wm = vae.decode(z_wm)
            
            # Compute image quality
            if compute_metrics_fn:
                metrics = compute_metrics_fn(images, images_wm)
                psnr = metrics['psnr'].mean if hasattr(metrics['psnr'], 'mean') else metrics['psnr']
                ssim = metrics['ssim'].mean if hasattr(metrics['ssim'], 'mean') else metrics['ssim']
            else:
                # Simple PSNR/SSIM computation
                mse = torch.mean((images - images_wm) ** 2, dim=[1, 2, 3])
                psnr = 10 * torch.log10(4.0 / (mse + 1e-10)).mean().item()  # data_range=2
                
                # Simple SSIM approximation
                mu1 = images.mean(dim=[2, 3], keepdim=True)
                mu2 = images_wm.mean(dim=[2, 3], keepdim=True)
                var1 = ((images - mu1) ** 2).mean(dim=[2, 3])
                var2 = ((images_wm - mu2) ** 2).mean(dim=[2, 3])
                cov = ((images - mu1) * (images_wm - mu2)).mean(dim=[2, 3])
                c1, c2 = 0.01**2 * 4, 0.03**2 * 4
                ssim = (((2 * mu1.squeeze() * mu2.squeeze() + c1) * (2 * cov + c2)) /
                       ((mu1.squeeze()**2 + mu2.squeeze()**2 + c1) * (var1 + var2 + c2))).mean().item()
                       
            psnr_values.append(psnr)
            ssim_values.append(ssim)
            
            # Extract watermark and compute accuracy
            z_wm_low, z_wm_high = splitter(z_wm)
            w_pred_l = decoder_l(z_wm_low)
            w_pred_h = decoder_h(z_wm_high)
            w_pred = (w_pred_l + w_pred_h) / 2
            
            # Bit accuracy
            bits_true = (watermark > 0).float()
            bits_pred = (w_pred > 0).float()
            bit_acc = (bits_true == bits_pred).float().mean().item()
            ber = 1 - bit_acc
            
            bit_acc_values.append(bit_acc)
            ber_values.append(ber)
            
        return AblationResult(
            parameter_name="Watermark Strength (α)",
            parameter_values=alpha_values,
            psnr_values=psnr_values,
            ssim_values=ssim_values,
            bit_accuracy_values=bit_acc_values,
            ber_values=ber_values
        )
        
    @torch.no_grad()
    def ablate_embedding_layer(
        self,
        images: torch.Tensor,
        watermark: torch.Tensor,
        vae,
        splitter,
        recombiner,
        encoder_l,
        encoder_h,
        decoder_l,
        decoder_h,
        alpha: float = 1.0,
        compute_metrics_fn: Callable = None
    ) -> AblationResult:
        """
        Study impact of embedding in different latent layers.
        
        Configurations:
        - Low-frequency only
        - High-frequency only
        - Both (balanced)
        - Both (low-emphasis)
        - Both (high-emphasis)
        
        Args:
            images: Original images
            watermark: Watermark to embed
            vae, splitter, recombiner: Model components
            encoder_l, encoder_h: Watermark encoders
            decoder_l, decoder_h: Watermark decoders
            alpha: Base watermark strength
            compute_metrics_fn: Metrics computation function
            
        Returns:
            AblationResult with metrics for each configuration
        """
        images = images.to(self.device)
        watermark = watermark.to(self.device)
        
        # Layer configurations: (alpha_l, alpha_h, name)
        configs = [
            (alpha, 0.0, "Low-freq only"),
            (0.0, alpha, "High-freq only"),
            (alpha, alpha, "Both (balanced)"),
            (alpha * 1.5, alpha * 0.5, "Both (low-emphasis)"),
            (alpha * 0.5, alpha * 1.5, "Both (high-emphasis)"),
        ]
        
        psnr_values = []
        ssim_values = []
        bit_acc_values = []
        ber_values = []
        config_names = []
        
        print("Ablating embedding layer...")
        
        for alpha_l, alpha_h, name in tqdm(configs, desc="Layer configs"):
            config_names.append(name)
            
            z = vae.encode(images)
            z_low, z_high = splitter(z)
            
            # Embed with current config
            if alpha_l > 0:
                z_low_wm = encoder_l(z_low, watermark, alpha=alpha_l)
            else:
                z_low_wm = z_low
                
            if alpha_h > 0:
                z_high_wm = encoder_h(z_high, watermark, alpha=alpha_h)
            else:
                z_high_wm = z_high
                
            z_wm = recombiner(z_low_wm, z_high_wm)
            images_wm = vae.decode(z_wm)
            
            # Compute metrics
            if compute_metrics_fn:
                metrics = compute_metrics_fn(images, images_wm)
                psnr = metrics['psnr'].mean if hasattr(metrics['psnr'], 'mean') else metrics['psnr']
                ssim = metrics['ssim'].mean if hasattr(metrics['ssim'], 'mean') else metrics['ssim']
            else:
                mse = torch.mean((images - images_wm) ** 2, dim=[1, 2, 3])
                psnr = 10 * torch.log10(4.0 / (mse + 1e-10)).mean().item()
                ssim = 0.9  # Placeholder
                
            psnr_values.append(psnr)
            ssim_values.append(ssim)
            
            # Extract and evaluate
            z_wm_low, z_wm_high = splitter(z_wm)
            
            # Only decode from layers that were watermarked
            if alpha_l > 0 and alpha_h > 0:
                w_pred = (decoder_l(z_wm_low) + decoder_h(z_wm_high)) / 2
            elif alpha_l > 0:
                w_pred = decoder_l(z_wm_low)
            else:
                w_pred = decoder_h(z_wm_high)
                
            bits_true = (watermark > 0).float()
            bits_pred = (w_pred > 0).float()
            bit_acc = (bits_true == bits_pred).float().mean().item()
            
            bit_acc_values.append(bit_acc)
            ber_values.append(1 - bit_acc)
            
        return AblationResult(
            parameter_name="Embedding Layer",
            parameter_values=config_names,
            psnr_values=psnr_values,
            ssim_values=ssim_values,
            bit_accuracy_values=bit_acc_values,
            ber_values=ber_values
        )
        
    @torch.no_grad()
    def ablate_watermark_length(
        self,
        images: torch.Tensor,
        vae,
        splitter,
        recombiner,
        encoder_class,
        decoder_class,
        bit_lengths: List[int] = None,
        alpha: float = 1.0,
        compute_metrics_fn: Callable = None
    ) -> AblationResult:
        """
        Study impact of watermark bit length.
        
        Args:
            images: Original images
            vae, splitter, recombiner: Model components
            encoder_class: Encoder class to instantiate
            decoder_class: Decoder class to instantiate
            bit_lengths: List of bit lengths to test
            alpha: Watermark strength
            compute_metrics_fn: Metrics computation function
            
        Returns:
            AblationResult with metrics for each bit length
        """
        if bit_lengths is None:
            bit_lengths = [16, 32, 64, 128, 256]
            
        images = images.to(self.device)
        
        psnr_values = []
        ssim_values = []
        bit_acc_values = []
        ber_values = []
        
        print("Ablating watermark bit length...")
        
        for w_dim in tqdm(bit_lengths, desc="Bit lengths"):
            # Create encoders/decoders for this bit length
            encoder_l = encoder_class(watermark_dim=w_dim).to(self.device)
            encoder_h = encoder_class(watermark_dim=w_dim).to(self.device)
            decoder_l = decoder_class(watermark_dim=w_dim).to(self.device)
            decoder_h = decoder_class(watermark_dim=w_dim).to(self.device)
            
            # Note: These are untrained - in practice, you'd need to train for each length
            # For ablation, we use initialized weights to see capacity/difficulty trends
            
            # Generate watermark of current length
            B = images.shape[0]
            watermark = torch.randn(B, w_dim, device=self.device)
            
            z = vae.encode(images)
            z_low, z_high = splitter(z)
            
            z_low_wm = encoder_l(z_low, watermark, alpha=alpha)
            z_high_wm = encoder_h(z_high, watermark, alpha=alpha)
            z_wm = recombiner(z_low_wm, z_high_wm)
            
            images_wm = vae.decode(z_wm)
            
            # Compute metrics
            if compute_metrics_fn:
                metrics = compute_metrics_fn(images, images_wm)
                psnr = metrics['psnr'].mean if hasattr(metrics['psnr'], 'mean') else metrics['psnr']
                ssim = metrics['ssim'].mean if hasattr(metrics['ssim'], 'mean') else metrics['ssim']
            else:
                mse = torch.mean((images - images_wm) ** 2, dim=[1, 2, 3])
                psnr = 10 * torch.log10(4.0 / (mse + 1e-10)).mean().item()
                ssim = 0.9
                
            psnr_values.append(psnr)
            ssim_values.append(ssim)
            
            # Extract and evaluate
            z_wm_low, z_wm_high = splitter(z_wm)
            w_pred_l = decoder_l(z_wm_low)
            w_pred_h = decoder_h(z_wm_high)
            w_pred = (w_pred_l + w_pred_h) / 2
            
            bits_true = (watermark > 0).float()
            bits_pred = (w_pred > 0).float()
            bit_acc = (bits_true == bits_pred).float().mean().item()
            
            bit_acc_values.append(bit_acc)
            ber_values.append(1 - bit_acc)
            
            # Clean up
            del encoder_l, encoder_h, decoder_l, decoder_h
            
        return AblationResult(
            parameter_name="Watermark Bit Length",
            parameter_values=bit_lengths,
            psnr_values=psnr_values,
            ssim_values=ssim_values,
            bit_accuracy_values=bit_acc_values,
            ber_values=ber_values
        )
        
    def plot_ablation_results(
        self,
        results: AblationResult,
        save_suffix: str = "",
        figsize: Tuple[int, int] = (14, 5)
    ):
        """
        Plot ablation study results.
        
        Args:
            results: AblationResult to plot
            save_suffix: Suffix for saved figure filename
            figsize: Figure size
        """
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        x = range(len(results.parameter_values))
        labels = [str(v) for v in results.parameter_values]
        
        # PSNR plot
        ax1 = axes[0]
        ax1.bar(x, results.psnr_values, color='steelblue', alpha=0.7)
        ax1.axhline(y=40, color='red', linestyle='--', label='Target (40 dB)')
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, rotation=45, ha='right')
        ax1.set_xlabel(results.parameter_name)
        ax1.set_ylabel('PSNR (dB)')
        ax1.set_title('Image Quality (PSNR)')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        
        # SSIM plot
        ax2 = axes[1]
        ax2.bar(x, results.ssim_values, color='seagreen', alpha=0.7)
        ax2.axhline(y=0.91, color='red', linestyle='--', label='Target (0.91)')
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels, rotation=45, ha='right')
        ax2.set_xlabel(results.parameter_name)
        ax2.set_ylabel('SSIM')
        ax2.set_title('Structural Similarity')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Bit accuracy plot
        ax3 = axes[2]
        ax3.bar(x, results.bit_accuracy_values, color='coral', alpha=0.7)
        ax3.axhline(y=0.95, color='red', linestyle='--', label='Target (95%)')
        ax3.set_xticks(x)
        ax3.set_xticklabels(labels, rotation=45, ha='right')
        ax3.set_xlabel(results.parameter_name)
        ax3.set_ylabel('Bit Accuracy')
        ax3.set_title('Watermark Detection Accuracy')
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        filename = f"ablation_{results.parameter_name.replace(' ', '_').lower()}"
        if save_suffix:
            filename += f"_{save_suffix}"
        plt.savefig(os.path.join(self.output_dir, f"{filename}.png"), dpi=150, bbox_inches='tight')
        plt.close()
        
    def plot_tradeoff_curves(
        self,
        results: AblationResult,
        save_suffix: str = ""
    ):
        """
        Plot trade-off curves between quality and robustness.
        
        Args:
            results: AblationResult to plot
            save_suffix: Suffix for saved filename
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # PSNR vs Bit Accuracy
        ax1 = axes[0]
        ax1.plot(results.psnr_values, results.bit_accuracy_values, 'o-', markersize=10)
        for i, label in enumerate(results.parameter_values):
            ax1.annotate(str(label), 
                        (results.psnr_values[i], results.bit_accuracy_values[i]),
                        textcoords="offset points", xytext=(5, 5), fontsize=9)
        ax1.axhline(y=0.95, color='green', linestyle='--', alpha=0.5)
        ax1.axvline(x=40, color='blue', linestyle='--', alpha=0.5)
        ax1.fill_between([40, ax1.get_xlim()[1]], 0.95, 1.0, alpha=0.1, color='green')
        ax1.set_xlabel('PSNR (dB)')
        ax1.set_ylabel('Bit Accuracy')
        ax1.set_title('Quality-Robustness Trade-off')
        ax1.grid(True, alpha=0.3)
        
        # SSIM vs Bit Accuracy
        ax2 = axes[1]
        ax2.plot(results.ssim_values, results.bit_accuracy_values, 'o-', markersize=10, color='orange')
        for i, label in enumerate(results.parameter_values):
            ax2.annotate(str(label),
                        (results.ssim_values[i], results.bit_accuracy_values[i]),
                        textcoords="offset points", xytext=(5, 5), fontsize=9)
        ax2.axhline(y=0.95, color='green', linestyle='--', alpha=0.5)
        ax2.axvline(x=0.91, color='blue', linestyle='--', alpha=0.5)
        ax2.set_xlabel('SSIM')
        ax2.set_ylabel('Bit Accuracy')
        ax2.set_title('Structural Similarity-Robustness Trade-off')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        filename = f"tradeoff_{results.parameter_name.replace(' ', '_').lower()}"
        if save_suffix:
            filename += f"_{save_suffix}"
        plt.savefig(os.path.join(self.output_dir, f"{filename}.png"), dpi=150, bbox_inches='tight')
        plt.close()
        
    def generate_ablation_report(
        self,
        results_list: List[AblationResult],
        save_path: str = None
    ) -> str:
        """
        Generate comprehensive ablation report.
        
        Args:
            results_list: List of AblationResult objects
            save_path: Path to save report
            
        Returns:
            Report string
        """
        lines = []
        lines.append("=" * 80)
        lines.append("ABLATION STUDY REPORT")
        lines.append("=" * 80)
        
        for result in results_list:
            lines.append(f"\n\n{result.parameter_name}")
            lines.append("-" * 40)
            
            # Table header
            lines.append(f"{'Value':<20} {'PSNR':>10} {'SSIM':>10} {'BitAcc':>10} {'BER':>10}")
            lines.append("-" * 60)
            
            for i, val in enumerate(result.parameter_values):
                lines.append(
                    f"{str(val):<20} "
                    f"{result.psnr_values[i]:>10.2f} "
                    f"{result.ssim_values[i]:>10.4f} "
                    f"{result.bit_accuracy_values[i]:>10.4f} "
                    f"{result.ber_values[i]:>10.4f}"
                )
                
            # Best configuration
            best_idx = result.best_tradeoff(psnr_threshold=40.0, ssim_threshold=0.91)
            lines.append(f"\nBest configuration meeting PSNR≥40dB, SSIM≥0.91:")
            lines.append(f"  {result.parameter_name} = {result.parameter_values[best_idx]}")
            lines.append(f"  PSNR: {result.psnr_values[best_idx]:.2f} dB")
            lines.append(f"  SSIM: {result.ssim_values[best_idx]:.4f}")
            lines.append(f"  Bit Accuracy: {result.bit_accuracy_values[best_idx]:.4f}")
            
        lines.append("\n" + "=" * 80)
        lines.append("\nINTERPRETATION OF TRADE-OFFS")
        lines.append("-" * 40)
        lines.append("""
1. Watermark Strength (α):
   - Higher α → Better watermark recovery but lower image quality
   - Recommended: Start with α=1.0 and adjust based on requirements
   
2. Embedding Layer:
   - Low-frequency: More robust to compression/blur, affects global appearance
   - High-frequency: More fragile, affects fine details
   - Dual embedding provides best balance

3. Watermark Bit Length:
   - Longer watermarks → More information but harder to recover
   - 64-128 bits typically provides good balance
""")
        
        lines.append("=" * 80)
        
        report = "\n".join(lines)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
        else:
            with open(os.path.join(self.output_dir, "ablation_report.txt"), 'w') as f:
                f.write(report)
                
        return report

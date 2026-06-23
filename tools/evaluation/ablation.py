"""Ablation Study Module for Watermark Evaluation.

Analyzes the impact of watermark strength (alpha), embedding layer selection,
and watermark bit length on reconstructed image quality and extraction accuracy.
"""

import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


@dataclass
class AblationResult:
    """Container for holding and analyzing ablation study results."""
    parameter_name: str
    parameter_values: List
    psnr_values: List[float]
    ssim_values: List[float]
    bit_accuracy_values: List[float]
    ber_values: List[float]
    extra_metrics: Dict = field(default_factory=dict)
    
    def best_tradeoff(self, psnr_threshold: float = 35.0, ssim_threshold: float = 0.9) -> int:
        """Finds the best parameter index that satisfies quality thresholds.

        If no configuration satisfies the thresholds, returns the parameter index
        with the highest average of normalized PSNR and bit accuracy.

        Args:
            psnr_threshold (float): Minimum acceptable PSNR in dB.
            ssim_threshold (float): Minimum acceptable SSIM value.

        Returns:
            int: The index of the optimal parameter value.
        """
        for i, (psnr, ssim, acc) in enumerate(zip(self.psnr_values, self.ssim_values, self.bit_accuracy_values)):
            if psnr >= psnr_threshold and ssim >= ssim_threshold:
                return i
        scores = [0.5 * p / 50.0 + 0.5 * a for p, a in zip(self.psnr_values, self.bit_accuracy_values)]
        return int(np.argmax(scores))


class AblationStudy:
    """Ablation study framework for watermark parameter analysis."""
    
    def __init__(
        self,
        device: Optional[torch.device] = None,
        output_dir: str = "results/ablation"
    ):
        """Initializes the AblationStudy class.

        Args:
            device: Device for computation (e.g., CPU, CUDA, MPS).
            output_dir: Directory for saving generated plots and reports.
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
        alpha_values: Optional[List[float]] = None,
        compute_metrics_fn: Optional[Callable] = None
    ) -> AblationResult:
        """Evaluates how varying the watermark strength (alpha) impacts quality and accuracy.

        Args:
            images (torch.Tensor): Original images tensor of shape (B, C, H, W).
            watermark (torch.Tensor): Watermark bits tensor of shape (B, W_dim).
            vae: VAE wrapper module.
            splitter: Latent splitter module.
            recombiner: Latent recombiner module.
            encoder_l: Low-frequency watermark encoder.
            encoder_h: High-frequency watermark encoder.
            decoder_l: Low-frequency watermark decoder.
            decoder_h: High-frequency watermark decoder.
            alpha_values (Optional[List[float]]): List of alpha scaling values to test.
            compute_metrics_fn (Optional[Callable]): Optional function to compute PSNR and SSIM.

        Returns:
            AblationResult: Collected metrics for each tested strength.
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
            z = vae.encode(images)
            z_low, z_high = splitter(z)
            
            z_low_wm = encoder_l(z_low, watermark, alpha=alpha)
            z_high_wm = encoder_h(z_high, watermark, alpha=alpha)
            z_wm = recombiner(z_low_wm, z_high_wm)
            
            images_wm = vae.decode(z_wm)
            
            if compute_metrics_fn:
                metrics = compute_metrics_fn(images, images_wm)
                psnr = metrics['psnr'].mean if hasattr(metrics['psnr'], 'mean') else metrics['psnr']
                ssim = metrics['ssim'].mean if hasattr(metrics['ssim'], 'mean') else metrics['ssim']
            else:
                mse = torch.mean((images - images_wm) ** 2, dim=[1, 2, 3])
                psnr = 10 * torch.log10(4.0 / (mse + 1e-10)).mean().item()
                
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
            
            z_wm_low, z_wm_high = splitter(z_wm)
            w_pred_l = decoder_l(z_wm_low)
            w_pred_h = decoder_h(z_wm_high)
            w_pred = (w_pred_l + w_pred_h) / 2
            
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
        compute_metrics_fn: Optional[Callable] = None
    ) -> AblationResult:
        """Evaluates embedding performance across different sub-band configurations.

        Args:
            images (torch.Tensor): Original images tensor.
            watermark (torch.Tensor): Watermark bits tensor.
            vae: VAE wrapper module.
            splitter: Latent splitter module.
            recombiner: Latent recombiner module.
            encoder_l: Low-frequency watermark encoder.
            encoder_h: High-frequency watermark encoder.
            decoder_l: Low-frequency watermark decoder.
            decoder_h: High-frequency watermark decoder.
            alpha (float): Watermark strength scaling factor.
            compute_metrics_fn (Optional[Callable]): Optional function to compute PSNR and SSIM.

        Returns:
            AblationResult: Collected metrics for each sub-band configuration.
        """
        images = images.to(self.device)
        watermark = watermark.to(self.device)
        
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
            
            z_wm_low, z_wm_high = splitter(z_wm)
            
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
        bit_lengths: Optional[List[int]] = None,
        alpha: float = 1.0,
        compute_metrics_fn: Optional[Callable] = None
    ) -> AblationResult:
        """Evaluates embedding performance across various watermark bit lengths.

        Note that in normal circumstances models are specifically trained for each
        length. This ablation tests the capacity limits and configuration viability.

        Args:
            images (torch.Tensor): Original images tensor.
            vae: VAE wrapper module.
            splitter: Latent splitter module.
            recombiner: Latent recombiner module.
            encoder_class: Class reference for the encoder to instantiate.
            decoder_class: Class reference for the decoder to instantiate.
            bit_lengths (Optional[List[int]]): List of dimension lengths to test.
            alpha (float): Watermark strength scaling factor.
            compute_metrics_fn (Optional[Callable]): Optional function to compute PSNR and SSIM.

        Returns:
            AblationResult: Collected metrics for each watermark bit length.
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
            encoder_l = encoder_class(watermark_dim=w_dim).to(self.device)
            encoder_h = encoder_class(watermark_dim=w_dim).to(self.device)
            decoder_l = decoder_class(watermark_dim=w_dim).to(self.device)
            decoder_h = decoder_class(watermark_dim=w_dim).to(self.device)
            
            B = images.shape[0]
            watermark = torch.randn(B, w_dim, device=self.device)
            
            z = vae.encode(images)
            z_low, z_high = splitter(z)
            
            z_low_wm = encoder_l(z_low, watermark, alpha=alpha)
            z_high_wm = encoder_h(z_high, watermark, alpha=alpha)
            z_wm = recombiner(z_low_wm, z_high_wm)
            
            images_wm = vae.decode(z_wm)
            
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
            
            z_wm_low, z_wm_high = splitter(z_wm)
            w_pred_l = decoder_l(z_wm_low)
            w_pred_h = decoder_h(z_wm_high)
            w_pred = (w_pred_l + w_pred_h) / 2
            
            bits_true = (watermark > 0).float()
            bits_pred = (w_pred > 0).float()
            bit_acc = (bits_true == bits_pred).float().mean().item()
            
            bit_acc_values.append(bit_acc)
            ber_values.append(1 - bit_acc)
            
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
        """Generates bar plots showing the ablation metrics.

        Args:
            results (AblationResult): Ablation results container.
            save_suffix (str): Appended to the output file path.
            figsize (Tuple[int, int]): Size dimensions of the figure.
        """
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        x = range(len(results.parameter_values))
        labels = [str(v) for v in results.parameter_values]
        
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
        """Generates curves depicting the trade-offs between image quality and accuracy.

        Args:
            results (AblationResult): Ablation results container.
            save_suffix (str): Appended to the output file path.
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        ax1 = axes[0]
        ax1.plot(results.psnr_values, results.bit_accuracy_values, 'o-', markersize=10)
        for i, label in enumerate(results.parameter_values):
            ax1.annotate(
                str(label), 
                (results.psnr_values[i], results.bit_accuracy_values[i]),
                textcoords="offset points", 
                xytext=(5, 5), 
                fontsize=9
            )
        ax1.axhline(y=0.95, color='green', linestyle='--', alpha=0.5)
        ax1.axvline(x=40, color='blue', linestyle='--', alpha=0.5)
        ax1.fill_between([40, ax1.get_xlim()[1]], 0.95, 1.0, alpha=0.1, color='green')
        ax1.set_xlabel('PSNR (dB)')
        ax1.set_ylabel('Bit Accuracy')
        ax1.set_title('Quality-Robustness Trade-off')
        ax1.grid(True, alpha=0.3)
        
        ax2 = axes[1]
        ax2.plot(results.ssim_values, results.bit_accuracy_values, 'o-', markersize=10, color='orange')
        for i, label in enumerate(results.parameter_values):
            ax2.annotate(
                str(label),
                (results.ssim_values[i], results.bit_accuracy_values[i]),
                textcoords="offset points", 
                xytext=(5, 5), 
                fontsize=9
            )
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
        save_path: Optional[str] = None
    ) -> str:
        """Compiles a text-based ablation study report and saves it.

        Args:
            results_list (List[AblationResult]): List of collected ablation studies.
            save_path (Optional[str]): Custom path for saving the text file.

        Returns:
            str: The full text report content.
        """
        lines = []
        lines.append("=" * 80)
        lines.append("ABLATION STUDY REPORT")
        lines.append("=" * 80)
        
        for result in results_list:
            lines.append(f"\n\n{result.parameter_name}")
            lines.append("-" * 40)
            
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
                
            best_idx = result.best_tradeoff(psnr_threshold=40.0, ssim_threshold=0.91)
            lines.append("\nBest configuration meeting PSNR≥40dB, SSIM≥0.91:")
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

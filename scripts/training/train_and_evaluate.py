#!/usr/bin/env python3
"""
Comprehensive Training and Evaluation Script for Latent Watermarking.

This script:
1. Trains the watermark encoder/decoder on 10k images
2. Targets 40dB PSNR and 0.91 SSIM
3. Runs comprehensive evaluation including:
   - Image quality metrics (PSNR, SSIM, LPIPS, FID)
   - Robustness evaluation under various attacks
   - Statistical analysis
   - Ablation studies
   - Computational analysis
   - Qualitative visualizations
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import yaml
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Subset
import numpy as np
from tqdm import tqdm
from datetime import datetime

# Model imports
from latent_watermarking.models.vae_wrapper import VAEWrapper
from latent_watermarking.models.latent_split import LatentSplitter
from latent_watermarking.models.watermark_encoder import WatermarkEncoder
from latent_watermarking.models.watermark_decoder import WatermarkDecoder
from latent_watermarking.models.recombination import LatentRecombiner

# Attack imports
from latent_watermarking.attacks.latent_noise import LatentNoiseAttack
from latent_watermarking.attacks.jpeg_sim import JpegSimAttack
from latent_watermarking.attacks.resize_crop import ResizeCropAttack

# Training imports
from latent_watermarking.training.losses import WatermarkLosses
from latent_watermarking.training.dataset import MirflickrDataset

# Evaluation imports
from latent_watermarking.evaluation.metrics import ImageQualityMetrics
from latent_watermarking.evaluation.robustness import RobustnessEvaluator
from latent_watermarking.evaluation.statistics import StatisticalAnalysis
from latent_watermarking.evaluation.ablation import AblationStudy
from latent_watermarking.evaluation.computational import ComputationalAnalysis
from latent_watermarking.evaluation.qualitative import QualitativeExperiments

from torchvision import transforms
from torchvision.utils import save_image


class WatermarkTrainer:
    """Enhanced trainer with quality monitoring and adaptive alpha adjustment."""
    
    def __init__(self, config, models, attacks, losses, dataloader, device):
        self.config = config
        self.models = models
        self.attacks = attacks
        self.losses = losses
        self.dataloader = dataloader
        self.device = device
        
        # Target metrics
        self.target_psnr = config.get('target_psnr', 40.0)
        self.target_ssim = config.get('target_ssim', 0.91)
        
        # Current alpha values (can be adjusted during training)
        self.alpha_l = config.get('alpha_l', 0.3)
        self.alpha_h = config.get('alpha_h', 0.15)
        
        # Optimizer
        params = list(models['encoder_l'].parameters()) + \
                 list(models['encoder_h'].parameters()) + \
                 list(models['decoder_l'].parameters()) + \
                 list(models['decoder_h'].parameters())
                 
        if config.get('latent_split') == 'learned':
            params += list(models['splitter'].parameters())
            params += list(models['recombiner'].parameters())
            
        self.optimizer = torch.optim.AdamW(params, lr=config['lr'])
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config['epochs'], eta_min=1e-6
        )
        
        # Metrics tracking
        self.metrics_history = {
            'psnr': [], 'ssim': [], 'bit_acc': [], 'loss': []
        }
        
        # Quality metrics
        self.quality_metrics = ImageQualityMetrics(device)
        
    def set_train_mode(self):
        self.models['encoder_l'].train()
        self.models['encoder_h'].train()
        self.models['decoder_l'].train()
        self.models['decoder_h'].train()
        self.models['vae'].eval()  # VAE stays frozen
        
    def set_eval_mode(self):
        for model in self.models.values():
            model.eval()
            
    def compute_image_quality(self, images_orig, images_wm):
        """Compute PSNR and SSIM."""
        # PSNR
        mse = torch.mean((images_orig - images_wm) ** 2, dim=[1, 2, 3])
        psnr = 10 * torch.log10(4.0 / (mse + 1e-10))  # data_range=2
        
        # Simple SSIM
        C1 = (0.01 * 2) ** 2
        C2 = (0.03 * 2) ** 2
        
        mu1 = F.avg_pool2d(images_orig, kernel_size=11, stride=1, padding=5)
        mu2 = F.avg_pool2d(images_wm, kernel_size=11, stride=1, padding=5)
        
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.avg_pool2d(images_orig ** 2, kernel_size=11, stride=1, padding=5) - mu1_sq
        sigma2_sq = F.avg_pool2d(images_wm ** 2, kernel_size=11, stride=1, padding=5) - mu2_sq
        sigma12 = F.avg_pool2d(images_orig * images_wm, kernel_size=11, stride=1, padding=5) - mu1_mu2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        ssim = ssim_map.mean(dim=[1, 2, 3])
        
        return psnr.mean().item(), ssim.mean().item()
        
    def adjust_alpha(self, psnr, ssim):
        """Adaptively adjust alpha to meet quality targets."""
        adjustment_rate = 0.05
        
        if psnr < self.target_psnr or ssim < self.target_ssim:
            # Reduce alpha to improve quality
            self.alpha_l = max(0.1, self.alpha_l * (1 - adjustment_rate))
            self.alpha_h = max(0.05, self.alpha_h * (1 - adjustment_rate))
        elif psnr > self.target_psnr + 5 and ssim > self.target_ssim + 0.05:
            # Can increase alpha for better robustness
            self.alpha_l = min(1.0, self.alpha_l * (1 + adjustment_rate))
            self.alpha_h = min(0.5, self.alpha_h * (1 + adjustment_rate))
            
    def train_epoch(self, epoch):
        self.set_train_mode()
        
        epoch_losses = []
        epoch_psnr = []
        epoch_ssim = []
        epoch_bit_acc = []
        
        pbar = tqdm(self.dataloader, desc=f"Epoch {epoch}")
        for batch_idx, batch in enumerate(pbar):
            if isinstance(batch, (list, tuple)):
                images = batch[0].to(self.device)
            else:
                images = batch.to(self.device)
                
            B = images.shape[0]
            
            # Forward pass
            z = self.models['vae'].encode(images)
            z_low, z_high = self.models['splitter'](z)
            
            # Generate watermark
            w = torch.randn(B, self.config['w_dim'], device=self.device)
            
            # Inject watermark
            z_low_wm = self.models['encoder_l'](z_low, w, alpha=self.alpha_l)
            z_high_wm = self.models['encoder_h'](z_high, w, alpha=self.alpha_h)
            z_wm = self.models['recombiner'](z_low_wm, z_high_wm)
            
            # Decode watermark (clean)
            z_wm_low, z_wm_high = self.models['splitter'](z_wm)
            w_pred_l = self.models['decoder_l'](z_wm_low)
            w_pred_h = self.models['decoder_h'](z_wm_high)
            
            # Apply attack and decode (robustness)
            attack_idx = torch.randint(0, len(self.attacks), (1,)).item()
            attack = self.attacks[attack_idx]
            attack.train()  # Enable attack
            z_attacked = attack(z_wm)
            
            z_att_low, z_att_high = self.models['splitter'](z_attacked)
            w_pred_rob_l = self.models['decoder_l'](z_att_low)
            w_pred_rob_h = self.models['decoder_h'](z_att_high)
            
            # Decode to images for perceptual loss
            with torch.no_grad():
                images_wm = self.models['vae'].decode(z_wm)
            
            # Compute loss
            loss, loss_dict = self.losses(
                w, w_pred_l, w_pred_h, z, z_wm,
                w_pred_rob_l, w_pred_rob_h,
                images, images_wm
            )
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if self.config.get('grad_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    list(self.models['encoder_l'].parameters()) +
                    list(self.models['encoder_h'].parameters()) +
                    list(self.models['decoder_l'].parameters()) +
                    list(self.models['decoder_h'].parameters()),
                    self.config['grad_clip']
                )
                
            self.optimizer.step()
            
            # Compute metrics
            with torch.no_grad():
                images_wm_eval = self.models['vae'].decode(z_wm)
                psnr, ssim = self.compute_image_quality(images, images_wm_eval)
                
                # Bit accuracy
                bits_true = (w > 0).float()
                bits_pred = ((w_pred_l + w_pred_h) / 2 > 0).float()
                bit_acc = (bits_true == bits_pred).float().mean().item()
                
            epoch_losses.append(loss.item())
            epoch_psnr.append(psnr)
            epoch_ssim.append(ssim)
            epoch_bit_acc.append(bit_acc)
            
            pbar.set_postfix(
                loss=loss.item(),
                psnr=f"{psnr:.1f}",
                ssim=f"{ssim:.3f}",
                bit_acc=f"{bit_acc:.3f}",
                alpha_l=f"{self.alpha_l:.3f}"
            )
            
            # Memory cleanup for MPS
            if self.device.type == 'mps':
                torch.mps.empty_cache()
            
        # Update learning rate
        self.scheduler.step()
        
        # Epoch statistics
        avg_loss = np.mean(epoch_losses)
        avg_psnr = np.mean(epoch_psnr)
        avg_ssim = np.mean(epoch_ssim)
        avg_bit_acc = np.mean(epoch_bit_acc)
        
        # Store metrics
        self.metrics_history['loss'].append(avg_loss)
        self.metrics_history['psnr'].append(avg_psnr)
        self.metrics_history['ssim'].append(avg_ssim)
        self.metrics_history['bit_acc'].append(avg_bit_acc)
        
        # Adaptive alpha adjustment
        self.adjust_alpha(avg_psnr, avg_ssim)
        
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Loss: {avg_loss:.4f}")
        print(f"  PSNR: {avg_psnr:.2f} dB (target: {self.target_psnr} dB)")
        print(f"  SSIM: {avg_ssim:.4f} (target: {self.target_ssim})")
        print(f"  Bit Accuracy: {avg_bit_acc:.4f}")
        print(f"  Alpha L/H: {self.alpha_l:.3f}/{self.alpha_h:.3f}")
        
        return avg_loss, avg_psnr, avg_ssim, avg_bit_acc
        
    def save_checkpoint(self, epoch, save_path):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'encoder_l': self.models['encoder_l'].state_dict(),
            'encoder_h': self.models['encoder_h'].state_dict(),
            'decoder_l': self.models['decoder_l'].state_dict(),
            'decoder_h': self.models['decoder_h'].state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'alpha_l': self.alpha_l,
            'alpha_h': self.alpha_h,
            'metrics_history': self.metrics_history,
            'config': self.config
        }
        
        if self.config.get('latent_split') == 'learned':
            checkpoint['splitter'] = self.models['splitter'].state_dict()
            checkpoint['recombiner'] = self.models['recombiner'].state_dict()
            
        torch.save(checkpoint, save_path)
        print(f"Checkpoint saved to {save_path}")


def run_evaluation(models, config, dataloader, device, output_dir):
    """Run comprehensive evaluation."""
    
    print("\n" + "=" * 60)
    print("RUNNING COMPREHENSIVE EVALUATION")
    print("=" * 60)
    
    vae = models['vae']
    splitter = models['splitter']
    recombiner = models['recombiner']
    encoder_l = models['encoder_l']
    encoder_h = models['encoder_h']
    decoder_l = models['decoder_l']
    decoder_h = models['decoder_h']
    
    # Set eval mode
    for model in models.values():
        model.eval()
        
    # Collect test images
    print("\nCollecting test images...")
    test_images = []
    watermarks = []
    images_wm = []
    
    max_test_samples = min(500, len(dataloader.dataset))
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Processing test data"):
            if isinstance(batch, (list, tuple)):
                images = batch[0].to(device)
            else:
                images = batch.to(device)
                
            B = images.shape[0]
            
            # Generate and embed watermarks
            w = torch.randn(B, config['w_dim'], device=device)
            
            z = vae.encode(images)
            z_low, z_high = splitter(z)
            
            z_low_wm = encoder_l(z_low, w, alpha=config['alpha_l'])
            z_high_wm = encoder_h(z_high, w, alpha=config['alpha_h'])
            z_wm = recombiner(z_low_wm, z_high_wm)
            
            img_wm = vae.decode(z_wm)
            
            test_images.append(images.cpu())
            watermarks.append(w.cpu())
            images_wm.append(img_wm.cpu())
            
            if sum(t.shape[0] for t in test_images) >= max_test_samples:
                break
                
    test_images = torch.cat(test_images, dim=0)[:max_test_samples]
    watermarks = torch.cat(watermarks, dim=0)[:max_test_samples]
    images_wm = torch.cat(images_wm, dim=0)[:max_test_samples]
    
    # 1. Image Quality Metrics
    print("\n" + "-" * 40)
    print("1. IMAGE QUALITY METRICS")
    print("-" * 40)
    
    metrics = ImageQualityMetrics(device)
    quality_results = metrics.evaluate(
        test_images.to(device), 
        images_wm.to(device),
        compute_fid=len(test_images) >= 100
    )
    metrics.print_results(quality_results)
    
    # Save results
    quality_summary = {
        'psnr_mean': quality_results['psnr'].mean,
        'psnr_std': quality_results['psnr'].std,
        'ssim_mean': quality_results['ssim'].mean,
        'ssim_std': quality_results['ssim'].std,
        'lpips_mean': quality_results['lpips'].mean,
        'lpips_std': quality_results['lpips'].std,
        'fid': quality_results['fid']
    }
    
    with open(os.path.join(output_dir, 'quality_metrics.json'), 'w') as f:
        json.dump(quality_summary, f, indent=2)
        
    # 2. Robustness Evaluation
    print("\n" + "-" * 40)
    print("2. ROBUSTNESS EVALUATION")
    print("-" * 40)
    
    robustness = RobustnessEvaluator(device, output_dir=os.path.join(output_dir, 'robustness'))
    
    def extract_watermark(z):
        z_low, z_high = splitter(z)
        w_l = decoder_l(z_low)
        w_h = decoder_h(z_high)
        return (w_l + w_h) / 2
        
    robustness_results = robustness.evaluate_all_attacks(
        images_wm.to(device)[:100],  # Use subset for speed
        watermarks.to(device)[:100],
        extract_watermark,
        vae_encode_fn=vae.encode,
        vae_decode_fn=vae.decode
    )
    
    # Generate robustness plots and report
    robustness.plot_roc_curves(robustness_results)
    robustness.plot_attack_severity(robustness_results)
    robustness.generate_report(robustness_results)
    
    # 3. Statistical Analysis
    print("\n" + "-" * 40)
    print("3. STATISTICAL ANALYSIS")
    print("-" * 40)
    
    stats = StatisticalAnalysis(output_dir=os.path.join(output_dir, 'statistics'))
    
    psnr_ci = stats.confidence_interval(quality_results['psnr'].values)
    ssim_ci = stats.confidence_interval(quality_results['ssim'].values)
    
    print(f"PSNR 95% CI: {psnr_ci}")
    print(f"SSIM 95% CI: {ssim_ci}")
    
    # Trade-off analysis
    # Collect results at different alpha values
    alpha_psnr = []
    alpha_acc = []
    alpha_values = [0.1, 0.2, 0.3, 0.5, 0.75, 1.0]
    
    for alpha in alpha_values:
        with torch.no_grad():
            z = vae.encode(test_images[:50].to(device))
            z_low, z_high = splitter(z)
            z_low_wm = encoder_l(z_low, watermarks[:50].to(device), alpha=alpha)
            z_high_wm = encoder_h(z_high, watermarks[:50].to(device), alpha=alpha * 0.5)
            z_wm = recombiner(z_low_wm, z_high_wm)
            img_wm = vae.decode(z_wm)
            
            # PSNR
            mse = torch.mean((test_images[:50].to(device) - img_wm) ** 2, dim=[1,2,3])
            psnr = 10 * torch.log10(4.0 / (mse + 1e-10)).mean().item()
            
            # Accuracy
            w_pred = extract_watermark(z_wm)
            bits_true = (watermarks[:50].to(device) > 0).float()
            bits_pred = (w_pred > 0).float()
            acc = (bits_true == bits_pred).float().mean().item()
            
            alpha_psnr.append(psnr)
            alpha_acc.append(acc)
            
    tradeoff = stats.analyze_tradeoff(
        np.array(alpha_psnr),
        np.array(alpha_acc),
        method_names=[f"α={a}" for a in alpha_values],
        metric_names=("PSNR (dB)", "Bit Accuracy")
    )
    
    stats.plot_tradeoff(
        tradeoff,
        np.array(alpha_psnr),
        np.array(alpha_acc),
        method_names=[f"α={a}" for a in alpha_values]
    )
    
    stats_report = stats.generate_statistics_report({
        'PSNR': np.array(quality_results['psnr'].values),
        'SSIM': np.array(quality_results['ssim'].values),
        'LPIPS': np.array(quality_results['lpips'].values)
    })
    print(stats_report)
    
    # 4. Ablation Studies
    print("\n" + "-" * 40)
    print("4. ABLATION STUDIES")
    print("-" * 40)
    
    ablation = AblationStudy(device, output_dir=os.path.join(output_dir, 'ablation'))
    
    # Alpha ablation
    alpha_result = ablation.ablate_watermark_strength(
        test_images[:20].to(device),
        watermarks[:20].to(device),
        vae, splitter, recombiner,
        encoder_l, encoder_h, decoder_l, decoder_h
    )
    ablation.plot_ablation_results(alpha_result)
    ablation.plot_tradeoff_curves(alpha_result)
    
    # Layer ablation
    layer_result = ablation.ablate_embedding_layer(
        test_images[:20].to(device),
        watermarks[:20].to(device),
        vae, splitter, recombiner,
        encoder_l, encoder_h, decoder_l, decoder_h
    )
    ablation.plot_ablation_results(layer_result)
    
    ablation.generate_ablation_report([alpha_result, layer_result])
    
    # 5. Computational Analysis
    print("\n" + "-" * 40)
    print("5. COMPUTATIONAL ANALYSIS")
    print("-" * 40)
    
    computational = ComputationalAnalysis(device, output_dir=os.path.join(output_dir, 'computational'))
    
    comp_results = computational.analyze_watermark_pipeline(
        vae, splitter, recombiner,
        encoder_l, encoder_h, decoder_l, decoder_h,
        image_size=config['image_size'],
        batch_sizes=[1, 2, 4],
        n_iterations=20
    )
    
    computational.plot_results(comp_results, batch_sizes=[1, 2, 4])
    comp_report = computational.generate_report(comp_results, batch_sizes=[1, 2, 4])
    print(comp_report)
    
    # 6. Qualitative Experiments
    print("\n" + "-" * 40)
    print("6. QUALITATIVE EXPERIMENTS")
    print("-" * 40)
    
    qualitative = QualitativeExperiments(device, output_dir=os.path.join(output_dir, 'qualitative'))
    
    # Collect attacked images and predictions for visualization
    attacked_images = {}
    wm_predictions = {'Clean': None}
    
    with torch.no_grad():
        sample_images = test_images[:8].to(device)
        sample_wm = images_wm[:8].to(device)
        sample_w = watermarks[:8].to(device)
        
        # Clean prediction
        z = vae.encode(sample_wm)
        wm_predictions['Clean'] = extract_watermark(z)
        
        # Some attack predictions
        from latent_watermarking.evaluation.robustness import JPEGCompression, GaussianNoise, GaussianBlur
        
        attacks_vis = [
            ('JPEG Q50', JPEGCompression(50)),
            ('Noise 0.05', GaussianNoise(0.05)),
            ('Blur K5', GaussianBlur(5))
        ]
        
        for name, attack in attacks_vis:
            attack = attack.to(device)
            img_attacked = attack(sample_wm)
            attacked_images[name] = img_attacked.cpu()
            z_attacked = vae.encode(img_attacked)
            wm_predictions[name] = extract_watermark(z_attacked)
            
    # Generate visualizations
    qualitative.plot_side_by_side(
        sample_images.cpu(),
        sample_wm.cpu(),
        n_samples=4
    )
    
    qualitative.plot_difference_maps(
        sample_images.cpu(),
        sample_wm.cpu(),
        n_samples=4
    )
    
    qualitative.plot_bit_distributions(
        sample_w.cpu(),
        {k: v.cpu() for k, v in wm_predictions.items()}
    )
    
    qualitative.plot_attack_progression(
        sample_images.cpu(),
        sample_wm.cpu(),
        attacked_images,
        {k: v.cpu() for k, v in wm_predictions.items()},
        sample_w.cpu()
    )
    
    # Latent visualization
    with torch.no_grad():
        z_orig = vae.encode(sample_images)
        z_wm = vae.encode(sample_wm)
        
    qualitative.plot_latent_visualization(z_orig.cpu(), z_wm.cpu())
    
    print(f"\nAll evaluation results saved to {output_dir}")
    
    # Final summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"PSNR: {quality_results['psnr'].mean:.2f} ± {quality_results['psnr'].std:.2f} dB")
    print(f"SSIM: {quality_results['ssim'].mean:.4f} ± {quality_results['ssim'].std:.4f}")
    print(f"LPIPS: {quality_results['lpips'].mean:.4f} ± {quality_results['lpips'].std:.4f}")
    if quality_results['fid']:
        print(f"FID: {quality_results['fid']:.2f}")
        
    # Check targets
    meets_psnr = quality_results['psnr'].mean >= 40.0
    meets_ssim = quality_results['ssim'].mean >= 0.91
    
    print("\nTarget Achievement:")
    print(f"  PSNR ≥ 40 dB: {'✓' if meets_psnr else '✗'} ({quality_results['psnr'].mean:.2f} dB)")
    print(f"  SSIM ≥ 0.91: {'✓' if meets_ssim else '✗'} ({quality_results['ssim'].mean:.4f})")
    
    return quality_results


def main():
    # Get the directory containing this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    parser = argparse.ArgumentParser(description='Train and evaluate latent watermarking')
    parser.add_argument('--config', type=str, default=os.path.join(script_dir, 'configs/default.yaml'))
    parser.add_argument('--checkpoint', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--eval-only', action='store_true', help='Run evaluation only')
    parser.add_argument('--output-dir', type=str, default=os.path.join(script_dir, 'results'), help='Output directory')
    parser.add_argument('--num-images', type=int, default=10000, help='Number of training images')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    # Setup device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output_dir, f'run_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    
    # Save config
    with open(os.path.join(output_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)
        
    # Initialize models
    print("\nInitializing models...")
    vae = VAEWrapper().to(device)
    splitter = LatentSplitter(mode=config['latent_split']).to(device)
    recombiner = LatentRecombiner(mode=config['latent_split']).to(device)
    
    encoder_l = WatermarkEncoder(watermark_dim=config['w_dim']).to(device)
    encoder_h = WatermarkEncoder(watermark_dim=config['w_dim']).to(device)
    decoder_l = WatermarkDecoder(watermark_dim=config['w_dim']).to(device)
    decoder_h = WatermarkDecoder(watermark_dim=config['w_dim']).to(device)
    
    models = {
        'vae': vae,
        'splitter': splitter,
        'recombiner': recombiner,
        'encoder_l': encoder_l,
        'encoder_h': encoder_h,
        'decoder_l': decoder_l,
        'decoder_h': decoder_h
    }
    
    # Initialize attacks - use only lightweight attack for M2 Mac
    attacks = [
        LatentNoiseAttack().to(device),
        # JpegSimAttack(vae).to(device),  # Disabled to reduce memory
        # ResizeCropAttack(vae).to(device)  # Disabled to reduce memory
    ]
    
    # Initialize losses
    losses = WatermarkLosses(
        lambda_w=config['lambda_w'],
        lambda_cons=config['lambda_cons'],
        lambda_latent=config['lambda_latent'],
        lambda_robust=config['lambda_robust'],
        lambda_perceptual=config.get('lambda_perceptual', 0.0),
        device=device
    ).to(device)
    
    # Load dataset
    print("\nLoading dataset...")
    transform = transforms.Compose([
        transforms.Resize((config['image_size'], config['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    dataset_path = 'latent_watermarking/mirflickr'
    if os.path.exists(dataset_path):
        print(f"Loading Mirflickr dataset from {dataset_path}")
        dataset = MirflickrDataset(dataset_path, transform=transform, limit=args.num_images)
    else:
        print("Mirflickr dataset not found, using synthetic data for testing")
        # Use smaller synthetic dataset for M2 Mac
        num_samples = min(args.num_images, 100)  # Limit for memory
        dummy_data = torch.randn(num_samples, 3, config['image_size'], config['image_size'])
        dataset = TensorDataset(dummy_data)
        
    print(f"Dataset size: {len(dataset)} images")
    
    dataloader = DataLoader(
        dataset, 
        batch_size=config['batch_size'], 
        shuffle=True,
        num_workers=config.get('num_workers', 0),
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # Load checkpoint if provided
    start_epoch = 0
    if args.checkpoint:
        print(f"\nLoading checkpoint from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        encoder_l.load_state_dict(checkpoint['encoder_l'])
        encoder_h.load_state_dict(checkpoint['encoder_h'])
        decoder_l.load_state_dict(checkpoint['decoder_l'])
        decoder_h.load_state_dict(checkpoint['decoder_h'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        config['alpha_l'] = checkpoint.get('alpha_l', config['alpha_l'])
        config['alpha_h'] = checkpoint.get('alpha_h', config['alpha_h'])
        print(f"Resuming from epoch {start_epoch}")
        
    # Run evaluation only if requested
    if args.eval_only:
        run_evaluation(models, config, dataloader, device, output_dir)
        return
        
    # Initialize trainer
    trainer = WatermarkTrainer(config, models, attacks, losses, dataloader, device)
    
    # Training loop
    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)
    print(f"Target PSNR: {config['target_psnr']} dB")
    print(f"Target SSIM: {config['target_ssim']}")
    print(f"Training images: {len(dataset)}")
    print(f"Epochs: {config['epochs']}")
    
    best_combined_score = 0
    
    for epoch in range(start_epoch, config['epochs']):
        avg_loss, avg_psnr, avg_ssim, avg_bit_acc = trainer.train_epoch(epoch)
        
        # Save checkpoint
        if (epoch + 1) % config.get('save_interval', 5) == 0:
            trainer.save_checkpoint(
                epoch,
                os.path.join(output_dir, f'checkpoint_epoch_{epoch}.pth')
            )
            
        # Track best model based on combined score
        combined_score = avg_bit_acc * (avg_psnr / 40) * (avg_ssim / 0.91)
        if combined_score > best_combined_score:
            best_combined_score = combined_score
            trainer.save_checkpoint(epoch, os.path.join(output_dir, 'best_model.pth'))
            
        # Run evaluation at intervals
        if (epoch + 1) % config.get('eval_interval', 10) == 0:
            eval_output = os.path.join(output_dir, f'eval_epoch_{epoch}')
            os.makedirs(eval_output, exist_ok=True)
            run_evaluation(models, config, dataloader, device, eval_output)
            
    # Final checkpoint
    trainer.save_checkpoint(config['epochs'] - 1, os.path.join(output_dir, 'final_model.pth'))
    
    # Final evaluation
    print("\n" + "=" * 60)
    print("FINAL EVALUATION")
    print("=" * 60)
    
    final_eval_output = os.path.join(output_dir, 'final_evaluation')
    os.makedirs(final_eval_output, exist_ok=True)
    
    quality_results = run_evaluation(models, config, dataloader, device, final_eval_output)
    
    print("\nTraining complete!")
    print(f"Results saved to: {output_dir}")


if __name__ == '__main__':
    main()

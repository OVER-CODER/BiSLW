#!/usr/bin/env python3
"""
Attack-Aware Training using existing roundtrip cache.
Applies batch-wise attacks for speed. Saves frequent checkpoints.

Approach:
- Phase 1: Light attacks (JPEG-90, Noise-0.01) - 20 epochs
- Phase 2: Medium attacks (JPEG-70, Noise-0.03) - 20 epochs  
- Phase 3: Hard attacks (JPEG-50, Noise-0.05, Blur) - 20 epochs
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from datetime import datetime
from torch.utils.data import DataLoader, TensorDataset
import io
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from latent_watermarking.models.latent_split import LatentSplitter
from latent_watermarking.models.recombination import LatentRecombiner
from latent_watermarking.models.watermark_encoder import WatermarkEncoder
from latent_watermarking.models.watermark_decoder import WatermarkDecoder


# ============================================================
# BATCH-WISE ATTACKS (faster than per-sample)
# ============================================================

def jpeg_attack_batch(images, quality=70):
    """Simulated JPEG via downscale-upscale (fast, batch-wise)."""
    scale = max(0.3, quality / 100)
    B, C, H, W = images.shape
    h_s, w_s = max(4, int(H * scale)), max(4, int(W * scale))
    down = F.interpolate(images, size=(h_s, w_s), mode='bilinear', align_corners=False)
    up = F.interpolate(down, size=(H, W), mode='bilinear', align_corners=False)
    # Add quantization noise
    noise = torch.randn_like(images) * (1 - quality/100) * 0.1
    return ((quality/100) * images + (1 - quality/100) * up + noise).clamp(-1, 1)


def gaussian_noise_attack(images, sigma=0.05):
    """Fast Gaussian noise."""
    return (images + torch.randn_like(images) * sigma).clamp(-1, 1)


def gaussian_blur_attack(images, kernel_size=5):
    """Fast Gaussian blur."""
    if kernel_size % 2 == 0:
        kernel_size += 1
    sigma = kernel_size / 3
    x = torch.arange(kernel_size, device=images.device, dtype=images.dtype) - kernel_size // 2
    kernel_1d = torch.exp(-x**2 / (2 * sigma**2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = kernel_1d.unsqueeze(1) * kernel_1d.unsqueeze(0)
    kernel = kernel_2d.unsqueeze(0).unsqueeze(0).expand(3, 1, -1, -1)
    pad = kernel_size // 2
    return F.conv2d(images, kernel, padding=pad, groups=3)


# Attack phases
PHASE_1_ATTACKS = [
    ('clean', None, 0.5),
    ('jpeg_90', lambda x: jpeg_attack_batch(x, 90), 0.25),
    ('noise_0.01', lambda x: gaussian_noise_attack(x, 0.01), 0.25),
]

PHASE_2_ATTACKS = [
    ('clean', None, 0.3),
    ('jpeg_70', lambda x: jpeg_attack_batch(x, 70), 0.3),
    ('noise_0.03', lambda x: gaussian_noise_attack(x, 0.03), 0.2),
    ('blur_3', lambda x: gaussian_blur_attack(x, 3), 0.2),
]

PHASE_3_ATTACKS = [
    ('clean', None, 0.2),
    ('jpeg_50', lambda x: jpeg_attack_batch(x, 50), 0.25),
    ('noise_0.05', lambda x: gaussian_noise_attack(x, 0.05), 0.25),
    ('blur_5', lambda x: gaussian_blur_attack(x, 5), 0.15),
    ('jpeg_30', lambda x: jpeg_attack_batch(x, 30), 0.15),
]


class VAEWrapper:
    """Lightweight VAE wrapper with lazy loading."""
    
    def __init__(self, device='cpu'):
        self.device = device
        self._vae = None
        self.scaling_factor = 0.18215
        
    def load(self):
        if self._vae is None:
            print("Loading VAE...")
            from diffusers import AutoencoderKL
            self._vae = AutoencoderKL.from_pretrained(
                'runwayml/stable-diffusion-v1-5',
                subfolder='vae',
                torch_dtype=torch.float32
            ).to(self.device)
            self._vae.eval()
            for p in self._vae.parameters():
                p.requires_grad = False
        return self._vae
    
    @torch.no_grad()
    def decode(self, z):
        vae = self.load()
        return vae.decode(z / self.scaling_factor).sample
    
    @torch.no_grad()
    def encode(self, img):
        vae = self.load()
        return vae.encode(img).latent_dist.mean * self.scaling_factor


class AttackAwareTrainer:
    """Trainer with phased attack-aware training."""
    
    def __init__(self, config, device):
        self.config = config
        self.device = device
        
        w_dim = config.get('w_dim', 32)
        
        self.splitter = LatentSplitter(mode=config.get('latent_split', 'dct')).to(device)
        self.recombiner = LatentRecombiner(mode=config.get('latent_split', 'dct')).to(device)
        
        self.encoder_l = WatermarkEncoder(watermark_dim=w_dim).to(device)
        self.encoder_h = WatermarkEncoder(watermark_dim=w_dim).to(device)
        self.decoder_l = WatermarkDecoder(watermark_dim=w_dim).to(device)
        self.decoder_h = WatermarkDecoder(watermark_dim=w_dim).to(device)
        
        self.alpha_l = config.get('alpha_l', 0.02)
        self.alpha_h = config.get('alpha_h', 0.01)
        
        self.all_params = (
            list(self.encoder_l.parameters()) +
            list(self.encoder_h.parameters()) +
            list(self.decoder_l.parameters()) +
            list(self.decoder_h.parameters())
        )
        
        self.vae = VAEWrapper(device)
        self.optimizer = None
        
    def setup_optimizer(self, lr=1e-4):
        self.optimizer = torch.optim.AdamW(self.all_params, lr=lr)
        return self.optimizer
        
    def embed_watermark(self, z, w):
        z_low, z_high = self.splitter(z)
        z_low_wm = self.encoder_l(z_low, w, alpha=self.alpha_l)
        z_high_wm = self.encoder_h(z_high, w, alpha=self.alpha_h)
        return self.recombiner(z_low_wm, z_high_wm)
    
    def extract_watermark(self, z):
        z_low, z_high = self.splitter(z)
        return self.decoder_l(z_low), self.decoder_h(z_high)
    
    def compute_bit_accuracy(self, w_true, w_pred_l, w_pred_h):
        bits_true = (w_true > 0).float()
        bits_pred = ((w_pred_l + w_pred_h) / 2 > 0).float()
        return (bits_true == bits_pred).float().mean().item()
    
    def train_step_with_attacks(self, z_orig, w_batch, attacks):
        """
        Training step applying attacks to images.
        Uses VAE decode→attack→encode pipeline.
        """
        # Embed watermark
        z_wm = self.embed_watermark(z_orig, w_batch)
        
        # Decode to image
        img_wm = self.vae.decode(z_wm)
        
        # Select attack based on probabilities
        attack_probs = [p for _, _, p in attacks]
        attack_probs = [p / sum(attack_probs) for p in attack_probs]
        attack_idx = np.random.choice(len(attacks), p=attack_probs)
        attack_name, attack_fn, _ = attacks[attack_idx]
        
        # Apply attack
        if attack_fn is not None:
            img_attacked = attack_fn(img_wm)
        else:
            img_attacked = img_wm
        
        # Re-encode
        z_attacked = self.vae.encode(img_attacked)
        
        # Extract watermark from attacked version
        w_pred_l, w_pred_h = self.extract_watermark(z_attacked)
        
        # Also extract from clean latent embedding
        w_clean_l, w_clean_h = self.extract_watermark(z_wm)
        
        # Losses
        loss_attacked = F.mse_loss(w_pred_l, w_batch) + F.mse_loss(w_pred_h, w_batch)
        loss_clean = F.mse_loss(w_clean_l, w_batch) + F.mse_loss(w_clean_h, w_batch)
        loss_cons = F.mse_loss(w_pred_l, w_pred_h) + F.mse_loss(w_clean_l, w_clean_h)
        loss_latent = F.mse_loss(z_wm, z_orig)
        
        total_loss = (
            2.0 * loss_attacked +
            0.3 * loss_clean +
            0.2 * loss_cons +
            0.5 * loss_latent
        )
        
        acc_attacked = self.compute_bit_accuracy(w_batch, w_pred_l, w_pred_h)
        acc_clean = self.compute_bit_accuracy(w_batch, w_clean_l, w_clean_h)
        
        return total_loss, {
            'loss': total_loss.item(),
            'acc_attacked': acc_attacked,
            'acc_clean': acc_clean,
            'attack': attack_name
        }
    
    def train_epoch(self, z_orig, watermarks, attacks, epoch, batch_size=4):
        """Train one epoch with specified attacks."""
        self._set_train_mode()
        
        n = len(z_orig)
        indices = torch.randperm(n)
        
        epoch_metrics = {'loss': [], 'acc_attacked': [], 'acc_clean': []}
        attack_counts = {}
        
        pbar = tqdm(range(0, n - batch_size, batch_size), desc=f"Epoch {epoch}")
        for start in pbar:
            idx = indices[start:start + batch_size]
            
            z_batch = z_orig[idx].to(self.device)
            w_batch = watermarks[idx].to(self.device)
            
            self.optimizer.zero_grad()
            loss, metrics = self.train_step_with_attacks(z_batch, w_batch, attacks)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.all_params, 1.0)
            self.optimizer.step()
            
            epoch_metrics['loss'].append(metrics['loss'])
            epoch_metrics['acc_attacked'].append(metrics['acc_attacked'])
            epoch_metrics['acc_clean'].append(metrics['acc_clean'])
            attack_counts[metrics['attack']] = attack_counts.get(metrics['attack'], 0) + 1
            
            pbar.set_postfix(
                loss=f"{metrics['loss']:.3f}",
                att=f"{metrics['acc_attacked']:.3f}"
            )
            
            if self.device.type == 'mps':
                torch.mps.empty_cache()
        
        return {
            'loss': np.mean(epoch_metrics['loss']),
            'acc_attacked': np.mean(epoch_metrics['acc_attacked']),
            'acc_clean': np.mean(epoch_metrics['acc_clean']),
            'attacks': attack_counts
        }
    
    def _set_train_mode(self):
        self.encoder_l.train()
        self.encoder_h.train()
        self.decoder_l.train()
        self.decoder_h.train()
    
    def _set_eval_mode(self):
        self.encoder_l.eval()
        self.encoder_h.eval()
        self.decoder_l.eval()
        self.decoder_h.eval()
    
    def save_checkpoint(self, path, epoch, phase, metrics):
        torch.save({
            'epoch': epoch,
            'phase': phase,
            'encoder_l': self.encoder_l.state_dict(),
            'encoder_h': self.encoder_h.state_dict(),
            'decoder_l': self.decoder_l.state_dict(),
            'decoder_h': self.decoder_h.state_dict(),
            'optimizer': self.optimizer.state_dict() if self.optimizer else None,
            'alpha_l': self.alpha_l,
            'alpha_h': self.alpha_h,
            'metrics': metrics,
            'config': self.config
        }, path)
        print(f"  Checkpoint saved: {path}")
    
    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.encoder_l.load_state_dict(checkpoint['encoder_l'])
        self.encoder_h.load_state_dict(checkpoint['encoder_h'])
        self.decoder_l.load_state_dict(checkpoint['decoder_l'])
        self.decoder_h.load_state_dict(checkpoint['decoder_h'])
        self.alpha_l = checkpoint.get('alpha_l', self.alpha_l)
        self.alpha_h = checkpoint.get('alpha_h', self.alpha_h)
        print(f"Loaded checkpoint from {path}")
        return checkpoint.get('epoch', 0), checkpoint.get('phase', 1)
    
    @torch.no_grad()
    def quick_eval(self, z_orig, watermarks, n_samples=50):
        """Quick evaluation under attacks."""
        self._set_eval_mode()
        
        results = {'clean': [], 'jpeg_70': [], 'noise_0.05': [], 'blur_5': []}
        attacks_eval = {
            'clean': None,
            'jpeg_70': lambda x: jpeg_attack_batch(x, 70),
            'noise_0.05': lambda x: gaussian_noise_attack(x, 0.05),
            'blur_5': lambda x: gaussian_blur_attack(x, 5),
        }
        
        indices = torch.randperm(len(z_orig))[:n_samples]
        
        for i in indices:
            z = z_orig[i:i+1].to(self.device)
            w = watermarks[i:i+1].to(self.device)
            
            z_wm = self.embed_watermark(z, w)
            img_wm = self.vae.decode(z_wm)
            
            for name, attack_fn in attacks_eval.items():
                if attack_fn is not None:
                    img_att = attack_fn(img_wm)
                else:
                    img_att = img_wm
                
                z_att = self.vae.encode(img_att)
                w_l, w_h = self.extract_watermark(z_att)
                acc = self.compute_bit_accuracy(w, w_l, w_h)
                results[name].append(acc)
            
            if self.device.type == 'mps':
                torch.mps.empty_cache()
        
        return {k: np.mean(v) for k, v in results.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--cache', type=str, default='cache/roundtrip_20000.pt')
    parser.add_argument('--resume', type=str, required=True)
    parser.add_argument('--samples', type=int, default=5000, help='Samples per phase')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--phase1-epochs', type=int, default=15)
    parser.add_argument('--phase2-epochs', type=int, default=15)
    parser.add_argument('--phase3-epochs', type=int, default=15)
    parser.add_argument('--checkpoint-interval', type=int, default=5)
    parser.add_argument('--start-phase', type=int, default=1, choices=[1, 2, 3])
    args = parser.parse_args()
    
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load cache
    print(f"Loading cache: {args.cache}")
    cache = torch.load(args.cache, map_location='cpu', weights_only=False)
    z_orig = cache['z_orig'][:args.samples]
    watermarks = cache['watermarks'][:args.samples]
    print(f"Using {len(z_orig)} samples")
    
    # Create trainer and load checkpoint
    trainer = AttackAwareTrainer(config, device)
    start_epoch, start_phase = trainer.load_checkpoint(args.resume)
    
    # Override start phase if specified
    if args.start_phase > start_phase:
        start_phase = args.start_phase
    
    # Output dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"results/attack_aware_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("PHASED ATTACK-AWARE TRAINING")
    print("="*60)
    print(f"Samples: {len(z_orig)}")
    print(f"Batch size: {args.batch_size}")
    print(f"Phases: 1 ({args.phase1_epochs}ep) → 2 ({args.phase2_epochs}ep) → 3 ({args.phase3_epochs}ep)")
    print(f"Checkpoint interval: {args.checkpoint_interval} epochs")
    print(f"Starting from phase: {start_phase}")
    print(f"Output: {output_dir}")
    print("="*60 + "\n")
    
    phases = [
        (1, PHASE_1_ATTACKS, args.phase1_epochs, 3e-5, "Light attacks"),
        (2, PHASE_2_ATTACKS, args.phase2_epochs, 2e-5, "Medium attacks"),
        (3, PHASE_3_ATTACKS, args.phase3_epochs, 1e-5, "Hard attacks"),
    ]
    
    best_acc = 0.0
    
    for phase_num, attacks, n_epochs, lr, desc in phases:
        if phase_num < start_phase:
            print(f"Skipping Phase {phase_num} (already completed)")
            continue
            
        print(f"\n{'='*60}")
        print(f"PHASE {phase_num}: {desc}")
        print(f"{'='*60}")
        print(f"Attacks: {[name for name, _, _ in attacks]}")
        print(f"Learning rate: {lr}")
        
        trainer.setup_optimizer(lr=lr)
        
        for epoch in range(1, n_epochs + 1):
            metrics = trainer.train_epoch(
                z_orig, watermarks, attacks,
                epoch, batch_size=args.batch_size
            )
            
            print(f"  Epoch {epoch}/{n_epochs}: Loss={metrics['loss']:.4f}, "
                  f"Attack Acc={metrics['acc_attacked']:.4f}, Clean Acc={metrics['acc_clean']:.4f}")
            
            # Save checkpoint
            if epoch % args.checkpoint_interval == 0:
                trainer.save_checkpoint(
                    f"{output_dir}/phase{phase_num}_epoch{epoch}.pt",
                    epoch, phase_num, metrics
                )
            
            # Track best
            if metrics['acc_attacked'] > best_acc:
                best_acc = metrics['acc_attacked']
                trainer.save_checkpoint(
                    f"{output_dir}/best.pt",
                    epoch, phase_num, metrics
                )
                print(f"    New best: {best_acc:.4f}")
        
        # Save phase checkpoint
        trainer.save_checkpoint(
            f"{output_dir}/phase{phase_num}_final.pt",
            n_epochs, phase_num, metrics
        )
        
        # Quick eval after phase
        print(f"\n  Phase {phase_num} evaluation:")
        eval_results = trainer.quick_eval(z_orig, watermarks, n_samples=30)
        for name, acc in eval_results.items():
            print(f"    {name}: {acc:.4f}")
    
    # Final checkpoint
    trainer.save_checkpoint(f"{output_dir}/final.pt", 0, 3, metrics)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Best attack accuracy: {best_acc:.4f}")
    print(f"Results saved to: {output_dir}")
    
    # Final evaluation
    print("\nFinal evaluation:")
    final_results = trainer.quick_eval(z_orig, watermarks, n_samples=50)
    for name, acc in final_results.items():
        print(f"  {name}: {acc:.4f}")


if __name__ == "__main__":
    main()

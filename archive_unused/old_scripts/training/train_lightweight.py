#!/usr/bin/env python3
"""
Ultra-lightweight attack-aware training.
No VAE operations - uses latent-space augmentation only.
~30 seconds per epoch, minimal system load.
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from latent_watermarking.models.latent_split import LatentSplitter
from latent_watermarking.models.recombination import LatentRecombiner
from latent_watermarking.models.watermark_encoder import WatermarkEncoder
from latent_watermarking.models.watermark_decoder import WatermarkDecoder


def latent_noise(z, sigma=0.1):
    """Add Gaussian noise in latent space."""
    return z + torch.randn_like(z) * sigma


def latent_blur(z, kernel_size=3):
    """Blur in latent space (simulates image blur effect)."""
    if kernel_size % 2 == 0:
        kernel_size += 1
    sigma = kernel_size / 3
    x = torch.arange(kernel_size, dtype=z.dtype, device=z.device) - kernel_size // 2
    k1d = torch.exp(-x**2 / (2 * sigma**2))
    k1d = k1d / k1d.sum()
    k2d = k1d.unsqueeze(1) @ k1d.unsqueeze(0)
    kernel = k2d.unsqueeze(0).unsqueeze(0).expand(4, 1, -1, -1)
    return F.conv2d(z, kernel, padding=kernel_size // 2, groups=4)


def latent_scale(z, scale=0.95):
    """Scale latent values (simulates compression artifacts)."""
    return z * scale


def latent_quantize(z, levels=32):
    """Quantize latents (simulates JPEG-like quantization)."""
    z_min, z_max = z.min(), z.max()
    z_norm = (z - z_min) / (z_max - z_min + 1e-8)
    z_quant = torch.round(z_norm * levels) / levels
    return z_quant * (z_max - z_min) + z_min


def latent_dropout(z, p=0.1):
    """Random dropout of latent channels."""
    mask = torch.rand_like(z) > p
    return z * mask


# Augmentation schedule
AUGMENTATIONS = [
    ('noise_0.05', lambda z: latent_noise(z, 0.05)),
    ('noise_0.1', lambda z: latent_noise(z, 0.1)),
    ('noise_0.15', lambda z: latent_noise(z, 0.15)),
    ('blur_3', lambda z: latent_blur(z, 3)),
    ('scale_0.95', lambda z: latent_scale(z, 0.95)),
    ('scale_0.9', lambda z: latent_scale(z, 0.9)),
    ('quant_32', lambda z: latent_quantize(z, 32)),
    ('quant_16', lambda z: latent_quantize(z, 16)),
    ('dropout_0.1', lambda z: latent_dropout(z, 0.1)),
]


class LightweightTrainer:
    def __init__(self, config, device):
        self.device = device
        w_dim = config.get('w_dim', 32)
        
        self.splitter = LatentSplitter(mode='dct').to(device)
        self.recombiner = LatentRecombiner(mode='dct').to(device)
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
        self.optimizer = torch.optim.AdamW(self.all_params, lr=5e-5)
        self.config = config
    
    def embed(self, z, w):
        z_l, z_h = self.splitter(z)
        z_l_wm = self.encoder_l(z_l, w, alpha=self.alpha_l)
        z_h_wm = self.encoder_h(z_h, w, alpha=self.alpha_h)
        return self.recombiner(z_l_wm, z_h_wm)
    
    def extract(self, z):
        z_l, z_h = self.splitter(z)
        return self.decoder_l(z_l), self.decoder_h(z_h)
    
    def bit_acc(self, w_true, w_l, w_h):
        bits_true = (w_true > 0).float()
        bits_pred = ((w_l + w_h) / 2 > 0).float()
        return (bits_true == bits_pred).float().mean().item()
    
    def train_epoch(self, z_orig, z_roundtrip, watermarks, epoch, batch_size=64):
        """Train one epoch with latent augmentation."""
        self.encoder_l.train()
        self.encoder_h.train()
        self.decoder_l.train()
        self.decoder_h.train()
        
        n = len(z_orig)
        indices = torch.randperm(n)
        
        losses, accs_clean, accs_aug = [], [], []
        
        pbar = tqdm(range(0, n - batch_size, batch_size), desc=f"Epoch {epoch}")
        for start in pbar:
            idx = indices[start:start + batch_size]
            
            z = z_orig[idx].to(self.device)
            z_rt = z_roundtrip[idx].to(self.device)
            w = watermarks[idx].to(self.device)
            
            # Embed watermark
            z_wm = self.embed(z, w)
            
            # Apply random augmentation to watermarked latent
            aug_idx = np.random.randint(len(AUGMENTATIONS))
            aug_name, aug_fn = AUGMENTATIONS[aug_idx]
            z_aug = aug_fn(z_wm)
            
            # Also use precomputed roundtrip (simulates VAE pass)
            # Mix with augmented version
            if np.random.rand() < 0.3:
                # Use roundtrip version (already has watermark from cache)
                z_aug = z_rt + torch.randn_like(z_rt) * 0.05
            
            # Extract from augmented
            w_l_aug, w_h_aug = self.extract(z_aug)
            
            # Extract from clean
            w_l_clean, w_h_clean = self.extract(z_wm)
            
            # Losses
            loss_aug = F.mse_loss(w_l_aug, w) + F.mse_loss(w_h_aug, w)
            loss_clean = F.mse_loss(w_l_clean, w) + F.mse_loss(w_h_clean, w)
            loss_cons = F.mse_loss(w_l_aug, w_h_aug)
            loss_latent = F.mse_loss(z_wm, z)
            
            loss = 1.5 * loss_aug + 0.5 * loss_clean + 0.2 * loss_cons + 0.3 * loss_latent
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.all_params, 1.0)
            self.optimizer.step()
            
            acc_clean = self.bit_acc(w, w_l_clean, w_h_clean)
            acc_aug = self.bit_acc(w, w_l_aug, w_h_aug)
            
            losses.append(loss.item())
            accs_clean.append(acc_clean)
            accs_aug.append(acc_aug)
            
            pbar.set_postfix(loss=f"{loss.item():.3f}", aug=f"{acc_aug:.3f}")
        
        return {
            'loss': np.mean(losses),
            'acc_clean': np.mean(accs_clean),
            'acc_aug': np.mean(accs_aug)
        }
    
    def save(self, path, epoch, metrics):
        torch.save({
            'epoch': epoch,
            'encoder_l': self.encoder_l.state_dict(),
            'encoder_h': self.encoder_h.state_dict(),
            'decoder_l': self.decoder_l.state_dict(),
            'decoder_h': self.decoder_h.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'alpha_l': self.alpha_l,
            'alpha_h': self.alpha_h,
            'metrics': metrics,
            'config': self.config
        }, path)
    
    def load(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.encoder_l.load_state_dict(ckpt['encoder_l'])
        self.encoder_h.load_state_dict(ckpt['encoder_h'])
        self.decoder_l.load_state_dict(ckpt['decoder_l'])
        self.decoder_h.load_state_dict(ckpt['decoder_h'])
        self.alpha_l = ckpt.get('alpha_l', self.alpha_l)
        self.alpha_h = ckpt.get('alpha_h', self.alpha_h)
        return ckpt.get('epoch', 0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/default.yaml')
    parser.add_argument('--cache', default='cache/roundtrip_20000.pt')
    parser.add_argument('--resume', required=True)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--samples', type=int, default=10000)
    args = parser.parse_args()
    
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load cache
    print(f"Loading cache: {args.cache}")
    cache = torch.load(args.cache, map_location='cpu', weights_only=False)
    z_orig = cache['z_orig'][:args.samples]
    z_roundtrip = cache['z_roundtrip'][:args.samples]
    watermarks = cache['watermarks'][:args.samples]
    print(f"Samples: {len(z_orig)}")
    
    # Create trainer
    trainer = LightweightTrainer(config, device)
    start_epoch = trainer.load(args.resume)
    print(f"Loaded: {args.resume}")
    
    # Output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"results/lightweight_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*50}")
    print("LIGHTWEIGHT ATTACK-AWARE TRAINING")
    print(f"{'='*50}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Augmentations: {[name for name, _ in AUGMENTATIONS]}")
    print(f"Output: {output_dir}")
    print(f"{'='*50}\n")
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(trainer.optimizer, T_max=args.epochs)
    
    best_acc = 0
    
    for epoch in range(1, args.epochs + 1):
        metrics = trainer.train_epoch(z_orig, z_roundtrip, watermarks, epoch, args.batch_size)
        scheduler.step()
        
        print(f"Epoch {epoch}: Loss={metrics['loss']:.4f}, Clean={metrics['acc_clean']:.3f}, Aug={metrics['acc_aug']:.3f}")
        
        if metrics['acc_aug'] > best_acc:
            best_acc = metrics['acc_aug']
            trainer.save(f"{output_dir}/best.pt", epoch, metrics)
            print(f"  -> Best: {best_acc:.4f}")
        
        # Checkpoint every 10 epochs
        if epoch % 10 == 0:
            trainer.save(f"{output_dir}/epoch_{epoch}.pt", epoch, metrics)
            print(f"  -> Checkpoint saved")
    
    trainer.save(f"{output_dir}/final.pt", args.epochs, metrics)
    print(f"\nDone! Best aug accuracy: {best_acc:.4f}")
    print(f"Results: {output_dir}")


if __name__ == "__main__":
    main()

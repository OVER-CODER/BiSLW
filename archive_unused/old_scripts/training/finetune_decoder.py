#!/usr/bin/env python3
"""
Fast decoder fine-tuning with simulated latent-space noise.
No VAE needed - very fast (~2 min/epoch).
Improves robustness by training decoder to handle noisy latents.
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
    """Add Gaussian noise to latent."""
    return z + torch.randn_like(z) * sigma


def latent_blur(z, kernel_size=3):
    """Blur latent spatially."""
    if kernel_size % 2 == 0:
        kernel_size += 1
    sigma = kernel_size / 3
    x = torch.arange(kernel_size, device=z.device, dtype=z.dtype) - kernel_size // 2
    kernel_1d = torch.exp(-x**2 / (2 * sigma**2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = kernel_1d.unsqueeze(1) * kernel_1d.unsqueeze(0)
    
    B, C, H, W = z.shape
    z_out = []
    for c in range(C):
        z_c = z[:, c:c+1, :, :]
        kernel = kernel_2d.unsqueeze(0).unsqueeze(0)
        pad = kernel_size // 2
        z_c_blurred = F.conv2d(z_c, kernel, padding=pad)
        z_out.append(z_c_blurred)
    return torch.cat(z_out, dim=1)


def latent_dropout(z, p=0.1):
    """Random dropout on latent."""
    mask = torch.rand_like(z) > p
    return z * mask


def latent_quantize(z, levels=16):
    """Quantize latent values."""
    z_min, z_max = z.min(), z.max()
    z_norm = (z - z_min) / (z_max - z_min + 1e-8)
    z_quant = torch.round(z_norm * levels) / levels
    return z_quant * (z_max - z_min) + z_min


LATENT_ATTACKS = [
    ('clean', lambda z: z, 0.3),
    ('noise_0.05', lambda z: latent_noise(z, 0.05), 0.15),
    ('noise_0.1', lambda z: latent_noise(z, 0.1), 0.15),
    ('noise_0.15', lambda z: latent_noise(z, 0.15), 0.1),
    ('blur_3', lambda z: latent_blur(z, 3), 0.15),
    ('dropout_0.1', lambda z: latent_dropout(z, 0.1), 0.1),
    ('quantize', lambda z: latent_quantize(z, 16), 0.05),
]


class DecoderFineTuner:
    """Fine-tune decoder for robustness using latent-space augmentations."""
    
    def __init__(self, config, device):
        self.config = config
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
        
        # Only train decoders (and maybe encoders with low lr)
        self.decoder_params = list(self.decoder_l.parameters()) + list(self.decoder_h.parameters())
        self.encoder_params = list(self.encoder_l.parameters()) + list(self.encoder_h.parameters())
        
        self.optimizer = torch.optim.AdamW([
            {'params': self.decoder_params, 'lr': config.get('lr', 1e-4) * 0.5},
            {'params': self.encoder_params, 'lr': config.get('lr', 1e-4) * 0.1},  # Lower LR for encoders
        ])
    
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
    
    def train_step(self, z_orig, w_batch, z_roundtrip):
        """Train with latent-space augmentations."""
        
        # Apply random latent attack to roundtrip latents
        probs = [p for _, _, p in LATENT_ATTACKS]
        probs = [p / sum(probs) for p in probs]
        
        z_augmented = z_roundtrip.clone()
        for i in range(z_roundtrip.shape[0]):
            attack_idx = np.random.choice(len(LATENT_ATTACKS), p=probs)
            _, attack_fn, _ = LATENT_ATTACKS[attack_idx]
            z_augmented[i:i+1] = attack_fn(z_roundtrip[i:i+1])
        
        # Extract from augmented
        w_pred_l, w_pred_h = self.extract_watermark(z_augmented)
        loss_aug = F.mse_loss(w_pred_l, w_batch) + F.mse_loss(w_pred_h, w_batch)
        
        # Also extract from clean embedded (consistency)
        z_wm = self.embed_watermark(z_orig, w_batch)
        w_pred_clean_l, w_pred_clean_h = self.extract_watermark(z_wm)
        loss_clean = F.mse_loss(w_pred_clean_l, w_batch) + F.mse_loss(w_pred_clean_h, w_batch)
        
        # Latent fidelity
        loss_latent = F.mse_loss(z_wm, z_orig)
        
        # Consistency
        loss_cons = F.mse_loss(w_pred_l, w_pred_h)
        
        total_loss = (
            2.0 * loss_aug +
            0.5 * loss_clean +
            0.3 * loss_cons +
            0.5 * loss_latent
        )
        
        acc_aug = self.compute_bit_accuracy(w_batch, w_pred_l, w_pred_h)
        acc_clean = self.compute_bit_accuracy(w_batch, w_pred_clean_l, w_pred_clean_h)
        
        return total_loss, {'loss': total_loss.item(), 'acc_aug': acc_aug, 'acc_clean': acc_clean}
    
    def train_epoch(self, z_orig, watermarks, z_roundtrip, epoch, batch_size=64):
        """Train one epoch - very fast with large batches."""
        self._set_train_mode()
        
        n = len(z_orig)
        indices = torch.randperm(n)
        
        metrics = {'loss': [], 'acc_aug': [], 'acc_clean': []}
        
        pbar = tqdm(range(0, n - batch_size, batch_size), desc=f"Epoch {epoch}")
        for start in pbar:
            idx = indices[start:start + batch_size]
            
            z_batch = z_orig[idx].to(self.device)
            w_batch = watermarks[idx].to(self.device)
            z_rt_batch = z_roundtrip[idx].to(self.device)
            
            self.optimizer.zero_grad()
            loss, m = self.train_step(z_batch, w_batch, z_rt_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.decoder_params + self.encoder_params, 1.0)
            self.optimizer.step()
            
            for k, v in m.items():
                metrics[k].append(v)
            
            pbar.set_postfix(loss=f"{m['loss']:.3f}", aug=f"{m['acc_aug']:.3f}")
        
        return {k: np.mean(v) for k, v in metrics.items()}
    
    def _set_train_mode(self):
        self.encoder_l.train()
        self.encoder_h.train()
        self.decoder_l.train()
        self.decoder_h.train()
    
    def save_checkpoint(self, path, epoch, metrics):
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
        print(f"Saved: {path}")
    
    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.encoder_l.load_state_dict(ckpt['encoder_l'])
        self.encoder_h.load_state_dict(ckpt['encoder_h'])
        self.decoder_l.load_state_dict(ckpt['decoder_l'])
        self.decoder_h.load_state_dict(ckpt['decoder_h'])
        self.alpha_l = ckpt.get('alpha_l', self.alpha_l)
        self.alpha_h = ckpt.get('alpha_h', self.alpha_h)
        print(f"Loaded: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--cache', type=str, default='cache/roundtrip_20000.pt')
    parser.add_argument('--resume', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=64)
    args = parser.parse_args()
    
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load cache
    print(f"Loading cache: {args.cache}")
    cache = torch.load(args.cache, map_location='cpu', weights_only=False)
    z_orig = cache['z_orig']
    watermarks = cache['watermarks']
    z_roundtrip = cache['z_roundtrip']
    print(f"Loaded {len(z_orig)} samples")
    
    # Create trainer
    trainer = DecoderFineTuner(config, device)
    trainer.load_checkpoint(args.resume)
    
    # Output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"results/decoder_ft_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("FAST DECODER FINE-TUNING")
    print("="*60)
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Latent attacks: {[name for name, _, _ in LATENT_ATTACKS]}")
    print(f"Output: {output_dir}")
    print("="*60 + "\n")
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(trainer.optimizer, T_max=args.epochs)
    
    best_acc = 0.0
    
    for epoch in range(args.epochs):
        metrics = trainer.train_epoch(z_orig, watermarks, z_roundtrip, epoch + 1, args.batch_size)
        scheduler.step()
        
        print(f"Epoch {epoch+1}: Loss={metrics['loss']:.4f}, "
              f"Aug={metrics['acc_aug']:.4f}, Clean={metrics['acc_clean']:.4f}")
        
        if metrics['acc_aug'] > best_acc:
            best_acc = metrics['acc_aug']
            trainer.save_checkpoint(f"{output_dir}/best.pt", epoch+1, metrics)
            print(f"  New best: {best_acc:.4f}")
    
    trainer.save_checkpoint(f"{output_dir}/final.pt", args.epochs, metrics)
    print(f"\nDone! Best augmented accuracy: {best_acc:.4f}")
    print(f"Results: {output_dir}")


if __name__ == "__main__":
    main()

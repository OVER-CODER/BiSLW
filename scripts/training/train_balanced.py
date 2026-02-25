#!/usr/bin/env python3
"""
Balanced Training: Optimizes for BOTH quality (PSNR/SSIM) AND robustness.

Key differences from existing training:
1. Higher weight on latent quality loss (lower distortion)
2. Adaptive alpha - starts low, increases during training
3. Quality-gated robustness training
4. Uses precomputed attacked latents for speed

Target: PSNR >= 36 dB AND robustness >= 70%
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from datetime import datetime

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.dirname(project_root))

from latent_watermarking.models.latent_split import LatentSplitter
from latent_watermarking.models.recombination import LatentRecombiner
from latent_watermarking.models.watermark_encoder import WatermarkEncoder
from latent_watermarking.models.watermark_decoder import WatermarkDecoder


# Attack names matching precomputed cache
ATTACK_NAMES = ['clean', 'center_crop', 'random_crop', 'resize', 'rotation', 
                'blur', 'contrast', 'brightness', 'jpeg', 'combined']


class BalancedTrainer:
    """
    Balanced trainer prioritizing both quality and robustness.
    """
    
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
        
        # ADAPTIVE ALPHA - key for quality-robustness balance
        # Start lower than lightweight model, increase during training
        self.base_alpha_l = config.get('alpha_l', 0.015)  # Lower than 0.02
        self.base_alpha_h = config.get('alpha_h', 0.0075)  # Lower than 0.01
        self.alpha_warmup_epochs = config.get('alpha_warmup', 20)
        
        self.all_params = (
            list(self.encoder_l.parameters()) +
            list(self.encoder_h.parameters()) +
            list(self.decoder_l.parameters()) +
            list(self.decoder_h.parameters())
        )
        
        self.optimizer = torch.optim.AdamW(
            self.all_params, 
            lr=config.get('lr', 8e-5),
            weight_decay=1e-5
        )
        
        # Loss weights - HIGHER quality weight than attack_fast
        self.w_attacked = config.get('w_attacked', 2.0)  # Lower than 3.0 in lightweight
        self.w_clean = config.get('w_clean', 1.0)  # Higher than 0.5
        self.w_latent = config.get('w_latent', 2.5)  # MUCH higher than 0.5 - key for quality
        self.w_cons = config.get('w_cons', 0.3)
        
    def get_alpha(self, epoch):
        """Warmup alpha from 50% to 100% over warmup epochs."""
        if epoch >= self.alpha_warmup_epochs:
            return self.base_alpha_l, self.base_alpha_h
        
        progress = epoch / self.alpha_warmup_epochs
        # Start at 50%, ramp to 100%
        scale = 0.5 + 0.5 * progress
        return self.base_alpha_l * scale, self.base_alpha_h * scale
        
    def embed_watermark(self, z, w, alpha_l, alpha_h):
        z_low, z_high = self.splitter(z)
        z_low_wm = self.encoder_l(z_low, w, alpha=alpha_l)
        z_high_wm = self.encoder_h(z_high, w, alpha=alpha_h)
        return self.recombiner(z_low_wm, z_high_wm)
    
    def extract_watermark(self, z):
        z_low, z_high = self.splitter(z)
        return self.decoder_l(z_low), self.decoder_h(z_high)
    
    def compute_bit_accuracy(self, w_true, w_pred_l, w_pred_h):
        bits_true = (w_true > 0).float()
        bits_pred = ((w_pred_l + w_pred_h) / 2 > 0).float()
        return (bits_true == bits_pred).float().mean().item()
    
    def train_step(self, z_orig, w_batch, attacked_latents_dict, epoch):
        """Training step with balanced loss."""
        B = z_orig.shape[0]
        alpha_l, alpha_h = self.get_alpha(epoch)
        
        # Embed watermark with current alpha
        z_wm = self.embed_watermark(z_orig, w_batch, alpha_l, alpha_h)
        
        # === QUALITY LOSSES ===
        # 1. Latent distortion loss - KEY for PSNR
        loss_latent = F.mse_loss(z_wm, z_orig)
        
        # 2. Frequency preservation loss - preserve high-freq components
        z_l_orig, z_h_orig = self.splitter(z_orig)
        z_l_wm, z_h_wm = self.splitter(z_wm)
        loss_freq = F.mse_loss(z_h_wm, z_h_orig)  # High-freq more important for quality
        
        # === ROBUSTNESS LOSSES ===
        # 3. Clean extraction
        w_pred_l, w_pred_h = self.extract_watermark(z_wm)
        loss_clean = F.mse_loss(w_pred_l, w_batch) + F.mse_loss(w_pred_h, w_batch)
        
        # 4. Attacked extraction - sample random attacks
        attack_indices = np.random.choice(len(ATTACK_NAMES), B, replace=True)
        z_attacked = torch.zeros_like(z_orig)
        for i, idx in enumerate(attack_indices):
            attack_name = ATTACK_NAMES[idx]
            z_attacked[i] = attacked_latents_dict[attack_name][i]
        
        w_att_l, w_att_h = self.extract_watermark(z_attacked)
        loss_attacked = F.mse_loss(w_att_l, w_batch) + F.mse_loss(w_att_h, w_batch)
        
        # 5. Consistency loss
        loss_cons = F.mse_loss(w_pred_l, w_pred_h)
        
        # === COMBINED LOSS ===
        total_loss = (
            self.w_attacked * loss_attacked +
            self.w_clean * loss_clean +
            self.w_latent * loss_latent +
            0.5 * loss_freq +  # Extra frequency preservation
            self.w_cons * loss_cons
        )
        
        # Metrics
        bit_acc_clean = self.compute_bit_accuracy(w_batch, w_pred_l, w_pred_h)
        bit_acc_attacked = self.compute_bit_accuracy(w_batch, w_att_l, w_att_h)
        
        return total_loss, {
            'loss': total_loss.item(),
            'loss_latent': loss_latent.item(),
            'bit_acc_clean': bit_acc_clean,
            'bit_acc_attacked': bit_acc_attacked,
            'alpha_l': alpha_l,
        }
    
    def train_epoch(self, cache_data, epoch, batch_size=32):
        """Train one epoch."""
        self._set_train_mode()
        
        z_orig = cache_data['z_orig']
        watermarks = cache_data['watermarks']
        n = len(z_orig)
        
        # Get attacked latents
        attacked_dict = {name: cache_data[f'z_{name}'] for name in ATTACK_NAMES 
                        if f'z_{name}' in cache_data}
        
        # Fallback if some attacks missing
        for name in ATTACK_NAMES:
            if name not in attacked_dict:
                attacked_dict[name] = cache_data.get('z_clean', z_orig)
        
        indices = torch.randperm(n)
        epoch_metrics = {'loss': [], 'loss_latent': [], 'bit_acc_clean': [], 
                        'bit_acc_attacked': [], 'alpha_l': []}
        
        pbar = tqdm(range(0, n, batch_size), desc=f"Epoch {epoch}")
        for start in pbar:
            end = min(start + batch_size, n)
            idx = indices[start:end]
            
            z_batch = z_orig[idx].to(self.device)
            w_batch = watermarks[idx].to(self.device)
            
            batch_attacked = {name: attacked_dict[name][idx].to(self.device) 
                             for name in ATTACK_NAMES}
            
            self.optimizer.zero_grad()
            loss, metrics = self.train_step(z_batch, w_batch, batch_attacked, epoch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.all_params, 1.0)
            self.optimizer.step()
            
            for k, v in metrics.items():
                epoch_metrics[k].append(v)
            
            pbar.set_postfix(
                loss=f"{metrics['loss']:.3f}",
                lat=f"{metrics['loss_latent']:.4f}",
                clean=f"{metrics['bit_acc_clean']:.3f}",
                att=f"{metrics['bit_acc_attacked']:.3f}"
            )
            
            if self.device.type == 'mps':
                torch.mps.empty_cache()
        
        return {k: np.mean(v) for k, v in epoch_metrics.items()}
    
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
    
    def save_checkpoint(self, path, epoch, metrics):
        torch.save({
            'epoch': epoch,
            'encoder_l': self.encoder_l.state_dict(),
            'encoder_h': self.encoder_h.state_dict(),
            'decoder_l': self.decoder_l.state_dict(),
            'decoder_h': self.decoder_h.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self.config,
            'metrics': metrics,
            'alpha_l': self.base_alpha_l,
            'alpha_h': self.base_alpha_h,
        }, path)
        print(f"  Saved: {path}")


def load_cache(cache_path):
    """Load precomputed attack cache."""
    print(f"Loading cache: {cache_path}")
    data = torch.load(cache_path, map_location='cpu', weights_only=False)
    
    print(f"  Keys available: {list(data.keys())}")
    print(f"  Samples: {len(data.get('z_orig', data.get('z_clean', [])))}")
    
    return data


def main():
    parser = argparse.ArgumentParser(description='Balanced watermark training')
    parser.add_argument('--cache', type=str, default='cache/attacked_5000.pt',
                       help='Path to precomputed attack cache')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=8e-5)
    parser.add_argument('--alpha_l', type=float, default=0.015,
                       help='Base alpha for low freq (default: 0.015, lower than lightweight)')
    parser.add_argument('--alpha_h', type=float, default=0.0075,
                       help='Base alpha for high freq (default: 0.0075)')
    parser.add_argument('--w_latent', type=float, default=2.5,
                       help='Weight for latent quality loss (higher = better PSNR)')
    parser.add_argument('--w_attacked', type=float, default=2.0,
                       help='Weight for attack robustness loss')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 
                         'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Setup output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(project_root, f'results/balanced_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output: {output_dir}")
    
    # Load cache
    cache_path = os.path.join(project_root, args.cache)
    cache_data = load_cache(cache_path)
    
    # Config
    config = {
        'w_dim': 32,
        'alpha_l': args.alpha_l,
        'alpha_h': args.alpha_h,
        'lr': args.lr,
        'w_latent': args.w_latent,
        'w_attacked': args.w_attacked,
        'alpha_warmup': 15,
    }
    
    print(f"\nConfig:")
    print(f"  alpha_l: {args.alpha_l} (lightweight uses 0.02)")
    print(f"  alpha_h: {args.alpha_h} (lightweight uses 0.01)")
    print(f"  w_latent: {args.w_latent} (quality weight, lightweight uses 0.5)")
    print(f"  w_attacked: {args.w_attacked} (robustness weight, lightweight uses 3.0)")
    
    # Initialize trainer
    trainer = BalancedTrainer(config, device)
    
    best_score = 0
    best_epoch = 0
    
    print(f"\nStarting balanced training for {args.epochs} epochs...")
    print("=" * 60)
    
    for epoch in range(1, args.epochs + 1):
        metrics = trainer.train_epoch(cache_data, epoch, args.batch_size)
        
        # Score = average of clean and attacked accuracy
        score = (metrics['bit_acc_clean'] + metrics['bit_acc_attacked']) / 2
        
        print(f"\nEpoch {epoch}: loss={metrics['loss']:.3f}, "
              f"clean={metrics['bit_acc_clean']:.3f}, "
              f"attacked={metrics['bit_acc_attacked']:.3f}, "
              f"latent_mse={metrics['loss_latent']:.5f}, "
              f"alpha_l={metrics['alpha_l']:.4f}")
        
        # Save best
        if score > best_score:
            best_score = score
            best_epoch = epoch
            trainer.save_checkpoint(
                os.path.join(output_dir, 'best.pt'),
                epoch, metrics
            )
        
        # Save periodic checkpoints
        if epoch % 10 == 0:
            trainer.save_checkpoint(
                os.path.join(output_dir, f'epoch_{epoch}.pt'),
                epoch, metrics
            )
    
    # Save final
    trainer.save_checkpoint(
        os.path.join(output_dir, 'final.pt'),
        args.epochs, metrics
    )
    
    print(f"\n{'=' * 60}")
    print(f"Training complete!")
    print(f"Best epoch: {best_epoch} (score: {best_score:.3f})")
    print(f"Output: {output_dir}")
    print(f"\nTo evaluate: python scripts/evaluation/compare_models_attacks.py")


if __name__ == '__main__':
    main()

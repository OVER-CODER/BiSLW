#!/usr/bin/env python3
"""
Focused Attack-Aware Training with higher alpha.
- 4x weight on weak attacks (crop, rotation)
- Higher alpha for stronger watermark
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

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.dirname(project_root))

from latent_watermarking.models.latent_split import LatentSplitter
from latent_watermarking.models.recombination import LatentRecombiner
from latent_watermarking.models.watermark_encoder import WatermarkEncoder
from latent_watermarking.models.watermark_decoder import WatermarkDecoder


# Attack names with WEIGHTS - higher weight = sampled more often
ATTACK_CONFIG = {
    'clean': 1.0,
    'center_crop': 4.0,   # 4x weight - weak attack
    'random_crop': 4.0,   # 4x weight - weak attack
    'resize': 1.0,
    'rotation': 4.0,      # 4x weight - weak attack
    'blur': 1.0,
    'contrast': 1.0,
    'brightness': 2.0,    # 2x weight - moderate
    'jpeg': 1.0,
    'combined': 1.0,
}

ATTACK_NAMES = list(ATTACK_CONFIG.keys())
ATTACK_WEIGHTS = np.array([ATTACK_CONFIG[name] for name in ATTACK_NAMES])
ATTACK_PROBS = ATTACK_WEIGHTS / ATTACK_WEIGHTS.sum()


class FocusedTrainer:
    """Trainer with focused sampling and higher alpha."""
    
    def __init__(self, config, device, alpha_multiplier=2.0):
        self.config = config
        self.device = device
        
        w_dim = config.get('w_dim', 32)
        
        self.splitter = LatentSplitter(mode='dct').to(device)
        self.recombiner = LatentRecombiner(mode='dct').to(device)
        
        self.encoder_l = WatermarkEncoder(watermark_dim=w_dim).to(device)
        self.encoder_h = WatermarkEncoder(watermark_dim=w_dim).to(device)
        self.decoder_l = WatermarkDecoder(watermark_dim=w_dim).to(device)
        self.decoder_h = WatermarkDecoder(watermark_dim=w_dim).to(device)
        
        # Higher alpha for stronger watermark
        base_alpha_l = config.get('alpha_l', 0.02)
        base_alpha_h = config.get('alpha_h', 0.01)
        self.alpha_l = base_alpha_l * alpha_multiplier
        self.alpha_h = base_alpha_h * alpha_multiplier
        
        print(f"  Alpha L: {base_alpha_l} -> {self.alpha_l} ({alpha_multiplier}x)")
        print(f"  Alpha H: {base_alpha_h} -> {self.alpha_h} ({alpha_multiplier}x)")
        
        self.all_params = (
            list(self.encoder_l.parameters()) +
            list(self.encoder_h.parameters()) +
            list(self.decoder_l.parameters()) +
            list(self.decoder_h.parameters())
        )
        
        self.optimizer = torch.optim.AdamW(
            self.all_params, 
            lr=config.get('lr', 1e-4),
            weight_decay=1e-5
        )
        
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
    
    def train_step(self, z_orig, w_batch, attacked_latents_dict):
        """Training step with focused attack sampling."""
        B = z_orig.shape[0]
        
        # Sample attacks with weighted probabilities (focus on weak attacks)
        attack_indices = np.random.choice(len(ATTACK_NAMES), B, p=ATTACK_PROBS)
        
        z_attacked = torch.zeros_like(z_orig)
        for i, idx in enumerate(attack_indices):
            attack_name = ATTACK_NAMES[idx]
            z_attacked[i] = attacked_latents_dict[attack_name][i]
        
        w_pred_l, w_pred_h = self.extract_watermark(z_attacked)
        
        z_wm = self.embed_watermark(z_orig, w_batch)
        w_pred_clean_l, w_pred_clean_h = self.extract_watermark(z_wm)
        
        # Losses with higher weight on attacked
        loss_attacked = F.mse_loss(w_pred_l, w_batch) + F.mse_loss(w_pred_h, w_batch)
        loss_clean = F.mse_loss(w_pred_clean_l, w_batch) + F.mse_loss(w_pred_clean_h, w_batch)
        loss_cons = F.mse_loss(w_pred_l, w_pred_h)
        loss_latent = F.mse_loss(z_wm, z_orig)
        
        total_loss = (
            4.0 * loss_attacked +   # Higher weight on robustness
            0.3 * loss_clean +
            0.2 * loss_cons +
            0.3 * loss_latent       # Lower weight on quality
        )
        
        bit_acc_attacked = self.compute_bit_accuracy(w_batch, w_pred_l, w_pred_h)
        bit_acc_clean = self.compute_bit_accuracy(w_batch, w_pred_clean_l, w_pred_clean_h)
        
        return total_loss, {
            'loss': total_loss.item(),
            'bit_acc_attacked': bit_acc_attacked,
            'bit_acc_clean': bit_acc_clean
        }
    
    def train_epoch(self, cache_data, epoch, batch_size=64):
        self._set_train_mode()
        
        z_orig = cache_data['z_orig']
        watermarks = cache_data['watermarks']
        n = len(z_orig)
        
        attacked_dict = {name: cache_data[f'z_{name}'] for name in ATTACK_NAMES}
        
        indices = torch.randperm(n)
        epoch_metrics = {'loss': [], 'bit_acc_attacked': [], 'bit_acc_clean': []}
        
        pbar = tqdm(range(0, n, batch_size), desc=f"Epoch {epoch}")
        for start in pbar:
            end = min(start + batch_size, n)
            idx = indices[start:end]
            
            z_batch = z_orig[idx].to(self.device)
            w_batch = watermarks[idx].to(self.device)
            
            batch_attacked = {name: attacked_dict[name][idx].to(self.device) 
                             for name in ATTACK_NAMES}
            
            self.optimizer.zero_grad()
            loss, metrics = self.train_step(z_batch, w_batch, batch_attacked)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.all_params, 1.0)
            self.optimizer.step()
            
            for k, v in metrics.items():
                epoch_metrics[k].append(v)
            
            pbar.set_postfix(
                loss=f"{metrics['loss']:.3f}",
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
            'alpha_l': self.alpha_l,
            'alpha_h': self.alpha_h,
            'metrics': metrics,
            'config': self.config
        }, path)
        print(f"  Saved: {path}")
    
    def load_checkpoint(self, path, load_alpha=False):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.encoder_l.load_state_dict(checkpoint['encoder_l'])
        self.encoder_h.load_state_dict(checkpoint['encoder_h'])
        self.decoder_l.load_state_dict(checkpoint['decoder_l'])
        self.decoder_h.load_state_dict(checkpoint['decoder_h'])
        # Don't load optimizer - fresh start with new LR
        if load_alpha:
            self.alpha_l = checkpoint.get('alpha_l', self.alpha_l)
            self.alpha_h = checkpoint.get('alpha_h', self.alpha_h)
        print(f"  Loaded weights from: {path}")
        return checkpoint.get('epoch', 0)
    
    @torch.no_grad()
    def evaluate(self, cache_data, n_samples=200):
        self._set_eval_mode()
        
        z_orig = cache_data['z_orig']
        watermarks = cache_data['watermarks']
        
        n = min(n_samples, len(z_orig))
        indices = torch.randperm(len(z_orig))[:n]
        
        results = {name: [] for name in ATTACK_NAMES}
        
        for i in tqdm(indices, desc="Evaluating"):
            z = z_orig[i:i+1].to(self.device)
            w = watermarks[i:i+1].to(self.device)
            
            for name in ATTACK_NAMES:
                z_att = cache_data[f'z_{name}'][i:i+1].to(self.device)
                w_pred_l, w_pred_h = self.extract_watermark(z_att)
                acc = self.compute_bit_accuracy(w, w_pred_l, w_pred_h)
                results[name].append(acc)
        
        name_map = {
            'clean': 'None',
            'center_crop': 'C. Crop 0.1',
            'random_crop': 'R. Crop 0.1',
            'resize': 'Resize 0.7',
            'rotation': 'Rot. 15',
            'blur': 'Blur',
            'contrast': 'Contr. 2.0',
            'brightness': 'Bright. 2.0',
            'jpeg': 'JPEG 70',
            'combined': 'Comb.'
        }
        
        return {name_map.get(k, k): np.mean(v) for k, v in results.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--cache', type=str, required=True)
    parser.add_argument('--resume', type=str, help='Load weights from checkpoint')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--alpha-mult', type=float, default=2.0, help='Alpha multiplier')
    parser.add_argument('--eval-interval', type=int, default=10)
    parser.add_argument('--save-interval', type=int, default=10)
    args = parser.parse_args()
    
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")
    
    print(f"Loading cache: {args.cache}")
    cache_data = torch.load(args.cache, map_location='cpu', weights_only=False)
    n_samples = len(cache_data['z_orig'])
    print(f"Samples: {n_samples}")
    
    print(f"\nAttack sampling weights:")
    for name, weight in ATTACK_CONFIG.items():
        prob = ATTACK_PROBS[ATTACK_NAMES.index(name)]
        print(f"  {name:15s}: {weight:.1f}x ({prob*100:.1f}%)")
    
    trainer = FocusedTrainer(config, device, alpha_multiplier=args.alpha_mult)
    
    if args.resume:
        trainer.load_checkpoint(args.resume, load_alpha=False)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"results/focused_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("FOCUSED ATTACK-AWARE TRAINING")
    print("="*60)
    print(f"Epochs: {args.epochs}")
    print(f"Alpha multiplier: {args.alpha_mult}x")
    print(f"Output: {output_dir}")
    print("="*60 + "\n")
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        trainer.optimizer, T_max=args.epochs, eta_min=1e-6
    )
    
    best_avg_acc = 0.0
    
    print("Initial evaluation...")
    init_eval = trainer.evaluate(cache_data, n_samples=100)
    print("\nInitial:")
    for name, acc in init_eval.items():
        print(f"  {name:15s}: {acc:.4f}")
    print(f"\n  Average: {np.mean(list(init_eval.values())):.4f}\n")
    
    for epoch in range(1, args.epochs + 1):
        metrics = trainer.train_epoch(cache_data, epoch, batch_size=args.batch_size)
        scheduler.step()
        
        lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch}: Loss={metrics['loss']:.4f}, "
              f"Att={metrics['bit_acc_attacked']:.4f}, LR={lr:.2e}")
        
        if epoch % args.save_interval == 0:
            trainer.save_checkpoint(f"{output_dir}/epoch_{epoch}.pt", epoch, metrics)
        
        if epoch % args.eval_interval == 0:
            eval_results = trainer.evaluate(cache_data, n_samples=200)
            
            print("\n  Per-attack:")
            for name, acc in eval_results.items():
                marker = "**" if acc < 0.75 else ""
                print(f"    {name:15s}: {acc:.4f} {marker}")
            
            avg_acc = np.mean(list(eval_results.values()))
            print(f"\n  Average: {avg_acc:.4f}")
            
            if avg_acc > best_avg_acc:
                best_avg_acc = avg_acc
                trainer.save_checkpoint(f"{output_dir}/best.pt", epoch, {**metrics, 'eval': eval_results})
                print(f"  *** New best: {best_avg_acc:.4f} ***")
            print()
    
    trainer.save_checkpoint(f"{output_dir}/final.pt", args.epochs, metrics)
    
    print("\n" + "="*60)
    print("FINAL EVALUATION")
    print("="*60)
    
    final_eval = trainer.evaluate(cache_data, n_samples=500)
    print("\nFinal:")
    for name, acc in final_eval.items():
        print(f"  {name:15s}: {acc:.4f}")
    
    avg_final = np.mean(list(final_eval.values()))
    print(f"\n  Final Average: {avg_final:.4f}")
    print(f"  Best Average:  {best_avg_acc:.4f}")
    print(f"\nResults: {output_dir}")


if __name__ == "__main__":
    main()

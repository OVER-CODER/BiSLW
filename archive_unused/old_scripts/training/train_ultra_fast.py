#!/usr/bin/env python3
"""
Ultra-fast attack-aware training on precomputed data.
No VAE in training loop = ~30 seconds per epoch.
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


class UltraFastTrainer:
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
        
        # All parameters
        self.all_params = (
            list(self.encoder_l.parameters()) +
            list(self.encoder_h.parameters()) +
            list(self.decoder_l.parameters()) +
            list(self.decoder_h.parameters())
        )
        self.optimizer = torch.optim.AdamW(self.all_params, lr=config.get('lr', 1e-4) * 0.1)
    
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
    
    def train_step(self, z_orig, w_batch, z_attacked_dict):
        """Train on batch with multiple attack types."""
        # Fresh embedding
        z_wm = self.embed(z_orig, w_batch)
        
        # Clean extraction loss
        w_l, w_h = self.extract(z_wm)
        loss_clean = F.mse_loss(w_l, w_batch) + F.mse_loss(w_h, w_batch)
        acc_clean = self.bit_acc(w_batch, w_l, w_h)
        
        # Attacked extraction loss
        loss_attacked = 0
        acc_attacked = []
        
        for name, z_att in z_attacked_dict.items():
            w_l, w_h = self.extract(z_att)
            loss_attacked += F.mse_loss(w_l, w_batch) + F.mse_loss(w_h, w_batch)
            acc_attacked.append(self.bit_acc(w_batch, w_l, w_h))
        
        loss_attacked /= len(z_attacked_dict)
        
        # Latent fidelity
        loss_fidelity = F.mse_loss(z_wm, z_orig)
        
        # Consistency
        w_l, w_h = self.extract(z_wm)
        loss_cons = F.mse_loss(w_l, w_h)
        
        # Total loss
        total_loss = 2.0 * loss_attacked + 0.5 * loss_clean + 0.5 * loss_fidelity + 0.2 * loss_cons
        
        return total_loss, {
            'loss': total_loss.item(),
            'acc_clean': acc_clean,
            'acc_attacked': np.mean(acc_attacked),
        }
    
    def train_epoch(self, cache, batch_size=64):
        """Train one epoch - very fast without VAE."""
        self._train_mode()
        
        z_orig = cache['z_orig']
        watermarks = cache['watermarks']
        attack_keys = [k for k in cache.keys() if k.startswith('z_') and k != 'z_orig']
        
        n = len(z_orig)
        indices = torch.randperm(n)
        
        metrics = {'loss': [], 'acc_clean': [], 'acc_attacked': []}
        
        for start in range(0, n - batch_size, batch_size):
            idx = indices[start:start + batch_size]
            
            z_batch = z_orig[idx].to(self.device)
            w_batch = watermarks[idx].to(self.device)
            
            # Get all attacked versions for this batch
            z_attacked = {k: cache[k][idx].to(self.device) for k in attack_keys}
            
            self.optimizer.zero_grad()
            loss, m = self.train_step(z_batch, w_batch, z_attacked)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.all_params, 1.0)
            self.optimizer.step()
            
            for k, v in m.items():
                metrics[k].append(v)
        
        return {k: np.mean(v) for k, v in metrics.items()}
    
    def _train_mode(self):
        for m in [self.encoder_l, self.encoder_h, self.decoder_l, self.decoder_h]:
            m.train()
    
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
            'config': {'w_dim': 32, 'alpha_l': self.alpha_l, 'alpha_h': self.alpha_h}
        }, path)
    
    def load(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.encoder_l.load_state_dict(ckpt['encoder_l'])
        self.encoder_h.load_state_dict(ckpt['encoder_h'])
        self.decoder_l.load_state_dict(ckpt['decoder_l'])
        self.decoder_h.load_state_dict(ckpt['decoder_h'])
        self.alpha_l = ckpt.get('alpha_l', 0.02)
        self.alpha_h = ckpt.get('alpha_h', 0.01)
        return ckpt.get('epoch', 0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', required=True, help='Precomputed attacked cache')
    parser.add_argument('--resume', required=True, help='Checkpoint to resume from')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--save-interval', type=int, default=10)
    args = parser.parse_args()
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load precomputed cache
    print(f"Loading cache: {args.cache}")
    cache = torch.load(args.cache, map_location='cpu', weights_only=False)
    
    attack_keys = [k for k in cache.keys() if k.startswith('z_') and k != 'z_orig']
    print(f"Samples: {len(cache['z_orig'])}")
    print(f"Attacks: {attack_keys}")
    
    # Create trainer
    config = {'w_dim': 32, 'alpha_l': 0.02, 'alpha_h': 0.01, 'lr': 1e-4}
    trainer = UltraFastTrainer(config, device)
    start_epoch = trainer.load(args.resume)
    print(f"Loaded checkpoint: {args.resume}")
    
    # Output dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"results/attack_ultra_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*50)
    print("ULTRA-FAST ATTACK-AWARE TRAINING")
    print("="*50)
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Output: {output_dir}")
    print("="*50 + "\n")
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(trainer.optimizer, T_max=args.epochs)
    
    best_acc = 0.0
    
    for epoch in range(1, args.epochs + 1):
        metrics = trainer.train_epoch(cache, batch_size=args.batch_size)
        scheduler.step()
        
        print(f"Epoch {epoch:3d}: Loss={metrics['loss']:.4f}, "
              f"Clean={metrics['acc_clean']:.4f}, Attack={metrics['acc_attacked']:.4f}")
        
        # Save best
        if metrics['acc_attacked'] > best_acc:
            best_acc = metrics['acc_attacked']
            trainer.save(f"{output_dir}/best.pt", epoch, metrics)
            print(f"         -> New best: {best_acc:.4f}")
        
        # Save checkpoint
        if epoch % args.save_interval == 0:
            trainer.save(f"{output_dir}/epoch_{epoch}.pt", epoch, metrics)
            print(f"         -> Checkpoint saved")
    
    # Final save
    trainer.save(f"{output_dir}/final.pt", args.epochs, metrics)
    
    print("\n" + "="*50)
    print("TRAINING COMPLETE")
    print("="*50)
    print(f"Best attack accuracy: {best_acc:.4f}")
    print(f"Results: {output_dir}")


if __name__ == "__main__":
    main()

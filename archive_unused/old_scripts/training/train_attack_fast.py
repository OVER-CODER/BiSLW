#!/usr/bin/env python3
"""
Fast attack-aware training using precomputed attacked latents.
No VAE in the training loop = very fast (~5 min/epoch).
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from latent_watermarking.models.latent_split import LatentSplitter
from latent_watermarking.models.recombination import LatentRecombiner
from latent_watermarking.models.watermark_encoder import WatermarkEncoder
from latent_watermarking.models.watermark_decoder import WatermarkDecoder


class FastAttackTrainer:
    """Fast trainer using precomputed attacked latents."""
    
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
        self.optimizer = torch.optim.AdamW(self.all_params, lr=config.get('lr', 1e-4) * 0.2)
        
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
    
    def train_step(self, z_orig, w_batch, z_attacked_list):
        """
        Train on mixed attacked latents.
        z_attacked_list: list of (name, z_attacked) tuples
        """
        # Fresh embedding using current encoder weights
        z_wm = self.embed_watermark(z_orig, w_batch)
        
        # Extract from clean embedding (latent consistency)
        w_pred_clean_l, w_pred_clean_h = self.extract_watermark(z_wm)
        loss_clean = F.mse_loss(w_pred_clean_l, w_batch) + F.mse_loss(w_pred_clean_h, w_batch)
        
        # Extract from each attacked version
        loss_attacked = 0
        acc_attacked = []
        
        for name, z_att in z_attacked_list:
            # Re-embed with current weights before extraction
            # The attacked latents were computed with old weights, so we need to
            # account for the fact that the watermark is embedded differently now
            # Actually, for decoder training, we train decoder to extract from attacked latents
            # that had watermark embedded, so we should use the precomputed z_att directly
            
            w_pred_l, w_pred_h = self.extract_watermark(z_att)
            loss_attacked += F.mse_loss(w_pred_l, w_batch) + F.mse_loss(w_pred_h, w_batch)
            acc_attacked.append(self.compute_bit_accuracy(w_batch, w_pred_l, w_pred_h))
        
        loss_attacked /= len(z_attacked_list)
        
        # Latent fidelity
        loss_latent = F.mse_loss(z_wm, z_orig)
        
        # Consistency between decoders
        loss_cons = F.mse_loss(w_pred_clean_l, w_pred_clean_h)
        
        total_loss = (
            2.0 * loss_attacked +
            0.5 * loss_clean +
            0.3 * loss_cons +
            0.5 * loss_latent
        )
        
        acc_clean = self.compute_bit_accuracy(w_batch, w_pred_clean_l, w_pred_clean_h)
        
        return total_loss, {
            'loss': total_loss.item(),
            'acc_clean': acc_clean,
            'acc_attacked': np.mean(acc_attacked),
        }
    
    def train_epoch(self, cache, epoch, batch_size=16):
        """Train one epoch using precomputed cache."""
        self._set_train_mode()
        
        z_orig = cache['z_orig']
        watermarks = cache['watermarks']
        attack_names = [k for k in cache.keys() if k.startswith('z_') and k != 'z_orig']
        
        n = len(z_orig)
        indices = torch.randperm(n)
        
        epoch_metrics = {'loss': [], 'acc_clean': [], 'acc_attacked': []}
        
        pbar = tqdm(range(0, n - batch_size, batch_size), desc=f"Epoch {epoch}")
        for start in pbar:
            idx = indices[start:start + batch_size]
            
            z_batch = z_orig[idx].to(self.device)
            w_batch = watermarks[idx].to(self.device)
            
            # Randomly select 2-3 attacks per batch for variety
            selected_attacks = np.random.choice(attack_names, size=min(3, len(attack_names)), replace=False)
            z_attacked_list = [(name, cache[name][idx].to(self.device)) for name in selected_attacks]
            
            self.optimizer.zero_grad()
            loss, metrics = self.train_step(z_batch, w_batch, z_attacked_list)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.all_params, 1.0)
            self.optimizer.step()
            
            for k, v in metrics.items():
                epoch_metrics[k].append(v)
            
            pbar.set_postfix(
                loss=f"{metrics['loss']:.3f}",
                att=f"{metrics['acc_attacked']:.3f}"
            )
            
            if self.device.type == 'mps':
                torch.mps.empty_cache()
        
        return {k: np.mean(v) for k, v in epoch_metrics.items()}
    
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
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.encoder_l.load_state_dict(checkpoint['encoder_l'])
        self.encoder_h.load_state_dict(checkpoint['encoder_h'])
        self.decoder_l.load_state_dict(checkpoint['decoder_l'])
        self.decoder_h.load_state_dict(checkpoint['decoder_h'])
        if 'optimizer' in checkpoint:
            try:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            except:
                pass  # Different optimizer config, skip
        self.alpha_l = checkpoint.get('alpha_l', self.alpha_l)
        self.alpha_h = checkpoint.get('alpha_h', self.alpha_h)
        print(f"Loaded: {path}")
        return checkpoint.get('epoch', 0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--cache', type=str, default='cache/attacked_10000.pt')
    parser.add_argument('--resume', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()
    
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load precomputed attacked cache
    print(f"Loading cache: {args.cache}")
    cache = torch.load(args.cache, map_location='cpu', weights_only=False)
    
    attack_names = [k for k in cache.keys() if k.startswith('z_') and k != 'z_orig']
    print(f"Samples: {len(cache['z_orig'])}")
    print(f"Attacks: {attack_names}")
    
    # Create trainer
    trainer = FastAttackTrainer(config, device)
    trainer.load_checkpoint(args.resume)
    
    # Output dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"results/attack_fast_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("FAST ATTACK-AWARE TRAINING")
    print("="*60)
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Output: {output_dir}")
    print("="*60 + "\n")
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(trainer.optimizer, T_max=args.epochs)
    
    best_acc = 0.0
    
    for epoch in range(args.epochs):
        metrics = trainer.train_epoch(cache, epoch + 1, batch_size=args.batch_size)
        scheduler.step()
        
        print(f"Epoch {epoch+1}: Loss={metrics['loss']:.4f}, "
              f"Clean={metrics['acc_clean']:.4f}, Attack={metrics['acc_attacked']:.4f}")
        
        if metrics['acc_attacked'] > best_acc:
            best_acc = metrics['acc_attacked']
            trainer.save_checkpoint(f"{output_dir}/best.pt", epoch+1, metrics)
            print(f"  New best: {best_acc:.4f}")
        
        if (epoch + 1) % 10 == 0:
            trainer.save_checkpoint(f"{output_dir}/epoch_{epoch+1}.pt", epoch+1, metrics)
    
    trainer.save_checkpoint(f"{output_dir}/final.pt", args.epochs, metrics)
    print(f"\nDone! Results: {output_dir}")
    print(f"Best attack accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    main()

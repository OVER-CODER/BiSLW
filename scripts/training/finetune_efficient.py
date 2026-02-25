#!/usr/bin/env python3
"""
Fine-tune the efficient model (best quality) with attack robustness.

Strategy: Start from efficient model weights, add attack training
but with HIGH quality preservation weight to maintain PSNR.
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


ATTACK_NAMES = ['clean', 'center_crop', 'random_crop', 'resize', 'rotation', 
                'blur', 'contrast', 'brightness', 'jpeg', 'combined']


class FineTuneTrainer:
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
        
        # IMPORTANT: Lower LR for fine-tuning to preserve quality
        self.all_params = (
            list(self.encoder_l.parameters()) +
            list(self.encoder_h.parameters()) +
            list(self.decoder_l.parameters()) +
            list(self.decoder_h.parameters())
        )
        
        self.optimizer = torch.optim.AdamW(
            self.all_params, 
            lr=config.get('lr', 2e-5),  # Very low LR for fine-tuning
            weight_decay=1e-5
        )
        
    def load_pretrained(self, checkpoint_path):
        """Load pretrained efficient model weights."""
        print(f"Loading pretrained: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        self.encoder_l.load_state_dict(ckpt['encoder_l'])
        self.encoder_h.load_state_dict(ckpt['encoder_h'])
        self.decoder_l.load_state_dict(ckpt['decoder_l'])
        self.decoder_h.load_state_dict(ckpt['decoder_h'])
        
        # Use original alpha values
        self.alpha_l = ckpt.get('alpha_l', self.alpha_l)
        self.alpha_h = ckpt.get('alpha_h', self.alpha_h)
        print(f"  Loaded alpha_l={self.alpha_l}, alpha_h={self.alpha_h}")
        
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
        """Fine-tuning step - high weight on quality preservation."""
        B = z_orig.shape[0]
        
        # Embed watermark
        z_wm = self.embed_watermark(z_orig, w_batch)
        
        # === QUALITY PRESERVATION (high weight) ===
        loss_latent = F.mse_loss(z_wm, z_orig)
        
        # === CLEAN ACCURACY ===
        w_pred_l, w_pred_h = self.extract_watermark(z_wm)
        loss_clean = F.mse_loss(w_pred_l, w_batch) + F.mse_loss(w_pred_h, w_batch)
        
        # === ATTACK ROBUSTNESS (moderate weight) ===
        attack_indices = np.random.choice(len(ATTACK_NAMES), B, replace=True)
        z_attacked = torch.zeros_like(z_orig)
        for i, idx in enumerate(attack_indices):
            attack_name = ATTACK_NAMES[idx]
            z_attacked[i] = attacked_latents_dict[attack_name][i]
        
        w_att_l, w_att_h = self.extract_watermark(z_attacked)
        loss_attacked = F.mse_loss(w_att_l, w_batch) + F.mse_loss(w_att_h, w_batch)
        
        # === LOSS BALANCE ===
        # Key: HIGH quality weight to maintain efficient model's PSNR
        total_loss = (
            1.5 * loss_attacked +   # Moderate robustness training
            1.0 * loss_clean +      # Maintain clean accuracy
            3.0 * loss_latent       # HIGH quality preservation
        )
        
        bit_acc_clean = self.compute_bit_accuracy(w_batch, w_pred_l, w_pred_h)
        bit_acc_attacked = self.compute_bit_accuracy(w_batch, w_att_l, w_att_h)
        
        return total_loss, {
            'loss': total_loss.item(),
            'loss_latent': loss_latent.item(),
            'bit_acc_clean': bit_acc_clean,
            'bit_acc_attacked': bit_acc_attacked,
        }
    
    def train_epoch(self, cache_data, epoch, batch_size=32):
        self._set_train_mode()
        
        z_orig = cache_data['z_orig']
        watermarks = cache_data['watermarks']
        n = len(z_orig)
        
        attacked_dict = {name: cache_data.get(f'z_{name}', z_orig) for name in ATTACK_NAMES}
        
        indices = torch.randperm(n)
        epoch_metrics = {'loss': [], 'loss_latent': [], 'bit_acc_clean': [], 'bit_acc_attacked': []}
        
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
            torch.nn.utils.clip_grad_norm_(self.all_params, 0.5)  # Lower clip for stability
            self.optimizer.step()
            
            for k, v in metrics.items():
                epoch_metrics[k].append(v)
            
            pbar.set_postfix(
                loss=f"{metrics['loss']:.3f}",
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
            'alpha_l': self.alpha_l,
            'alpha_h': self.alpha_h,
        }, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained', type=str, 
                       default='best res/efficient_20260222_004718/best_model.pth',
                       help='Path to pretrained efficient model')
    parser.add_argument('--cache', type=str, default='cache/attacked_5000.pt')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Fewer epochs for fine-tuning')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=2e-5,
                       help='Low LR to preserve quality')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 
                         'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(project_root, f'results/finetune_efficient_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output: {output_dir}")
    
    # Load cache
    cache_path = os.path.join(project_root, args.cache)
    print(f"Loading cache: {cache_path}")
    cache_data = torch.load(cache_path, map_location='cpu', weights_only=False)
    
    # Config
    config = {
        'w_dim': 32,
        'alpha_l': 0.02,
        'alpha_h': 0.01,
        'lr': args.lr,
    }
    
    # Initialize and load pretrained
    trainer = FineTuneTrainer(config, device)
    pretrained_path = os.path.join(project_root, args.pretrained)
    trainer.load_pretrained(pretrained_path)
    
    best_score = 0
    
    print(f"\nFine-tuning efficient model for {args.epochs} epochs...")
    print("Strategy: Low LR + High quality weight to maintain PSNR while adding robustness")
    print("=" * 60)
    
    for epoch in range(1, args.epochs + 1):
        metrics = trainer.train_epoch(cache_data, epoch, args.batch_size)
        
        score = (metrics['bit_acc_clean'] + metrics['bit_acc_attacked']) / 2
        
        print(f"Epoch {epoch}: loss={metrics['loss']:.3f}, "
              f"clean={metrics['bit_acc_clean']:.3f}, "
              f"attacked={metrics['bit_acc_attacked']:.3f}, "
              f"latent_mse={metrics['loss_latent']:.5f}")
        
        if score > best_score:
            best_score = score
            trainer.save_checkpoint(os.path.join(output_dir, 'best.pt'), epoch, metrics)
    
    trainer.save_checkpoint(os.path.join(output_dir, 'final.pt'), args.epochs, metrics)
    
    print(f"\nFine-tuning complete! Output: {output_dir}")
    print("Run: python scripts/evaluation/compare_models_attacks.py")


if __name__ == '__main__':
    main()

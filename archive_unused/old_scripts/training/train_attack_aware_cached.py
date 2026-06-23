#!/usr/bin/env python3
"""
Attack-Aware Training using existing roundtrip cache.
Applies attacks on-the-fly to avoid long precomputation.
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
# FAST ATTACKS (applied on-the-fly)
# ============================================================

def jpeg_attack_batch(images, quality=70):
    """JPEG compression - must process one at a time."""
    device = images.device
    results = []
    for i in range(images.shape[0]):
        img = images[i].cpu()
        img_np = ((img.permute(1, 2, 0) + 1) / 2 * 255).clamp(0, 255).numpy().astype(np.uint8)
        pil_img = Image.fromarray(img_np, mode='RGB')
        
        buffer = io.BytesIO()
        pil_img.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        compressed = Image.open(buffer).convert('RGB')
        
        img_out = torch.from_numpy(np.array(compressed)).float() / 255.0
        img_out = img_out.permute(2, 0, 1) * 2 - 1
        results.append(img_out)
    
    return torch.stack(results).to(device)


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


# Attack registry with probabilities
ATTACKS = [
    ('clean', None, 0.4),                    # 40% no attack
    ('jpeg_70', lambda x: jpeg_attack_batch(x, 70), 0.15),
    ('jpeg_50', lambda x: jpeg_attack_batch(x, 50), 0.1),
    ('noise_0.03', lambda x: gaussian_noise_attack(x, 0.03), 0.15),
    ('noise_0.05', lambda x: gaussian_noise_attack(x, 0.05), 0.1),
    ('blur_3', lambda x: gaussian_blur_attack(x, 3), 0.1),
]


class VAEWrapper:
    """Lightweight VAE wrapper."""
    
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
    
    def unload(self):
        if self._vae is not None:
            del self._vae
            self._vae = None
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
    
    @torch.no_grad()
    def decode(self, z):
        vae = self.load()
        return vae.decode(z / self.scaling_factor).sample
    
    @torch.no_grad()
    def encode(self, img):
        vae = self.load()
        return vae.encode(img).latent_dist.mean * self.scaling_factor


class AttackAwareTrainer:
    """Trainer using existing cache with on-the-fly attacks."""
    
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
        self.optimizer = torch.optim.AdamW(self.all_params, lr=config.get('lr', 1e-4) * 0.3)
        
        self.vae = None
        
    def _get_vae(self):
        if self.vae is None:
            self.vae = VAEWrapper(device=self.device)
        return self.vae
    
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
    
    def train_step(self, z_orig, w_batch, z_roundtrip, apply_attack_prob=0.5):
        """
        Training step with mixed clean/attacked samples.
        
        Uses cached z_roundtrip for clean samples.
        Applies on-the-fly attacks for attacked samples.
        """
        vae = self._get_vae()
        B = z_orig.shape[0]
        
        # Decide which samples get attacked
        attack_mask = torch.rand(B) < apply_attack_prob
        
        # For clean samples, use cached roundtrip
        z_for_extraction = z_roundtrip.clone()
        
        # For attacked samples, apply attack pipeline
        if attack_mask.any():
            attack_indices = attack_mask.nonzero().squeeze(-1)
            
            # Fresh embed for attacked samples
            z_wm = self.embed_watermark(z_orig[attack_indices], w_batch[attack_indices])
            
            # Decode to image
            img_wm = vae.decode(z_wm)
            
            # Select and apply random attack
            attack_probs = [p for _, _, p in ATTACKS]
            attack_probs = [p / sum(attack_probs) for p in attack_probs]
            
            for i, idx in enumerate(attack_indices):
                attack_idx = np.random.choice(len(ATTACKS), p=attack_probs)
                attack_name, attack_fn, _ = ATTACKS[attack_idx]
                
                if attack_fn is not None:
                    img_attacked = attack_fn(img_wm[i:i+1])
                else:
                    img_attacked = img_wm[i:i+1]
                
                z_attacked = vae.encode(img_attacked)
                z_for_extraction[idx] = z_attacked.squeeze(0)
        
        # Extract watermark
        w_pred_l, w_pred_h = self.extract_watermark(z_for_extraction)
        
        # Also extract from clean embedding for consistency
        z_wm_clean = self.embed_watermark(z_orig, w_batch)
        w_pred_clean_l, w_pred_clean_h = self.extract_watermark(z_wm_clean)
        
        # Losses
        loss_attacked = F.mse_loss(w_pred_l, w_batch) + F.mse_loss(w_pred_h, w_batch)
        loss_clean = F.mse_loss(w_pred_clean_l, w_batch) + F.mse_loss(w_pred_clean_h, w_batch)
        loss_cons = F.mse_loss(w_pred_l, w_pred_h)
        loss_latent = F.mse_loss(z_wm_clean, z_orig)
        
        total_loss = (
            2.0 * loss_attacked +
            0.5 * loss_clean +
            0.3 * loss_cons +
            1.0 * loss_latent
        )
        
        bit_acc_attacked = self.compute_bit_accuracy(w_batch, w_pred_l, w_pred_h)
        bit_acc_clean = self.compute_bit_accuracy(w_batch, w_pred_clean_l, w_pred_clean_h)
        
        return total_loss, {
            'loss': total_loss.item(),
            'bit_acc_attacked': bit_acc_attacked,
            'bit_acc_clean': bit_acc_clean
        }
    
    def train_epoch(self, z_orig, watermarks, z_roundtrip, epoch, batch_size=8, attack_prob=0.5):
        """Train one epoch with on-the-fly attacks."""
        self._set_train_mode()
        
        dataset = TensorDataset(z_orig, watermarks, z_roundtrip)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        
        epoch_metrics = {'loss': [], 'bit_acc_attacked': [], 'bit_acc_clean': []}
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for z_batch, w_batch, z_rt_batch in pbar:
            z_batch = z_batch.to(self.device)
            w_batch = w_batch.to(self.device)
            z_rt_batch = z_rt_batch.to(self.device)
            
            self.optimizer.zero_grad()
            loss, metrics = self.train_step(z_batch, w_batch, z_rt_batch, attack_prob)
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
        print(f"Checkpoint saved: {path}")
    
    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.encoder_l.load_state_dict(checkpoint['encoder_l'])
        self.encoder_h.load_state_dict(checkpoint['encoder_h'])
        self.decoder_l.load_state_dict(checkpoint['decoder_l'])
        self.decoder_h.load_state_dict(checkpoint['decoder_h'])
        if 'optimizer' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.alpha_l = checkpoint.get('alpha_l', self.alpha_l)
        self.alpha_h = checkpoint.get('alpha_h', self.alpha_h)
        print(f"Loaded checkpoint from {path}")
        return checkpoint.get('epoch', 0)
    
    @torch.no_grad()
    def evaluate(self, z_orig, watermarks, n_samples=100):
        """Evaluate under various attacks."""
        self._set_eval_mode()
        vae = self._get_vae()
        
        results = {name: [] for name, _, _ in ATTACKS}
        
        indices = torch.randperm(len(z_orig))[:n_samples]
        
        for i in tqdm(indices, desc="Evaluating"):
            z = z_orig[i:i+1].to(self.device)
            w = watermarks[i:i+1].to(self.device)
            
            z_wm = self.embed_watermark(z, w)
            img_wm = vae.decode(z_wm)
            
            for name, attack_fn, _ in ATTACKS:
                if attack_fn is not None:
                    img_att = attack_fn(img_wm)
                else:
                    img_att = img_wm
                
                z_att = vae.encode(img_att)
                w_pred_l, w_pred_h = self.extract_watermark(z_att)
                acc = self.compute_bit_accuracy(w, w_pred_l, w_pred_h)
                results[name].append(acc)
            
            if self.device.type == 'mps':
                torch.mps.empty_cache()
        
        return {k: np.mean(v) for k, v in results.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--cache', type=str, default='cache/roundtrip_20000.pt')
    parser.add_argument('--resume', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--attack-prob', type=float, default=0.5)
    parser.add_argument('--eval-interval', type=int, default=10)
    args = parser.parse_args()
    
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load existing cache
    print(f"Loading cache: {args.cache}")
    cache = torch.load(args.cache, map_location='cpu', weights_only=False)
    z_orig = cache['z_orig']
    watermarks = cache['watermarks']
    z_roundtrip = cache['z_roundtrip']
    print(f"Loaded {len(z_orig)} samples")
    
    # Create trainer and load checkpoint
    trainer = AttackAwareTrainer(config, device)
    trainer.load_checkpoint(args.resume)
    
    # Output dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"results/attack_aware_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("ATTACK-AWARE TRAINING (using existing cache)")
    print("="*60)
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Attack probability: {args.attack_prob}")
    print(f"Attacks: {[name for name, _, _ in ATTACKS]}")
    print(f"Output: {output_dir}")
    print("="*60 + "\n")
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(trainer.optimizer, T_max=args.epochs)
    
    best_att_acc = 0.0
    
    for epoch in range(args.epochs):
        metrics = trainer.train_epoch(
            z_orig, watermarks, z_roundtrip, 
            epoch + 1, 
            batch_size=args.batch_size,
            attack_prob=args.attack_prob
        )
        scheduler.step()
        
        print(f"Epoch {epoch+1}: Loss={metrics['loss']:.4f}, "
              f"Attack Acc={metrics['bit_acc_attacked']:.4f}, Clean Acc={metrics['bit_acc_clean']:.4f}")
        
        if metrics['bit_acc_attacked'] > best_att_acc:
            best_att_acc = metrics['bit_acc_attacked']
            trainer.save_checkpoint(f"{output_dir}/best.pt", epoch+1, metrics)
            print(f"  New best: {best_att_acc:.4f}")
        
        if (epoch + 1) % args.eval_interval == 0:
            eval_metrics = trainer.evaluate(z_orig, watermarks, n_samples=50)
            print("  Eval per attack:")
            for name, acc in eval_metrics.items():
                print(f"    {name}: {acc:.4f}")
    
    trainer.save_checkpoint(f"{output_dir}/final.pt", args.epochs, metrics)
    
    # Final eval
    print("\n" + "="*60)
    print("FINAL EVALUATION")
    print("="*60)
    
    final_metrics = trainer.evaluate(z_orig, watermarks, n_samples=100)
    for name, acc in final_metrics.items():
        print(f"  {name}: {acc:.4f}")
    
    avg = np.mean(list(final_metrics.values()))
    print(f"\nAverage: {avg:.4f}")
    print(f"Results: {output_dir}")


if __name__ == "__main__":
    main()

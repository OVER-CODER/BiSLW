#!/usr/bin/env python3
"""
Attack-Aware Training for Robust Latent Watermarking.

Precomputes attacked latents: embed → decode → [attacks] → encode
Then trains decoder to extract watermarks from attacked latents.
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
# ATTACKS (Image Space)
# ============================================================

def jpeg_attack(images, quality=70):
    """Apply JPEG compression to images."""
    device = images.device
    B, C, H, W = images.shape
    
    # Convert to PIL, compress, convert back
    results = []
    for i in range(B):
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
    """Add Gaussian noise."""
    noise = torch.randn_like(images) * sigma
    return (images + noise).clamp(-1, 1)


def gaussian_blur_attack(images, kernel_size=5):
    """Apply Gaussian blur."""
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


def resize_attack(images, scale=0.5):
    """Resize down then up."""
    B, C, H, W = images.shape
    small = F.interpolate(images, scale_factor=scale, mode='bilinear', align_corners=False)
    return F.interpolate(small, size=(H, W), mode='bilinear', align_corners=False)


def identity_attack(images):
    """No attack (clean roundtrip)."""
    return images


# Attack configurations for training
TRAIN_ATTACKS = [
    ('clean', identity_attack, 0.3),        # 30% clean
    ('jpeg_70', lambda x: jpeg_attack(x, 70), 0.2),
    ('jpeg_50', lambda x: jpeg_attack(x, 50), 0.1),
    ('noise_0.03', lambda x: gaussian_noise_attack(x, 0.03), 0.15),
    ('noise_0.05', lambda x: gaussian_noise_attack(x, 0.05), 0.1),
    ('blur_3', lambda x: gaussian_blur_attack(x, 3), 0.1),
    ('resize_0.75', lambda x: resize_attack(x, 0.75), 0.05),
]


class VAEWrapper:
    """VAE wrapper."""
    
    def __init__(self, model_id='runwayml/stable-diffusion-v1-5', device='cpu'):
        self.device = device
        self.model_id = model_id
        self._vae = None
        self.scaling_factor = 0.18215
        
    def load(self):
        if self._vae is None:
            print("Loading VAE...")
            from diffusers import AutoencoderKL
            self._vae = AutoencoderKL.from_pretrained(
                self.model_id, subfolder='vae', torch_dtype=torch.float32
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
            print("VAE unloaded")
    
    @torch.no_grad()
    def decode(self, z):
        vae = self.load()
        return vae.decode(z / self.scaling_factor).sample
    
    @torch.no_grad()
    def encode(self, img):
        vae = self.load()
        return vae.encode(img).latent_dist.mean * self.scaling_factor


class AttackAwareTrainer:
    """Trainer with attack-aware precomputation."""
    
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
        self.optimizer = torch.optim.AdamW(self.all_params, lr=config.get('lr', 1e-4))
        
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
    
    @torch.no_grad()
    def precompute_attacked_data(self, latents, num_samples=5000, batch_size=4, cache_path=None):
        """
        Precompute attacked latents: embed → decode → attack → encode
        
        Creates training data with mix of clean and attacked samples.
        """
        if cache_path and os.path.exists(cache_path):
            print(f"Loading cached attack data from {cache_path}")
            data = torch.load(cache_path, map_location='cpu', weights_only=False)
            return data['z_orig'], data['watermarks'], data['z_attacked'], data['attack_types']
        
        print(f"Precomputing attacked latents for {num_samples} samples...")
        
        self._set_eval_mode()
        vae = VAEWrapper(device=self.device)
        
        indices = torch.randperm(len(latents))[:num_samples]
        selected_latents = latents[indices]
        
        all_z_orig = []
        all_watermarks = []
        all_z_attacked = []
        all_attack_types = []
        
        # Normalize attack weights
        total_weight = sum(w for _, _, w in TRAIN_ATTACKS)
        attack_probs = [w / total_weight for _, _, w in TRAIN_ATTACKS]
        
        for i in tqdm(range(0, num_samples, batch_size), desc="Precomputing attacks"):
            z_batch = selected_latents[i:i+batch_size].to(self.device)
            B = z_batch.shape[0]
            
            # Generate watermarks
            w_batch = torch.randn(B, self.config.get('w_dim', 32), device=self.device)
            
            # Embed watermark
            z_wm = self.embed_watermark(z_batch, w_batch)
            
            # Decode to image
            img_wm = vae.decode(z_wm)
            
            # Select random attack for each sample in batch
            for j in range(B):
                attack_idx = np.random.choice(len(TRAIN_ATTACKS), p=attack_probs)
                attack_name, attack_fn, _ = TRAIN_ATTACKS[attack_idx]
                
                # Apply attack
                img_attacked = attack_fn(img_wm[j:j+1])
                
                # Encode back to latent
                z_attacked = vae.encode(img_attacked)
                
                all_z_orig.append(z_batch[j:j+1].cpu())
                all_watermarks.append(w_batch[j:j+1].cpu())
                all_z_attacked.append(z_attacked.cpu())
                all_attack_types.append(attack_idx)
            
            if self.device.type == 'mps':
                torch.mps.empty_cache()
        
        vae.unload()
        
        z_orig = torch.cat(all_z_orig, dim=0)
        watermarks = torch.cat(all_watermarks, dim=0)
        z_attacked = torch.cat(all_z_attacked, dim=0)
        attack_types = torch.tensor(all_attack_types)
        
        if cache_path:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            torch.save({
                'z_orig': z_orig,
                'watermarks': watermarks, 
                'z_attacked': z_attacked,
                'attack_types': attack_types,
                'attack_names': [name for name, _, _ in TRAIN_ATTACKS]
            }, cache_path)
            print(f"Attack data cached to {cache_path}")
            
            # Print attack distribution
            print("Attack distribution:")
            for idx, (name, _, _) in enumerate(TRAIN_ATTACKS):
                count = (attack_types == idx).sum().item()
                print(f"  {name}: {count} ({100*count/len(attack_types):.1f}%)")
        
        return z_orig, watermarks, z_attacked, attack_types
    
    def train_step_attacked(self, z_orig, w_batch, z_attacked):
        """Training step on attacked latents."""
        # Extract watermark from attacked latent
        w_pred_att_l, w_pred_att_h = self.extract_watermark(z_attacked)
        
        # Also extract from fresh embedding
        z_wm = self.embed_watermark(z_orig, w_batch)
        w_pred_l, w_pred_h = self.extract_watermark(z_wm)
        
        # Losses
        loss_attack = F.mse_loss(w_pred_att_l, w_batch) + F.mse_loss(w_pred_att_h, w_batch)
        loss_clean = F.mse_loss(w_pred_l, w_batch) + F.mse_loss(w_pred_h, w_batch)
        loss_cons = F.mse_loss(w_pred_att_l, w_pred_att_h)
        loss_latent = F.mse_loss(z_wm, z_orig)
        
        total_loss = (
            2.0 * loss_attack +   # Primary: extract from attacked
            0.5 * loss_clean +    # Maintain clean extraction
            0.3 * loss_cons +
            1.0 * loss_latent
        )
        
        bit_acc_att = self.compute_bit_accuracy(w_batch, w_pred_att_l, w_pred_att_h)
        bit_acc_clean = self.compute_bit_accuracy(w_batch, w_pred_l, w_pred_h)
        
        return total_loss, {
            'loss': total_loss.item(),
            'bit_acc_attacked': bit_acc_att,
            'bit_acc_clean': bit_acc_clean
        }
    
    def train_epoch(self, z_orig, watermarks, z_attacked, epoch, batch_size=32):
        """Train one epoch on attacked data."""
        self._set_train_mode()
        
        dataset = TensorDataset(z_orig, watermarks, z_attacked)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        
        epoch_metrics = {'loss': [], 'bit_acc_attacked': [], 'bit_acc_clean': []}
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for z_batch, w_batch, z_att_batch in pbar:
            z_batch = z_batch.to(self.device)
            w_batch = w_batch.to(self.device)
            z_att_batch = z_att_batch.to(self.device)
            
            self.optimizer.zero_grad()
            loss, metrics = self.train_step_attacked(z_batch, w_batch, z_att_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.all_params, 1.0)
            self.optimizer.step()
            
            for k, v in metrics.items():
                epoch_metrics[k].append(v)
            
            pbar.set_postfix(
                loss=f"{metrics['loss']:.3f}",
                att_acc=f"{metrics['bit_acc_attacked']:.3f}"
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
    def evaluate_attacks(self, latents, n_samples=100):
        """Evaluate under various attacks."""
        self._set_eval_mode()
        vae = VAEWrapper(device=self.device)
        
        results = {'clean_roundtrip': []}
        for name, _, _ in TRAIN_ATTACKS:
            results[name] = []
        
        indices = torch.randperm(len(latents))[:n_samples]
        batch_size = 4
        
        for i in tqdm(range(0, n_samples, batch_size), desc="Eval attacks"):
            idx = indices[i:i+batch_size]
            z_batch = latents[idx].to(self.device)
            B = z_batch.shape[0]
            w_batch = torch.randn(B, self.config.get('w_dim', 32), device=self.device)
            
            z_wm = self.embed_watermark(z_batch, w_batch)
            img_wm = vae.decode(z_wm)
            
            for name, attack_fn, _ in TRAIN_ATTACKS:
                img_att = attack_fn(img_wm)
                z_att = vae.encode(img_att)
                w_pred_l, w_pred_h = self.extract_watermark(z_att)
                acc = self.compute_bit_accuracy(w_batch, w_pred_l, w_pred_h)
                results[name].append(acc)
            
            if self.device.type == 'mps':
                torch.mps.empty_cache()
        
        vae.unload()
        return {k: np.mean(v) for k, v in results.items()}


def main():
    parser = argparse.ArgumentParser(description="Attack-Aware Training")
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--samples', type=int, default=10000)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--eval-interval', type=int, default=10)
    parser.add_argument('--recompute', action='store_true')
    args = parser.parse_args()
    
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 
                          'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load latents
    num_images = config.get('num_images', 20000)
    image_size = config.get('image_size', 256)
    cache_path = f"cache/latents_{num_images}_{image_size}.pt"
    
    print(f"Loading latents from {cache_path}...")
    data = torch.load(cache_path, map_location='cpu', weights_only=False)
    latents = data['latents']
    print(f"Loaded {len(latents)} latents")
    
    # Create trainer
    trainer = AttackAwareTrainer(config, device)
    
    # Output dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"results/attack_aware_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Resume
    start_epoch = 0
    if args.resume:
        start_epoch = trainer.load_checkpoint(args.resume)
    
    print("\n" + "="*60)
    print("ATTACK-AWARE TRAINING")
    print("="*60)
    print(f"Epochs: {args.epochs}")
    print(f"Samples: {args.samples}")
    print(f"Attacks: {[name for name, _, _ in TRAIN_ATTACKS]}")
    print(f"Output: {output_dir}")
    print("="*60 + "\n")
    
    # Precompute attacked data
    attack_cache = f"{output_dir}/attack_cache_{args.samples}.pt"
    if args.recompute and os.path.exists(attack_cache):
        os.remove(attack_cache)
    
    z_orig, watermarks, z_attacked, attack_types = trainer.precompute_attacked_data(
        latents,
        num_samples=args.samples,
        batch_size=4,
        cache_path=attack_cache
    )
    
    # Lower LR for fine-tuning
    for param_group in trainer.optimizer.param_groups:
        param_group['lr'] = config.get('lr', 1e-4) * 0.3
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(trainer.optimizer, T_max=args.epochs)
    
    best_att_acc = 0.0
    
    for epoch in range(start_epoch, args.epochs):
        metrics = trainer.train_epoch(z_orig, watermarks, z_attacked, epoch + 1, args.batch_size)
        scheduler.step()
        
        print(f"Epoch {epoch+1}: Loss={metrics['loss']:.4f}, "
              f"Attack Acc={metrics['bit_acc_attacked']:.4f}, Clean Acc={metrics['bit_acc_clean']:.4f}")
        
        if metrics['bit_acc_attacked'] > best_att_acc:
            best_att_acc = metrics['bit_acc_attacked']
            trainer.save_checkpoint(f"{output_dir}/best_attack.pt", epoch+1, metrics)
            print(f"  New best attack accuracy: {best_att_acc:.4f}")
        
        if (epoch + 1) % args.eval_interval == 0:
            eval_metrics = trainer.evaluate_attacks(latents, n_samples=100)
            print(f"  Eval per attack:")
            for name, acc in eval_metrics.items():
                print(f"    {name}: {acc:.4f}")
        
        if (epoch + 1) % 10 == 0:
            trainer.save_checkpoint(f"{output_dir}/epoch{epoch+1}.pt", epoch+1, metrics)
    
    trainer.save_checkpoint(f"{output_dir}/final.pt", args.epochs, metrics)
    
    # Final evaluation
    print("\n" + "="*60)
    print("FINAL EVALUATION")
    print("="*60)
    
    final_metrics = trainer.evaluate_attacks(latents, n_samples=200)
    print("\nAccuracy per attack type:")
    for name, acc in final_metrics.items():
        print(f"  {name}: {acc:.4f}")
    
    avg_acc = np.mean(list(final_metrics.values()))
    print(f"\nAverage accuracy: {avg_acc:.4f}")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Full Attack-Aware Training with ALL attacks from evaluation:
- None (clean)
- Center Crop 0.1
- Random Crop 0.1
- Resize 0.7
- Rotation 15°
- Blur (Gaussian)
- Contrast 2.0
- Brightness 2.0
- JPEG 70
- Combined attacks

Optimized for MPS/M2 Mac with:
- Precomputed roundtrip latents
- Memory-efficient batch processing
- Regular checkpointing
- Progressive training (test with small samples first)
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

# Add parent directories to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.dirname(project_root))

from latent_watermarking.models.latent_split import LatentSplitter
from latent_watermarking.models.recombination import LatentRecombiner
from latent_watermarking.models.watermark_encoder import WatermarkEncoder
from latent_watermarking.models.watermark_decoder import WatermarkDecoder


# ============================================================
# ALL ATTACK FUNCTIONS
# ============================================================

def real_jpeg_attack(images, quality=70):
    """Real JPEG compression using PIL - most accurate."""
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


def center_crop_attack(images, crop_ratio=0.1):
    """Center crop: remove crop_ratio from each edge, resize back."""
    B, C, H, W = images.shape
    crop_h = int(H * crop_ratio)
    crop_w = int(W * crop_ratio)
    
    # Center crop
    cropped = images[:, :, crop_h:H-crop_h, crop_w:W-crop_w]
    
    # Resize back to original
    return F.interpolate(cropped, size=(H, W), mode='bilinear', align_corners=False)


def random_crop_attack(images, crop_ratio=0.1):
    """Random crop: remove crop_ratio randomly, resize back."""
    B, C, H, W = images.shape
    crop_h = int(H * crop_ratio * 2)  # Total crop amount
    crop_w = int(W * crop_ratio * 2)
    
    # Random offset
    top = torch.randint(0, crop_h + 1, (1,)).item() if crop_h > 0 else 0
    left = torch.randint(0, crop_w + 1, (1,)).item() if crop_w > 0 else 0
    
    # Crop
    new_h = H - crop_h
    new_w = W - crop_w
    cropped = images[:, :, top:top+new_h, left:left+new_w]
    
    # Resize back
    return F.interpolate(cropped, size=(H, W), mode='bilinear', align_corners=False)


def resize_attack(images, scale=0.7):
    """Resize down then back up."""
    B, C, H, W = images.shape
    h_small = max(1, int(H * scale))
    w_small = max(1, int(W * scale))
    
    down = F.interpolate(images, size=(h_small, w_small), mode='bilinear', align_corners=False)
    up = F.interpolate(down, size=(H, W), mode='bilinear', align_corners=False)
    return up


def rotation_attack(images, angle=15.0):
    """Rotate by given angle in degrees."""
    B, C, H, W = images.shape
    angle_rad = angle * np.pi / 180
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    
    theta = torch.tensor([
        [cos_a, -sin_a, 0],
        [sin_a, cos_a, 0]
    ], dtype=images.dtype, device=images.device).unsqueeze(0).expand(B, -1, -1)
    
    grid = F.affine_grid(theta, images.size(), align_corners=False)
    return F.grid_sample(images, grid, mode='bilinear', padding_mode='reflection', align_corners=False)


def gaussian_blur_attack(images, kernel_size=5):
    """Gaussian blur."""
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


def contrast_attack(images, factor=2.0):
    """Adjust contrast by given factor."""
    # Convert to [0, 1] range for processing
    img_01 = (images + 1) / 2
    mean = img_01.mean(dim=(2, 3), keepdim=True)
    adjusted = (img_01 - mean) * factor + mean
    # Convert back to [-1, 1]
    return (adjusted.clamp(0, 1) * 2 - 1)


def brightness_attack(images, factor=2.0):
    """Adjust brightness - factor > 1 brightens, < 1 darkens."""
    # For factor=2.0, we interpret as adding brightness shift
    # Let's use a reasonable interpretation: multiply luminance
    img_01 = (images + 1) / 2
    # Shift brightness - factor 2.0 means brighten significantly
    shift = (factor - 1) * 0.3  # Scale the shift
    adjusted = img_01 + shift
    return (adjusted.clamp(0, 1) * 2 - 1)


def combined_attack(images):
    """Apply combination of attacks (random subset)."""
    # Apply a random combination of mild attacks
    B, C, H, W = images.shape
    
    # Always apply at least one attack
    attacks_to_apply = []
    
    # Randomly select 2-3 attacks
    possible_attacks = [
        lambda x: resize_attack(x, 0.85),
        lambda x: gaussian_blur_attack(x, 3),
        lambda x: contrast_attack(x, 1.2),
        lambda x: brightness_attack(x, 1.1),
    ]
    
    n_attacks = np.random.randint(2, 4)
    selected = np.random.choice(len(possible_attacks), n_attacks, replace=False)
    
    result = images
    for idx in selected:
        result = possible_attacks[idx](result)
    
    return result


# Attack registry with names and probabilities for training
TRAINING_ATTACKS = [
    ('clean', None, 0.15),
    ('center_crop_0.1', lambda x: center_crop_attack(x, 0.1), 0.10),
    ('random_crop_0.1', lambda x: random_crop_attack(x, 0.1), 0.10),
    ('resize_0.7', lambda x: resize_attack(x, 0.7), 0.10),
    ('rotation_15', lambda x: rotation_attack(x, 15.0), 0.10),
    ('blur', lambda x: gaussian_blur_attack(x, 5), 0.10),
    ('contrast_2.0', lambda x: contrast_attack(x, 2.0), 0.08),
    ('brightness_2.0', lambda x: brightness_attack(x, 2.0), 0.08),
    ('jpeg_70', lambda x: real_jpeg_attack(x, 70), 0.12),
    ('combined', combined_attack, 0.07),
]

# Normalize probabilities
_total_prob = sum(p for _, _, p in TRAINING_ATTACKS)
TRAINING_ATTACKS = [(n, f, p/_total_prob) for n, f, p in TRAINING_ATTACKS]


class VAEWrapper:
    """Memory-efficient VAE wrapper with lazy loading."""
    
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
    """Trainer with comprehensive attack support."""
    
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
        
        # Use slightly lower LR for attack-aware fine-tuning
        self.optimizer = torch.optim.AdamW(
            self.all_params, 
            lr=config.get('lr', 1e-4) * 0.5,
            weight_decay=1e-5
        )
        
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
    
    def train_step_with_attacks(self, z_orig, w_batch, vae):
        """Training step applying random attacks."""
        B = z_orig.shape[0]
        
        # Embed watermark
        z_wm = self.embed_watermark(z_orig, w_batch)
        
        # Decode to image
        img_wm = vae.decode(z_wm)
        
        # Select and apply random attacks for each sample
        attack_probs = [p for _, _, p in TRAINING_ATTACKS]
        
        img_attacked_list = []
        attack_names = []
        
        for i in range(B):
            attack_idx = np.random.choice(len(TRAINING_ATTACKS), p=attack_probs)
            attack_name, attack_fn, _ = TRAINING_ATTACKS[attack_idx]
            attack_names.append(attack_name)
            
            if attack_fn is not None:
                img_att = attack_fn(img_wm[i:i+1])
            else:
                img_att = img_wm[i:i+1]
            
            img_attacked_list.append(img_att)
        
        img_attacked = torch.cat(img_attacked_list, dim=0)
        
        # Re-encode to latent
        z_attacked = vae.encode(img_attacked)
        
        # Extract watermark from attacked latents
        w_pred_l, w_pred_h = self.extract_watermark(z_attacked)
        
        # Also extract from clean embedding for consistency
        w_pred_clean_l, w_pred_clean_h = self.extract_watermark(z_wm)
        
        # Losses
        loss_attacked = F.mse_loss(w_pred_l, w_batch) + F.mse_loss(w_pred_h, w_batch)
        loss_clean = F.mse_loss(w_pred_clean_l, w_batch) + F.mse_loss(w_pred_clean_h, w_batch)
        loss_cons = F.mse_loss(w_pred_l, w_pred_h)  # Cross-band consistency
        loss_latent = F.mse_loss(z_wm, z_orig)  # Preserve latent quality
        
        # Weighted combination - prioritize attack robustness
        total_loss = (
            3.0 * loss_attacked +   # Main focus: robustness
            0.5 * loss_clean +      # Clean accuracy
            0.3 * loss_cons +       # Band consistency
            0.5 * loss_latent       # Quality preservation
        )
        
        bit_acc_attacked = self.compute_bit_accuracy(w_batch, w_pred_l, w_pred_h)
        bit_acc_clean = self.compute_bit_accuracy(w_batch, w_pred_clean_l, w_pred_clean_h)
        
        return total_loss, {
            'loss': total_loss.item(),
            'loss_attacked': loss_attacked.item(),
            'loss_clean': loss_clean.item(),
            'bit_acc_attacked': bit_acc_attacked,
            'bit_acc_clean': bit_acc_clean,
            'attacks_used': attack_names
        }
    
    def train_epoch(self, z_orig, watermarks, epoch, batch_size=4):
        """Train one epoch with attacks."""
        self._set_train_mode()
        vae = self._get_vae()
        
        dataset = TensorDataset(z_orig, watermarks)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        
        epoch_metrics = {
            'loss': [], 'loss_attacked': [], 'loss_clean': [],
            'bit_acc_attacked': [], 'bit_acc_clean': []
        }
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for z_batch, w_batch in pbar:
            z_batch = z_batch.to(self.device)
            w_batch = w_batch.to(self.device)
            
            self.optimizer.zero_grad()
            loss, metrics = self.train_step_with_attacks(z_batch, w_batch, vae)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.all_params, 1.0)
            self.optimizer.step()
            
            for k in epoch_metrics:
                if k in metrics:
                    epoch_metrics[k].append(metrics[k])
            
            pbar.set_postfix(
                loss=f"{metrics['loss']:.3f}",
                att=f"{metrics['bit_acc_attacked']:.3f}",
                clean=f"{metrics['bit_acc_clean']:.3f}"
            )
            
            if self.device.type == 'mps':
                torch.mps.empty_cache()
        
        return {k: np.mean(v) for k, v in epoch_metrics.items() if v}
    
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
        print(f"  Checkpoint saved: {path}")
    
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
                print("  Warning: Could not load optimizer state")
        self.alpha_l = checkpoint.get('alpha_l', self.alpha_l)
        self.alpha_h = checkpoint.get('alpha_h', self.alpha_h)
        print(f"  Loaded checkpoint from {path}")
        return checkpoint.get('epoch', 0)
    
    @torch.no_grad()
    def evaluate_all_attacks(self, z_orig, watermarks, n_samples=50):
        """Comprehensive evaluation under all attacks."""
        self._set_eval_mode()
        vae = self._get_vae()
        
        # All attacks to evaluate (matching the user's required list)
        eval_attacks = [
            ('None', None),
            ('C. Crop 0.1', lambda x: center_crop_attack(x, 0.1)),
            ('R. Crop 0.1', lambda x: random_crop_attack(x, 0.1)),
            ('Resize 0.7', lambda x: resize_attack(x, 0.7)),
            ('Rot. 15', lambda x: rotation_attack(x, 15.0)),
            ('Blur', lambda x: gaussian_blur_attack(x, 5)),
            ('Contr. 2.0', lambda x: contrast_attack(x, 2.0)),
            ('Bright. 2.0', lambda x: brightness_attack(x, 2.0)),
            ('JPEG 70', lambda x: real_jpeg_attack(x, 70)),
            ('Comb.', combined_attack),
        ]
        
        results = {name: [] for name, _ in eval_attacks}
        
        indices = torch.randperm(len(z_orig))[:n_samples]
        
        for i in tqdm(indices, desc="Evaluating"):
            z = z_orig[i:i+1].to(self.device)
            w = watermarks[i:i+1].to(self.device)
            
            z_wm = self.embed_watermark(z, w)
            img_wm = vae.decode(z_wm)
            
            for name, attack_fn in eval_attacks:
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
    parser.add_argument('--resume', type=str, help='Checkpoint to resume from')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--samples', type=int, default=10000, help='Number of training samples')
    parser.add_argument('--eval-interval', type=int, default=10)
    parser.add_argument('--save-interval', type=int, default=10)
    parser.add_argument('--test-run', action='store_true', help='Quick test with 100 samples')
    args = parser.parse_args()
    
    # Test run mode
    if args.test_run:
        args.samples = 100
        args.epochs = 5
        args.eval_interval = 2
        args.save_interval = 2
        print("\n*** TEST RUN MODE - Using minimal samples ***\n")
    
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load cache
    print(f"Loading cache: {args.cache}")
    
    # Try different cache formats
    if os.path.exists(args.cache):
        cache = torch.load(args.cache, map_location='cpu', weights_only=False)
        if 'z_orig' in cache:
            z_orig = cache['z_orig'][:args.samples]
            watermarks = cache['watermarks'][:args.samples]
        elif 'latents' in cache:
            z_orig = cache['latents'][:args.samples]
            watermarks = torch.randn(len(z_orig), config.get('w_dim', 32))
        else:
            z_orig = list(cache.values())[0][:args.samples]
            watermarks = torch.randn(len(z_orig), config.get('w_dim', 32))
    else:
        # Fall back to latents cache
        latent_cache = 'cache/latents_20000_256.pt'
        print(f"  Roundtrip cache not found, using: {latent_cache}")
        cache = torch.load(latent_cache, map_location='cpu', weights_only=False)
        z_orig = cache['latents'][:args.samples]
        watermarks = torch.randn(len(z_orig), config.get('w_dim', 32))
    
    # Ensure binary watermarks
    watermarks = (watermarks > 0).float() * 2 - 1
    
    print(f"Training samples: {len(z_orig)}")
    
    # Create trainer
    trainer = AttackAwareTrainer(config, device)
    
    # Load checkpoint if specified
    start_epoch = 0
    if args.resume:
        if os.path.exists(args.resume):
            start_epoch = trainer.load_checkpoint(args.resume)
        else:
            print(f"  Warning: Checkpoint {args.resume} not found, training from scratch")
    
    # Output dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"results/attack_aware_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("ATTACK-AWARE TRAINING")
    print("="*60)
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Samples: {len(z_orig)}")
    print(f"Attacks: {[name for name, _, _ in TRAINING_ATTACKS]}")
    print(f"Output: {output_dir}")
    print("="*60 + "\n")
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        trainer.optimizer, 
        T_max=args.epochs,
        eta_min=1e-6
    )
    
    best_avg_acc = 0.0
    
    # Initial evaluation
    print("Initial evaluation...")
    initial_eval = trainer.evaluate_all_attacks(z_orig, watermarks, n_samples=min(30, len(z_orig)))
    print("\nInitial Attack Robustness:")
    for name, acc in initial_eval.items():
        print(f"  {name:15s}: {acc:.4f}")
    avg_acc = np.mean(list(initial_eval.values()))
    print(f"\n  Average: {avg_acc:.4f}")
    
    # Training loop
    for epoch in range(start_epoch + 1, start_epoch + args.epochs + 1):
        metrics = trainer.train_epoch(z_orig, watermarks, epoch, batch_size=args.batch_size)
        scheduler.step()
        
        lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch}: Loss={metrics['loss']:.4f}, "
              f"Attack Acc={metrics['bit_acc_attacked']:.4f}, "
              f"Clean Acc={metrics['bit_acc_clean']:.4f}, "
              f"LR={lr:.2e}")
        
        # Save checkpoint
        if epoch % args.save_interval == 0:
            trainer.save_checkpoint(
                f"{output_dir}/epoch_{epoch}.pt", 
                epoch, 
                metrics
            )
        
        # Evaluation
        if epoch % args.eval_interval == 0:
            eval_results = trainer.evaluate_all_attacks(
                z_orig, watermarks, 
                n_samples=min(50, len(z_orig))
            )
            
            print("\n  Per-attack accuracy:")
            for name, acc in eval_results.items():
                print(f"    {name:15s}: {acc:.4f}")
            
            avg_acc = np.mean(list(eval_results.values()))
            print(f"\n  Average: {avg_acc:.4f}")
            
            if avg_acc > best_avg_acc:
                best_avg_acc = avg_acc
                trainer.save_checkpoint(f"{output_dir}/best.pt", epoch, {**metrics, 'eval': eval_results})
                print(f"  *** New best average: {best_avg_acc:.4f} ***")
            
            print()
    
    # Final checkpoint
    trainer.save_checkpoint(f"{output_dir}/final.pt", args.epochs, metrics)
    
    # Final comprehensive evaluation
    print("\n" + "="*60)
    print("FINAL EVALUATION")
    print("="*60)
    
    final_eval = trainer.evaluate_all_attacks(z_orig, watermarks, n_samples=min(100, len(z_orig)))
    
    print("\nFinal Attack Robustness:")
    print("-" * 35)
    for name, acc in final_eval.items():
        print(f"  {name:15s}: {acc:.4f}")
    print("-" * 35)
    
    avg_final = np.mean(list(final_eval.values()))
    print(f"\n  Final Average: {avg_final:.4f}")
    print(f"  Best Average:  {best_avg_acc:.4f}")
    print(f"\nResults saved to: {output_dir}")
    
    # Save results summary
    with open(f"{output_dir}/results.txt", 'w') as f:
        f.write("Attack-Aware Training Results\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Epochs: {args.epochs}\n")
        f.write(f"Samples: {len(z_orig)}\n")
        f.write(f"Best Average Accuracy: {best_avg_acc:.4f}\n\n")
        f.write("Final Per-Attack Accuracy:\n")
        for name, acc in final_eval.items():
            f.write(f"  {name:15s}: {acc:.4f}\n")


if __name__ == "__main__":
    main()

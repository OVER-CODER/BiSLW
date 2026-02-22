#!/usr/bin/env python3
"""
Efficient training script for latent watermarking.
Precomputes VAE latents to avoid heavy VAE operations during training.
Optimized for M2 Mac with limited memory.
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

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from latent_watermarking.models.latent_split import LatentSplitter
from latent_watermarking.models.recombination import LatentRecombiner
from latent_watermarking.models.watermark_encoder import WatermarkEncoder
from latent_watermarking.models.watermark_decoder import WatermarkDecoder
from latent_watermarking.attacks.latent_noise import LatentNoiseAttack


class LightweightVAE:
    """Lightweight VAE wrapper - only used for precomputation, not during training."""
    
    def __init__(self, model_id='runwayml/stable-diffusion-v1-5', device='cpu'):
        self.device = device
        self.model_id = model_id
        self._vae = None
        
    def _load_vae(self):
        """Lazy load VAE only when needed."""
        if self._vae is None:
            print("Loading VAE for precomputation (one-time)...")
            from diffusers import AutoencoderKL
            self._vae = AutoencoderKL.from_pretrained(
                self.model_id,
                subfolder='vae',
                torch_dtype=torch.float32
            ).to(self.device)
            self._vae.eval()
            for p in self._vae.parameters():
                p.requires_grad = False
        return self._vae
    
    @torch.no_grad()
    def encode(self, x):
        """Encode images to latents."""
        vae = self._load_vae()
        # VAE expects [0, 1] input
        x_norm = (x + 1) / 2  # [-1, 1] -> [0, 1]
        latent = vae.encode(x_norm).latent_dist.mean
        return latent * 0.18215  # SD scaling factor
    
    @torch.no_grad() 
    def decode(self, z):
        """Decode latents to images."""
        vae = self._load_vae()
        z_scaled = z / 0.18215
        img = vae.decode(z_scaled).sample
        return img * 2 - 1  # [0, 1] -> [-1, 1]
    
    def unload(self):
        """Free VAE memory after precomputation."""
        if self._vae is not None:
            del self._vae
            self._vae = None
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("VAE unloaded to free memory")


def precompute_latents(num_images, image_size, latent_size, device, cache_path):
    """Precompute or load cached latents."""
    
    if os.path.exists(cache_path):
        print(f"Loading cached latents from {cache_path}")
        data = torch.load(cache_path, map_location='cpu')
        return data['latents']
    
    print(f"Precomputing {num_images} latents...")
    
    # Use VAE for precomputation
    vae = LightweightVAE(device=device)
    
    all_latents = []
    batch_size = 4  # Small batch for memory
    
    for i in tqdm(range(0, num_images, batch_size), desc="Precomputing"):
        current_batch = min(batch_size, num_images - i)
        
        # Generate synthetic images (or load real ones)
        images = torch.randn(current_batch, 3, image_size, image_size, device=device)
        images = torch.clamp(images, -1, 1)
        
        # Encode to latents
        latents = vae.encode(images)
        all_latents.append(latents.cpu())
        
        # Clear memory
        del images, latents
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    
    # Unload VAE to free memory
    vae.unload()
    
    # Concatenate all latents
    all_latents = torch.cat(all_latents, dim=0)
    
    # Save cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    torch.save({'latents': all_latents}, cache_path)
    print(f"Latents cached to {cache_path}")
    
    return all_latents


class EfficientTrainer:
    """Memory-efficient trainer that works directly on precomputed latents."""
    
    def __init__(self, config, device):
        self.config = config
        self.device = device
        
        # Initialize lightweight models
        w_dim = config.get('w_dim', 32)
        
        self.splitter = LatentSplitter(mode=config.get('latent_split', 'dct')).to(device)
        self.recombiner = LatentRecombiner(mode=config.get('latent_split', 'dct')).to(device)
        
        self.encoder_l = WatermarkEncoder(watermark_dim=w_dim).to(device)
        self.encoder_h = WatermarkEncoder(watermark_dim=w_dim).to(device)
        self.decoder_l = WatermarkDecoder(watermark_dim=w_dim).to(device)
        self.decoder_h = WatermarkDecoder(watermark_dim=w_dim).to(device)
        
        # Simple noise attack (no VAE needed)
        self.attack = LatentNoiseAttack().to(device)
        
        # Alpha values
        self.alpha_l = config.get('alpha_l', 0.3)
        self.alpha_h = config.get('alpha_h', 0.15)
        
        # Optimizer - only watermark encoder/decoder params
        params = (
            list(self.encoder_l.parameters()) +
            list(self.encoder_h.parameters()) +
            list(self.decoder_l.parameters()) +
            list(self.decoder_h.parameters())
        )
        self.optimizer = torch.optim.AdamW(params, lr=config.get('lr', 1e-4))
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.get('epochs', 10)
        )
        
        # Metrics
        self.best_bit_acc = 0.0
        
    def train_step(self, z_batch, w_batch):
        """Single training step on precomputed latents."""
        
        # Split latent
        z_low, z_high = self.splitter(z_batch)
        
        # Encode watermark
        z_low_wm = self.encoder_l(z_low, w_batch, alpha=self.alpha_l)
        z_high_wm = self.encoder_h(z_high, w_batch, alpha=self.alpha_h)
        z_wm = self.recombiner(z_low_wm, z_high_wm)
        
        # Decode watermark (clean)
        z_wm_low, z_wm_high = self.splitter(z_wm)
        w_pred_l = self.decoder_l(z_wm_low)
        w_pred_h = self.decoder_h(z_wm_high)
        
        # Apply noise attack
        self.attack.train()
        z_attacked = self.attack(z_wm)
        
        # Decode after attack
        z_att_low, z_att_high = self.splitter(z_attacked)
        w_pred_rob_l = self.decoder_l(z_att_low)
        w_pred_rob_h = self.decoder_h(z_att_high)
        
        # Losses
        # 1. Watermark recovery
        loss_w = F.mse_loss(w_pred_l, w_batch) + F.mse_loss(w_pred_h, w_batch)
        
        # 2. Cross-band consistency
        loss_cons = F.mse_loss(w_pred_l, w_pred_h)
        
        # 3. Latent preservation (lightweight - no VAE decode)
        loss_latent = F.mse_loss(z_wm, z_batch)
        
        # 4. Robustness
        loss_robust = F.mse_loss(w_pred_rob_l, w_batch) + F.mse_loss(w_pred_rob_h, w_batch)
        
        # Combined loss
        total_loss = (
            self.config.get('lambda_w', 1.0) * loss_w +
            self.config.get('lambda_cons', 0.3) * loss_cons +
            self.config.get('lambda_latent', 2.0) * loss_latent +
            self.config.get('lambda_robust', 0.3) * loss_robust
        )
        
        # Compute bit accuracy
        with torch.no_grad():
            bits_true = (w_batch > 0).float()
            bits_pred = ((w_pred_l + w_pred_h) / 2 > 0).float()
            bit_acc = (bits_true == bits_pred).float().mean().item()
        
        return total_loss, {
            'loss': total_loss.item(),
            'loss_w': loss_w.item(),
            'loss_latent': loss_latent.item(),
            'bit_acc': bit_acc
        }
    
    def train_epoch(self, dataloader, epoch):
        """Train for one epoch."""
        self.encoder_l.train()
        self.encoder_h.train()
        self.decoder_l.train()
        self.decoder_h.train()
        
        epoch_loss = []
        epoch_bit_acc = []
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for batch in pbar:
            z_batch = batch[0].to(self.device)
            B = z_batch.shape[0]
            
            # Generate random watermark
            w_batch = torch.randn(B, self.config.get('w_dim', 32), device=self.device)
            
            # Training step
            self.optimizer.zero_grad()
            loss, metrics = self.train_step(z_batch, w_batch)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                list(self.encoder_l.parameters()) +
                list(self.encoder_h.parameters()) +
                list(self.decoder_l.parameters()) +
                list(self.decoder_h.parameters()),
                1.0
            )
            
            self.optimizer.step()
            
            epoch_loss.append(metrics['loss'])
            epoch_bit_acc.append(metrics['bit_acc'])
            
            pbar.set_postfix(
                loss=f"{metrics['loss']:.3f}",
                bit_acc=f"{metrics['bit_acc']:.3f}"
            )
            
            # Memory cleanup
            if self.device.type == 'mps':
                torch.mps.empty_cache()
        
        self.scheduler.step()
        
        avg_loss = np.mean(epoch_loss)
        avg_bit_acc = np.mean(epoch_bit_acc)
        
        print(f"\nEpoch {epoch}: Loss={avg_loss:.4f}, Bit Acc={avg_bit_acc:.4f}")
        
        return avg_loss, avg_bit_acc
    
    def save_checkpoint(self, path, epoch, metrics):
        """Save model checkpoint."""
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
    
    def evaluate(self, dataloader):
        """Evaluate model."""
        self.encoder_l.eval()
        self.encoder_h.eval()
        self.decoder_l.eval()
        self.decoder_h.eval()
        
        all_bit_acc = []
        all_bit_acc_attacked = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                z_batch = batch[0].to(self.device)
                B = z_batch.shape[0]
                w_batch = torch.randn(B, self.config.get('w_dim', 32), device=self.device)
                
                # Encode watermark
                z_low, z_high = self.splitter(z_batch)
                z_low_wm = self.encoder_l(z_low, w_batch, alpha=self.alpha_l)
                z_high_wm = self.encoder_h(z_high, w_batch, alpha=self.alpha_h)
                z_wm = self.recombiner(z_low_wm, z_high_wm)
                
                # Decode clean
                z_wm_low, z_wm_high = self.splitter(z_wm)
                w_pred_l = self.decoder_l(z_wm_low)
                w_pred_h = self.decoder_h(z_wm_high)
                
                # Bit accuracy (clean)
                bits_true = (w_batch > 0).float()
                bits_pred = ((w_pred_l + w_pred_h) / 2 > 0).float()
                bit_acc = (bits_true == bits_pred).float().mean().item()
                all_bit_acc.append(bit_acc)
                
                # Decode after attack
                self.attack.eval()
                z_attacked = self.attack(z_wm)
                z_att_low, z_att_high = self.splitter(z_attacked)
                w_pred_att_l = self.decoder_l(z_att_low)
                w_pred_att_h = self.decoder_h(z_att_high)
                
                bits_pred_att = ((w_pred_att_l + w_pred_att_h) / 2 > 0).float()
                bit_acc_att = (bits_true == bits_pred_att).float().mean().item()
                all_bit_acc_attacked.append(bit_acc_att)
                
                if self.device.type == 'mps':
                    torch.mps.empty_cache()
        
        results = {
            'bit_accuracy_clean': np.mean(all_bit_acc),
            'bit_accuracy_attacked': np.mean(all_bit_acc_attacked),
        }
        
        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        print(f"Bit Accuracy (clean): {results['bit_accuracy_clean']:.4f}")
        print(f"Bit Accuracy (attacked): {results['bit_accuracy_attacked']:.4f}")
        
        return results


def main():
    parser = argparse.ArgumentParser(description='Efficient watermark training')
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    parser.add_argument('--config', default=os.path.join(script_dir, 'configs/default.yaml'))
    parser.add_argument('--num-images', type=int, default=10000)
    parser.add_argument('--output-dir', default=os.path.join(script_dir, 'results'))
    parser.add_argument('--cache-dir', default=os.path.join(script_dir, 'cache'))
    args = parser.parse_args()
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Override for efficiency
    config['batch_size'] = config.get('batch_size', 8)
    config['epochs'] = config.get('epochs', 10)
    
    # Setup device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # Precompute latents (VAE only used once, then unloaded)
    image_size = config.get('image_size', 256)
    latent_size = image_size // 8  # SD VAE downscales by 8
    cache_path = os.path.join(args.cache_dir, f'latents_{args.num_images}_{image_size}.pt')
    
    latents = precompute_latents(
        num_images=args.num_images,
        image_size=image_size,
        latent_size=latent_size,
        device=device,
        cache_path=cache_path
    )
    
    print(f"Latents shape: {latents.shape}")
    
    # Create dataloader from cached latents
    dataset = TensorDataset(latents)
    dataloader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output_dir, f'efficient_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    
    # Save config
    with open(os.path.join(output_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)
    
    # Initialize trainer
    trainer = EfficientTrainer(config, device)
    
    # Training loop
    print("\n" + "="*60)
    print("STARTING EFFICIENT TRAINING")
    print("="*60)
    print(f"Training on {len(latents)} precomputed latents")
    print(f"Epochs: {config['epochs']}")
    print(f"Batch size: {config['batch_size']}")
    print(f"VAE unloaded - training on latents only (fast!)")
    print("="*60 + "\n")
    
    best_bit_acc = 0.0
    
    for epoch in range(config['epochs']):
        loss, bit_acc = trainer.train_epoch(dataloader, epoch)
        
        # Save checkpoint
        if bit_acc > best_bit_acc:
            best_bit_acc = bit_acc
            trainer.save_checkpoint(
                os.path.join(output_dir, 'best_model.pth'),
                epoch,
                {'loss': loss, 'bit_acc': bit_acc}
            )
        
        if (epoch + 1) % 5 == 0:
            trainer.save_checkpoint(
                os.path.join(output_dir, f'checkpoint_epoch_{epoch}.pth'),
                epoch,
                {'loss': loss, 'bit_acc': bit_acc}
            )
    
    # Final evaluation
    print("\n" + "="*60)
    print("FINAL EVALUATION")
    print("="*60)
    
    results = trainer.evaluate(dataloader)
    
    # Save results
    with open(os.path.join(output_dir, 'results.yaml'), 'w') as f:
        yaml.dump(results, f)
    
    print(f"\nResults saved to {output_dir}")
    print("Training complete!")


if __name__ == '__main__':
    main()

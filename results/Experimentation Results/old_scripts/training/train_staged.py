#!/usr/bin/env python3
"""
Staged VAE-Roundtrip Training for Robust Latent Watermarking.

This script implements two-phase training:
- Phase 1: Fast latent-only training to learn basic watermark embedding
- Phase 2: VAE roundtrip training to learn robustness to decode→encode cycle

The key insight is that watermarks must survive:
  latent → VAE decode → image → VAE encode → latent
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


class VAEWrapper:
    """VAE wrapper for roundtrip training."""
    
    def __init__(self, model_id='runwayml/stable-diffusion-v1-5', device='cpu'):
        self.device = device
        self.model_id = model_id
        self._vae = None
        self.scaling_factor = 0.18215
        
    def load(self):
        """Load VAE into memory."""
        if self._vae is None:
            print("Loading VAE for roundtrip training...")
            from diffusers import AutoencoderKL
            self._vae = AutoencoderKL.from_pretrained(
                self.model_id,
                subfolder='vae',
                torch_dtype=torch.float32
            ).to(self.device)
            self._vae.eval()
            # Freeze VAE parameters
            for p in self._vae.parameters():
                p.requires_grad = False
        return self._vae
    
    def unload(self):
        """Free VAE memory."""
        if self._vae is not None:
            del self._vae
            self._vae = None
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("VAE unloaded")
    
    def decode(self, z):
        """Decode latent to image. No gradients through VAE."""
        vae = self.load()
        z_scaled = z / self.scaling_factor
        with torch.no_grad():
            img = vae.decode(z_scaled).sample
        return img  # [0, 1] range
    
    def encode(self, img):
        """Encode image to latent. No gradients through VAE."""
        vae = self.load()
        with torch.no_grad():
            latent = vae.encode(img).latent_dist.mean
        return latent * self.scaling_factor
    
    def roundtrip(self, z):
        """
        Full roundtrip: latent → image → latent
        This simulates what happens when watermarked latent is used in SD pipeline.
        """
        img = self.decode(z)
        z_rt = self.encode(img)
        return z_rt


class StagedTrainer:
    """
    Two-phase trainer for robust latent watermarking.
    
    Phase 1: Latent-only training (fast)
        - Works directly on precomputed latents
        - No VAE operations during training
        - Learns basic watermark embedding
        
    Phase 2: VAE roundtrip training (robust)
        - Includes VAE decode→encode in training loop
        - Learns to embed watermarks that survive VAE roundtrip
        - Uses gradient accumulation to handle larger memory footprint
    """
    
    def __init__(self, config, device):
        self.config = config
        self.device = device
        
        # Initialize watermark models
        w_dim = config.get('w_dim', 32)
        
        self.splitter = LatentSplitter(mode=config.get('latent_split', 'dct')).to(device)
        self.recombiner = LatentRecombiner(mode=config.get('latent_split', 'dct')).to(device)
        
        self.encoder_l = WatermarkEncoder(watermark_dim=w_dim).to(device)
        self.encoder_h = WatermarkEncoder(watermark_dim=w_dim).to(device)
        self.decoder_l = WatermarkDecoder(watermark_dim=w_dim).to(device)
        self.decoder_h = WatermarkDecoder(watermark_dim=w_dim).to(device)
        
        # Noise attack for additional robustness
        self.attack = LatentNoiseAttack().to(device)
        
        # Alpha values
        self.alpha_l = config.get('alpha_l', 0.02)
        self.alpha_h = config.get('alpha_h', 0.01)
        
        # Optimizer
        self.params = (
            list(self.encoder_l.parameters()) +
            list(self.encoder_h.parameters()) +
            list(self.decoder_l.parameters()) +
            list(self.decoder_h.parameters())
        )
        self.optimizer = torch.optim.AdamW(self.params, lr=config.get('lr', 1e-4))
        
        # VAE wrapper (lazy loaded)
        self.vae = None
        
        # Training state
        self.current_epoch = 0
        self.best_metrics = {'bit_acc_roundtrip': 0.0}
        
    def _get_vae(self):
        """Get or create VAE wrapper."""
        if self.vae is None:
            self.vae = VAEWrapper(device=self.device)
        return self.vae
    
    def embed_watermark(self, z, w):
        """Embed watermark into latent."""
        z_low, z_high = self.splitter(z)
        z_low_wm = self.encoder_l(z_low, w, alpha=self.alpha_l)
        z_high_wm = self.encoder_h(z_high, w, alpha=self.alpha_h)
        z_wm = self.recombiner(z_low_wm, z_high_wm)
        return z_wm
    
    def extract_watermark(self, z):
        """Extract watermark from latent."""
        z_low, z_high = self.splitter(z)
        w_pred_l = self.decoder_l(z_low)
        w_pred_h = self.decoder_h(z_high)
        return w_pred_l, w_pred_h
    
    def compute_bit_accuracy(self, w_true, w_pred_l, w_pred_h):
        """Compute bit accuracy."""
        bits_true = (w_true > 0).float()
        bits_pred = ((w_pred_l + w_pred_h) / 2 > 0).float()
        return (bits_true == bits_pred).float().mean().item()
    
    # ========== Phase 1: Latent-only training ==========
    
    def train_step_phase1(self, z_batch, w_batch):
        """
        Phase 1 training step - latent only, no VAE.
        Fast training to learn basic watermark embedding.
        """
        # Embed watermark
        z_wm = self.embed_watermark(z_batch, w_batch)
        
        # Extract watermark (clean)
        w_pred_l, w_pred_h = self.extract_watermark(z_wm)
        
        # Apply noise attack
        self.attack.train()
        z_attacked = self.attack(z_wm)
        w_pred_rob_l, w_pred_rob_h = self.extract_watermark(z_attacked)
        
        # Losses
        loss_w = F.mse_loss(w_pred_l, w_batch) + F.mse_loss(w_pred_h, w_batch)
        loss_cons = F.mse_loss(w_pred_l, w_pred_h)
        loss_latent = F.mse_loss(z_wm, z_batch)
        loss_robust = F.mse_loss(w_pred_rob_l, w_batch) + F.mse_loss(w_pred_rob_h, w_batch)
        
        total_loss = (
            self.config.get('lambda_w', 1.0) * loss_w +
            self.config.get('lambda_cons', 0.3) * loss_cons +
            self.config.get('lambda_latent', 2.0) * loss_latent +
            self.config.get('lambda_robust', 0.3) * loss_robust
        )
        
        bit_acc = self.compute_bit_accuracy(w_batch, w_pred_l, w_pred_h)
        
        return total_loss, {
            'loss': total_loss.item(),
            'loss_w': loss_w.item(),
            'loss_latent': loss_latent.item(),
            'bit_acc': bit_acc
        }
    
    def train_epoch_phase1(self, dataloader, epoch):
        """Train one epoch in Phase 1 (latent-only)."""
        self._set_train_mode()
        
        epoch_loss = []
        epoch_bit_acc = []
        
        pbar = tqdm(dataloader, desc=f"Phase1 Epoch {epoch}")
        for batch in pbar:
            z_batch = batch[0].to(self.device)
            B = z_batch.shape[0]
            w_batch = torch.randn(B, self.config.get('w_dim', 32), device=self.device)
            
            self.optimizer.zero_grad()
            loss, metrics = self.train_step_phase1(z_batch, w_batch)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.params, 1.0)
            self.optimizer.step()
            
            epoch_loss.append(metrics['loss'])
            epoch_bit_acc.append(metrics['bit_acc'])
            
            pbar.set_postfix(loss=f"{metrics['loss']:.3f}", bit_acc=f"{metrics['bit_acc']:.3f}")
            
            if self.device.type == 'mps':
                torch.mps.empty_cache()
        
        return np.mean(epoch_loss), np.mean(epoch_bit_acc)
    
    # ========== Phase 2: VAE Roundtrip training ==========
    
    def train_step_phase2(self, z_batch, w_batch):
        """
        Phase 2 training step - with VAE roundtrip.
        
        Key difference: We extract watermark from the RE-ENCODED latent,
        not the original watermarked latent. This forces the model to
        learn embeddings that survive the VAE decode→encode cycle.
        """
        vae = self._get_vae()
        
        # Embed watermark
        z_wm = self.embed_watermark(z_batch, w_batch)
        
        # ===== VAE ROUNDTRIP =====
        # This is the key step - simulate what happens in real usage
        z_roundtrip = vae.roundtrip(z_wm)
        
        # Extract watermark from roundtrip latent (this is what we optimize for!)
        w_pred_rt_l, w_pred_rt_h = self.extract_watermark(z_roundtrip)
        
        # Also extract from clean watermarked latent (for comparison)
        w_pred_l, w_pred_h = self.extract_watermark(z_wm)
        
        # Apply noise attack on roundtrip latent (double robustness)
        self.attack.eval()  # Use fixed noise during phase 2
        z_attacked = self.attack(z_roundtrip)
        w_pred_att_l, w_pred_att_h = self.extract_watermark(z_attacked)
        
        # ===== LOSSES =====
        # Primary loss: watermark recovery after roundtrip (MOST IMPORTANT)
        loss_roundtrip = F.mse_loss(w_pred_rt_l, w_batch) + F.mse_loss(w_pred_rt_h, w_batch)
        
        # Secondary loss: clean watermark recovery
        loss_w = F.mse_loss(w_pred_l, w_batch) + F.mse_loss(w_pred_h, w_batch)
        
        # Cross-band consistency after roundtrip
        loss_cons_rt = F.mse_loss(w_pred_rt_l, w_pred_rt_h)
        
        # Latent preservation (between original and watermarked)
        loss_latent = F.mse_loss(z_wm, z_batch)
        
        # Robustness after roundtrip + attack
        loss_robust = F.mse_loss(w_pred_att_l, w_batch) + F.mse_loss(w_pred_att_h, w_batch)
        
        # Combined loss - heavily weight roundtrip recovery
        total_loss = (
            self.config.get('lambda_roundtrip', 2.0) * loss_roundtrip +  # Primary objective
            self.config.get('lambda_w', 0.5) * loss_w +
            self.config.get('lambda_cons', 0.3) * loss_cons_rt +
            self.config.get('lambda_latent', 1.0) * loss_latent +
            self.config.get('lambda_robust', 0.5) * loss_robust
        )
        
        # Compute metrics
        bit_acc_latent = self.compute_bit_accuracy(w_batch, w_pred_l, w_pred_h)
        bit_acc_roundtrip = self.compute_bit_accuracy(w_batch, w_pred_rt_l, w_pred_rt_h)
        bit_acc_attacked = self.compute_bit_accuracy(w_batch, w_pred_att_l, w_pred_att_h)
        
        return total_loss, {
            'loss': total_loss.item(),
            'loss_roundtrip': loss_roundtrip.item(),
            'loss_latent': loss_latent.item(),
            'bit_acc_latent': bit_acc_latent,
            'bit_acc_roundtrip': bit_acc_roundtrip,
            'bit_acc_attacked': bit_acc_attacked
        }
    
    def train_epoch_phase2(self, dataloader, epoch, accumulation_steps=4):
        """
        Train one epoch in Phase 2 (with VAE roundtrip).
        Uses gradient accumulation to handle memory constraints.
        """
        self._set_train_mode()
        
        epoch_metrics = {
            'loss': [], 'loss_roundtrip': [], 'loss_latent': [],
            'bit_acc_latent': [], 'bit_acc_roundtrip': [], 'bit_acc_attacked': []
        }
        
        self.optimizer.zero_grad()
        accumulated = 0
        
        pbar = tqdm(dataloader, desc=f"Phase2 Epoch {epoch}")
        for batch_idx, batch in enumerate(pbar):
            z_batch = batch[0].to(self.device)
            B = z_batch.shape[0]
            w_batch = torch.randn(B, self.config.get('w_dim', 32), device=self.device)
            
            # Forward pass
            loss, metrics = self.train_step_phase2(z_batch, w_batch)
            
            # Scale loss for accumulation
            scaled_loss = loss / accumulation_steps
            scaled_loss.backward()
            
            accumulated += 1
            
            # Update weights every accumulation_steps
            if accumulated >= accumulation_steps:
                torch.nn.utils.clip_grad_norm_(self.params, 1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()
                accumulated = 0
            
            # Record metrics
            for k, v in metrics.items():
                epoch_metrics[k].append(v)
            
            pbar.set_postfix(
                loss=f"{metrics['loss']:.3f}",
                rt_acc=f"{metrics['bit_acc_roundtrip']:.3f}"
            )
            
            # Aggressive memory cleanup
            del z_batch, w_batch, loss
            if self.device.type == 'mps':
                torch.mps.empty_cache()
        
        # Final update if any gradients left
        if accumulated > 0:
            torch.nn.utils.clip_grad_norm_(self.params, 1.0)
            self.optimizer.step()
            self.optimizer.zero_grad()
        
        return {k: np.mean(v) for k, v in epoch_metrics.items()}
    
    # ========== Utilities ==========
    
    def _set_train_mode(self):
        """Set models to training mode."""
        self.encoder_l.train()
        self.encoder_h.train()
        self.decoder_l.train()
        self.decoder_h.train()
    
    def _set_eval_mode(self):
        """Set models to evaluation mode."""
        self.encoder_l.eval()
        self.encoder_h.eval()
        self.decoder_l.eval()
        self.decoder_h.eval()
    
    def save_checkpoint(self, path, epoch, metrics, phase):
        """Save checkpoint."""
        torch.save({
            'epoch': epoch,
            'phase': phase,
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
        """Load checkpoint."""
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
        return checkpoint.get('epoch', 0), checkpoint.get('phase', 1)
    
    def evaluate_latent_only(self, dataloader, n_samples=200):
        """Fast evaluation - latent only, no VAE roundtrip."""
        self._set_eval_mode()
        
        bit_accs = []
        samples_seen = 0
        n_batches = (n_samples + dataloader.batch_size - 1) // dataloader.batch_size
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Eval(latent)", total=n_batches):
                if samples_seen >= n_samples:
                    break
                    
                z_batch = batch[0].to(self.device)
                B = z_batch.shape[0]
                w_batch = torch.randn(B, self.config.get('w_dim', 32), device=self.device)
                
                z_wm = self.embed_watermark(z_batch, w_batch)
                w_pred_l, w_pred_h = self.extract_watermark(z_wm)
                bit_accs.append(self.compute_bit_accuracy(w_batch, w_pred_l, w_pred_h))
                
                samples_seen += B
        
        return {'bit_acc_latent': np.mean(bit_accs)}
    
    def evaluate_roundtrip(self, dataloader, n_samples=50):
        """Evaluate with VAE roundtrip - uses fewer samples since it's slow."""
        self._set_eval_mode()
        vae = self._get_vae()
        
        metrics = {
            'bit_acc_latent': [],
            'bit_acc_roundtrip': [],
            'mse_latent': []
        }
        
        samples_seen = 0
        n_batches = (n_samples + dataloader.batch_size - 1) // dataloader.batch_size
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Eval(roundtrip)", total=n_batches):
                if samples_seen >= n_samples:
                    break
                    
                z_batch = batch[0].to(self.device)
                B = z_batch.shape[0]
                w_batch = torch.randn(B, self.config.get('w_dim', 32), device=self.device)
                
                # Embed watermark
                z_wm = self.embed_watermark(z_batch, w_batch)
                
                # Extract from latent
                w_pred_l, w_pred_h = self.extract_watermark(z_wm)
                bit_acc_latent = self.compute_bit_accuracy(w_batch, w_pred_l, w_pred_h)
                
                # VAE roundtrip
                z_rt = vae.roundtrip(z_wm)
                w_pred_rt_l, w_pred_rt_h = self.extract_watermark(z_rt)
                bit_acc_rt = self.compute_bit_accuracy(w_batch, w_pred_rt_l, w_pred_rt_h)
                
                # Latent MSE
                mse = F.mse_loss(z_wm, z_batch).item()
                
                metrics['bit_acc_latent'].append(bit_acc_latent)
                metrics['bit_acc_roundtrip'].append(bit_acc_rt)
                metrics['mse_latent'].append(mse)
                
                samples_seen += B
                
                if self.device.type == 'mps':
                    torch.mps.empty_cache()
        
        return {k: np.mean(v) for k, v in metrics.items()}


def load_precomputed_latents(cache_path):
    """Load precomputed latents from cache."""
    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"No cached latents at {cache_path}. Run train_efficient.py first.")
    data = torch.load(cache_path, map_location='cpu')
    return data['latents']


def main():
    parser = argparse.ArgumentParser(description="Staged VAE-Roundtrip Training")
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--phase1-epochs', type=int, default=50, help='Epochs for Phase 1 (latent-only)')
    parser.add_argument('--phase2-epochs', type=int, default=50, help='Epochs for Phase 2 (VAE roundtrip)')
    parser.add_argument('--batch-size-p1', type=int, default=32, help='Batch size for Phase 1')
    parser.add_argument('--batch-size-p2', type=int, default=8, help='Batch size for Phase 2 (smaller for memory)')
    parser.add_argument('--accumulation-steps', type=int, default=4, help='Gradient accumulation steps for Phase 2')
    parser.add_argument('--phase2-subset', type=int, default=2000, help='Use subset of data for Phase 2 (default: 2000, use -1 for all)')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--skip-phase1', action='store_true', help='Skip Phase 1 (use with --resume)')
    parser.add_argument('--eval-interval', type=int, default=10, help='Evaluate every N epochs')
    args = parser.parse_args()
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Device
    device = torch.device('mps' if torch.backends.mps.is_available() else 
                          'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load precomputed latents
    num_images = config.get('num_images', 20000)
    image_size = config.get('image_size', 256)
    cache_path = f"cache/latents_{num_images}_{image_size}.pt"
    
    print(f"Loading precomputed latents from {cache_path}...")
    latents = load_precomputed_latents(cache_path)
    print(f"Loaded {len(latents)} latents, shape: {latents.shape}")
    
    # Create trainer
    trainer = StagedTrainer(config, device)
    
    # Setup output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"results/staged_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Resume if specified
    start_phase = 1
    start_epoch = 0
    if args.resume:
        start_epoch, start_phase = trainer.load_checkpoint(args.resume)
        print(f"Resuming from Phase {start_phase}, Epoch {start_epoch}")
    
    # Skip Phase 1 if specified
    if args.skip_phase1:
        start_phase = 2
        start_epoch = 0
    
    print("\n" + "="*60)
    print("STAGED VAE-ROUNDTRIP TRAINING")
    print("="*60)
    print(f"Phase 1: {args.phase1_epochs} epochs (latent-only, fast)")
    print(f"Phase 2: {args.phase2_epochs} epochs (VAE roundtrip, robust)")
    print(f"Phase 2 subset: {args.phase2_subset} images (for speed)")
    print(f"Batch sizes: Phase1={args.batch_size_p1}, Phase2={args.batch_size_p2}")
    print(f"Gradient accumulation: {args.accumulation_steps} steps")
    print(f"Output: {output_dir}")
    print("="*60 + "\n")
    
    # ==================== PHASE 1 ====================
    if start_phase == 1:
        print("\n" + "="*60)
        print("PHASE 1: Latent-Only Training")
        print("="*60)
        
        dataset_p1 = TensorDataset(latents)
        dataloader_p1 = DataLoader(
            dataset_p1, 
            batch_size=args.batch_size_p1,
            shuffle=True,
            drop_last=True
        )
        
        # Scheduler for Phase 1
        scheduler_p1 = torch.optim.lr_scheduler.CosineAnnealingLR(
            trainer.optimizer, T_max=args.phase1_epochs
        )
        
        for epoch in range(start_epoch, args.phase1_epochs):
            loss, bit_acc = trainer.train_epoch_phase1(dataloader_p1, epoch + 1)
            scheduler_p1.step()
            
            print(f"Phase1 Epoch {epoch+1}: Loss={loss:.4f}, Bit Acc={bit_acc:.4f}")
            
            # Fast latent-only evaluation during Phase 1 (no VAE roundtrip yet)
            if (epoch + 1) % args.eval_interval == 0 or epoch == args.phase1_epochs - 1:
                eval_loader = DataLoader(dataset_p1, batch_size=args.batch_size_p1, shuffle=False)
                eval_metrics = trainer.evaluate_latent_only(eval_loader, n_samples=200)
                print(f"  Eval: Latent Acc={eval_metrics['bit_acc_latent']:.4f}")
            
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                trainer.save_checkpoint(
                    f"{output_dir}/phase1_epoch{epoch+1}.pt",
                    epoch + 1,
                    {'loss': loss, 'bit_acc': bit_acc},
                    phase=1
                )
        
        # Save Phase 1 final checkpoint
        trainer.save_checkpoint(
            f"{output_dir}/phase1_final.pt",
            args.phase1_epochs,
            {'loss': loss, 'bit_acc': bit_acc},
            phase=1
        )
        print("\nPhase 1 complete!")
        
        # Reset for Phase 2
        start_epoch = 0
    
    # ==================== PHASE 2 ====================
    print("\n" + "="*60)
    print("PHASE 2: VAE Roundtrip Training")
    print("="*60)
    
    # Use subset of data for Phase 2 (VAE is slow)
    if args.phase2_subset > 0 and args.phase2_subset < len(latents):
        latents_p2 = latents[:args.phase2_subset]
        print(f"Using {len(latents_p2)} images for Phase 2 (subset for speed)")
    else:
        latents_p2 = latents
        print(f"Using all {len(latents_p2)} images for Phase 2")
    
    # Smaller batch size for Phase 2 (VAE uses more memory)
    dataset_p2 = TensorDataset(latents_p2)
    dataloader_p2 = DataLoader(
        dataset_p2,
        batch_size=args.batch_size_p2,
        shuffle=True,
        drop_last=True
    )
    
    # Reset optimizer with lower learning rate for Phase 2
    for param_group in trainer.optimizer.param_groups:
        param_group['lr'] = config.get('lr', 1e-4) * 0.5  # Lower LR for fine-tuning
    
    # Scheduler for Phase 2
    scheduler_p2 = torch.optim.lr_scheduler.CosineAnnealingLR(
        trainer.optimizer, T_max=args.phase2_epochs
    )
    
    best_roundtrip_acc = 0.0
    
    for epoch in range(start_epoch, args.phase2_epochs):
        metrics = trainer.train_epoch_phase2(
            dataloader_p2, 
            epoch + 1,
            accumulation_steps=args.accumulation_steps
        )
        scheduler_p2.step()
        
        print(f"Phase2 Epoch {epoch+1}: Loss={metrics['loss']:.4f}, "
              f"Latent Acc={metrics['bit_acc_latent']:.4f}, "
              f"Roundtrip Acc={metrics['bit_acc_roundtrip']:.4f}")
        
        # Track best roundtrip accuracy
        if metrics['bit_acc_roundtrip'] > best_roundtrip_acc:
            best_roundtrip_acc = metrics['bit_acc_roundtrip']
            trainer.save_checkpoint(
                f"{output_dir}/best_roundtrip.pt",
                epoch + 1,
                metrics,
                phase=2
            )
            print(f"  New best roundtrip accuracy: {best_roundtrip_acc:.4f}")
        
        # Periodic full evaluation (limited samples for speed)
        if (epoch + 1) % args.eval_interval == 0 or epoch == args.phase2_epochs - 1:
            eval_loader = DataLoader(dataset_p2, batch_size=args.batch_size_p2, shuffle=False)
            eval_metrics = trainer.evaluate_roundtrip(eval_loader, n_samples=50)
            print(f"  Full Eval: Latent Acc={eval_metrics['bit_acc_latent']:.4f}, "
                  f"Roundtrip Acc={eval_metrics['bit_acc_roundtrip']:.4f}")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            trainer.save_checkpoint(
                f"{output_dir}/phase2_epoch{epoch+1}.pt",
                epoch + 1,
                metrics,
                phase=2
            )
    
    # Save final checkpoint
    trainer.save_checkpoint(
        f"{output_dir}/final.pt",
        args.phase2_epochs,
        metrics,
        phase=2
    )
    
    # ==================== FINAL EVALUATION ====================
    print("\n" + "="*60)
    print("FINAL EVALUATION")
    print("="*60)
    
    # Use original full dataset for final eval
    dataset_full = TensorDataset(latents)
    eval_loader = DataLoader(dataset_full, batch_size=args.batch_size_p2, shuffle=False)
    final_metrics = trainer.evaluate_roundtrip(eval_loader, n_samples=200)
    
    print(f"Latent Bit Accuracy:    {final_metrics['bit_acc_latent']:.4f}")
    print(f"Roundtrip Bit Accuracy: {final_metrics['bit_acc_roundtrip']:.4f}")
    print(f"Latent MSE:             {final_metrics['mse_latent']:.6f}")
    
    # Compare to pre-roundtrip training baseline
    print("\n" + "="*60)
    print("COMPARISON TO BASELINE")
    print("="*60)
    print("Before roundtrip training:")
    print("  Latent Acc: ~80%")
    print("  Roundtrip Acc: ~54% (nearly random)")
    print(f"\nAfter roundtrip training:")
    print(f"  Latent Acc: {final_metrics['bit_acc_latent']*100:.1f}%")
    print(f"  Roundtrip Acc: {final_metrics['bit_acc_roundtrip']*100:.1f}%")
    
    improvement = (final_metrics['bit_acc_roundtrip'] - 0.54) / (1.0 - 0.54) * 100
    print(f"\nImprovement in roundtrip accuracy: {improvement:.1f}% of possible gain")
    
    print(f"\nResults saved to: {output_dir}")
    print("Done!")


if __name__ == "__main__":
    main()

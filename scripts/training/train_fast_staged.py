#!/usr/bin/env python3
"""
Fast VAE-Roundtrip Training via Precomputation.

Instead of running VAE decode→encode in every training step (slow),
we precompute roundtrip latents once and train on cached data.

Phase 1: Latent-only training (learns encoder/decoder basics)
Phase 2a: Precompute VAE roundtrip for watermarked latents (one-time, ~15 min)
Phase 2b: Fast training on cached roundtrip latents (~5 min for 50 epochs)
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


class VAEWrapper:
    """VAE wrapper for roundtrip operations."""
    
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
    def roundtrip(self, z):
        """latent → image → latent"""
        vae = self.load()
        z_scaled = z / self.scaling_factor
        img = vae.decode(z_scaled).sample
        z_rt = vae.encode(img).latent_dist.mean * self.scaling_factor
        return z_rt


class FastRoundtripTrainer:
    """
    Fast trainer using precomputed VAE roundtrip latents.
    
    Key insight: Once encoder is trained, we can:
    1. Generate watermarked latents (fast)
    2. Run VAE roundtrip once (slow, but only once)
    3. Cache the results
    4. Train decoder on cached data (fast)
    """
    
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
        
        # Separate optimizers for encoder and decoder
        self.encoder_params = list(self.encoder_l.parameters()) + list(self.encoder_h.parameters())
        self.decoder_params = list(self.decoder_l.parameters()) + list(self.decoder_h.parameters())
        self.all_params = self.encoder_params + self.decoder_params
        
        self.optimizer = torch.optim.AdamW(self.all_params, lr=config.get('lr', 1e-4))
        
    def embed_watermark(self, z, w):
        """Embed watermark into latent."""
        z_low, z_high = self.splitter(z)
        z_low_wm = self.encoder_l(z_low, w, alpha=self.alpha_l)
        z_high_wm = self.encoder_h(z_high, w, alpha=self.alpha_h)
        return self.recombiner(z_low_wm, z_high_wm)
    
    def extract_watermark(self, z):
        """Extract watermark from latent."""
        z_low, z_high = self.splitter(z)
        return self.decoder_l(z_low), self.decoder_h(z_high)
    
    def compute_bit_accuracy(self, w_true, w_pred_l, w_pred_h):
        bits_true = (w_true > 0).float()
        bits_pred = ((w_pred_l + w_pred_h) / 2 > 0).float()
        return (bits_true == bits_pred).float().mean().item()
    
    # ===== Phase 1: Latent-only training (same as before) =====
    
    def train_step_phase1(self, z_batch, w_batch):
        z_wm = self.embed_watermark(z_batch, w_batch)
        w_pred_l, w_pred_h = self.extract_watermark(z_wm)
        
        loss_w = F.mse_loss(w_pred_l, w_batch) + F.mse_loss(w_pred_h, w_batch)
        loss_cons = F.mse_loss(w_pred_l, w_pred_h)
        loss_latent = F.mse_loss(z_wm, z_batch)
        
        total_loss = (
            self.config.get('lambda_w', 1.0) * loss_w +
            self.config.get('lambda_cons', 0.3) * loss_cons +
            self.config.get('lambda_latent', 2.0) * loss_latent
        )
        
        bit_acc = self.compute_bit_accuracy(w_batch, w_pred_l, w_pred_h)
        return total_loss, {'loss': total_loss.item(), 'bit_acc': bit_acc}
    
    def train_epoch_phase1(self, dataloader, epoch):
        self._set_train_mode()
        epoch_loss, epoch_bit_acc = [], []
        
        pbar = tqdm(dataloader, desc=f"Phase1 Epoch {epoch}")
        for batch in pbar:
            z_batch = batch[0].to(self.device)
            w_batch = torch.randn(z_batch.shape[0], self.config.get('w_dim', 32), device=self.device)
            
            self.optimizer.zero_grad()
            loss, metrics = self.train_step_phase1(z_batch, w_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.all_params, 1.0)
            self.optimizer.step()
            
            epoch_loss.append(metrics['loss'])
            epoch_bit_acc.append(metrics['bit_acc'])
            pbar.set_postfix(loss=f"{metrics['loss']:.3f}", bit_acc=f"{metrics['bit_acc']:.3f}")
            
            if self.device.type == 'mps':
                torch.mps.empty_cache()
        
        return np.mean(epoch_loss), np.mean(epoch_bit_acc)
    
    # ===== Phase 2a: Precompute VAE roundtrip latents =====
    
    @torch.no_grad()
    def precompute_roundtrip_data(self, latents, num_samples=5000, batch_size=4, cache_path=None):
        """
        Precompute VAE roundtrip for watermarked latents.
        
        Returns cached data: (original_latents, watermarks, roundtrip_latents)
        """
        if cache_path and os.path.exists(cache_path):
            print(f"Loading cached roundtrip data from {cache_path}")
            data = torch.load(cache_path, map_location='cpu', weights_only=False)
            return data['z_orig'], data['watermarks'], data['z_roundtrip']
        
        print(f"Precomputing VAE roundtrip for {num_samples} samples...")
        
        self._set_eval_mode()
        vae = VAEWrapper(device=self.device)
        
        # Select subset
        indices = torch.randperm(len(latents))[:num_samples]
        selected_latents = latents[indices]
        
        all_z_orig = []
        all_watermarks = []
        all_z_roundtrip = []
        
        for i in tqdm(range(0, num_samples, batch_size), desc="Precomputing roundtrip"):
            z_batch = selected_latents[i:i+batch_size].to(self.device)
            
            # Generate random watermarks
            w_batch = torch.randn(z_batch.shape[0], self.config.get('w_dim', 32), device=self.device)
            
            # Embed watermark
            z_wm = self.embed_watermark(z_batch, w_batch)
            
            # VAE roundtrip (the slow part - only done once!)
            z_rt = vae.roundtrip(z_wm)
            
            all_z_orig.append(z_batch.cpu())
            all_watermarks.append(w_batch.cpu())
            all_z_roundtrip.append(z_rt.cpu())
            
            if self.device.type == 'mps':
                torch.mps.empty_cache()
        
        vae.unload()
        
        z_orig = torch.cat(all_z_orig, dim=0)
        watermarks = torch.cat(all_watermarks, dim=0)
        z_roundtrip = torch.cat(all_z_roundtrip, dim=0)
        
        if cache_path:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            torch.save({
                'z_orig': z_orig,
                'watermarks': watermarks,
                'z_roundtrip': z_roundtrip
            }, cache_path)
            print(f"Roundtrip data cached to {cache_path}")
        
        return z_orig, watermarks, z_roundtrip
    
    # ===== Phase 2b: Fast training on cached roundtrip data =====
    
    def train_step_phase2_fast(self, z_orig, w_batch, z_roundtrip):
        """
        Fast Phase 2 training on precomputed roundtrip latents.
        
        - Encoder is frozen (watermarks already embedded in z_roundtrip)
        - Only train decoder to extract from roundtrip latents
        """
        # Extract watermark from roundtrip latent
        w_pred_rt_l, w_pred_rt_h = self.extract_watermark(z_roundtrip)
        
        # Also extract from fresh embedding (for encoder fine-tuning)
        z_wm = self.embed_watermark(z_orig, w_batch)
        w_pred_l, w_pred_h = self.extract_watermark(z_wm)
        
        # Losses
        loss_roundtrip = F.mse_loss(w_pred_rt_l, w_batch) + F.mse_loss(w_pred_rt_h, w_batch)
        loss_clean = F.mse_loss(w_pred_l, w_batch) + F.mse_loss(w_pred_h, w_batch)
        loss_cons = F.mse_loss(w_pred_rt_l, w_pred_rt_h)
        loss_latent = F.mse_loss(z_wm, z_orig)
        
        total_loss = (
            2.0 * loss_roundtrip +  # Primary: learn to extract from roundtrip
            0.5 * loss_clean +       # Secondary: maintain clean extraction
            0.3 * loss_cons +
            1.0 * loss_latent
        )
        
        bit_acc_rt = self.compute_bit_accuracy(w_batch, w_pred_rt_l, w_pred_rt_h)
        bit_acc_clean = self.compute_bit_accuracy(w_batch, w_pred_l, w_pred_h)
        
        return total_loss, {
            'loss': total_loss.item(),
            'bit_acc_roundtrip': bit_acc_rt,
            'bit_acc_clean': bit_acc_clean
        }
    
    def train_epoch_phase2_fast(self, z_orig, watermarks, z_roundtrip, epoch, batch_size=32):
        """Fast Phase 2 training on precomputed data."""
        self._set_train_mode()
        
        # Create dataset from precomputed data
        dataset = TensorDataset(z_orig, watermarks, z_roundtrip)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        
        epoch_metrics = {'loss': [], 'bit_acc_roundtrip': [], 'bit_acc_clean': []}
        
        pbar = tqdm(dataloader, desc=f"Phase2 Epoch {epoch}")
        for z_batch, w_batch, z_rt_batch in pbar:
            z_batch = z_batch.to(self.device)
            w_batch = w_batch.to(self.device)
            z_rt_batch = z_rt_batch.to(self.device)
            
            self.optimizer.zero_grad()
            loss, metrics = self.train_step_phase2_fast(z_batch, w_batch, z_rt_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.all_params, 1.0)
            self.optimizer.step()
            
            for k, v in metrics.items():
                epoch_metrics[k].append(v)
            
            pbar.set_postfix(
                loss=f"{metrics['loss']:.3f}",
                rt_acc=f"{metrics['bit_acc_roundtrip']:.3f}"
            )
            
            if self.device.type == 'mps':
                torch.mps.empty_cache()
        
        return {k: np.mean(v) for k, v in epoch_metrics.items()}
    
    # ===== Utilities =====
    
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
    
    def save_checkpoint(self, path, epoch, metrics, phase):
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
    
    @torch.no_grad()
    def evaluate_roundtrip(self, latents, n_samples=200):
        """Evaluate with fresh VAE roundtrip."""
        self._set_eval_mode()
        vae = VAEWrapper(device=self.device)
        
        indices = torch.randperm(len(latents))[:n_samples]
        
        bit_acc_clean = []
        bit_acc_rt = []
        
        batch_size = 4
        for i in tqdm(range(0, n_samples, batch_size), desc="Eval roundtrip"):
            idx = indices[i:i+batch_size]
            z_batch = latents[idx].to(self.device)
            w_batch = torch.randn(z_batch.shape[0], self.config.get('w_dim', 32), device=self.device)
            
            # Embed and extract (clean)
            z_wm = self.embed_watermark(z_batch, w_batch)
            w_pred_l, w_pred_h = self.extract_watermark(z_wm)
            bit_acc_clean.append(self.compute_bit_accuracy(w_batch, w_pred_l, w_pred_h))
            
            # VAE roundtrip
            z_rt = vae.roundtrip(z_wm)
            w_pred_rt_l, w_pred_rt_h = self.extract_watermark(z_rt)
            bit_acc_rt.append(self.compute_bit_accuracy(w_batch, w_pred_rt_l, w_pred_rt_h))
            
            if self.device.type == 'mps':
                torch.mps.empty_cache()
        
        vae.unload()
        
        return {
            'bit_acc_clean': np.mean(bit_acc_clean),
            'bit_acc_roundtrip': np.mean(bit_acc_rt)
        }


def main():
    parser = argparse.ArgumentParser(description="Fast VAE-Roundtrip Training")
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--phase1-epochs', type=int, default=50)
    parser.add_argument('--phase2-epochs', type=int, default=50)
    parser.add_argument('--phase2-samples', type=int, default=5000, help='Samples for roundtrip precomputation')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--skip-phase1', action='store_true')
    parser.add_argument('--eval-interval', type=int, default=10)
    parser.add_argument('--recompute-roundtrip', action='store_true', help='Force recompute roundtrip cache')
    args = parser.parse_args()
    
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 
                          'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load precomputed latents
    num_images = config.get('num_images', 20000)
    image_size = config.get('image_size', 256)
    cache_path = f"cache/latents_{num_images}_{image_size}.pt"
    
    print(f"Loading latents from {cache_path}...")
    data = torch.load(cache_path, map_location='cpu', weights_only=False)
    latents = data['latents']
    print(f"Loaded {len(latents)} latents")
    
    # Create trainer
    trainer = FastRoundtripTrainer(config, device)
    
    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"results/fast_staged_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Resume
    start_phase = 1
    start_epoch = 0
    if args.resume:
        start_epoch, start_phase = trainer.load_checkpoint(args.resume)
        print(f"Resuming from Phase {start_phase}, Epoch {start_epoch}")
    
    if args.skip_phase1:
        start_phase = 2
        start_epoch = 0
    
    print("\n" + "="*60)
    print("FAST STAGED TRAINING (with roundtrip precomputation)")
    print("="*60)
    print(f"Phase 1: {args.phase1_epochs} epochs (latent-only)")
    print(f"Phase 2a: Precompute {args.phase2_samples} roundtrip samples (one-time)")
    print(f"Phase 2b: {args.phase2_epochs} epochs (fast, on cached data)")
    print(f"Batch size: {args.batch_size}")
    print(f"Output: {output_dir}")
    print("="*60 + "\n")
    
    # ===== PHASE 1 =====
    if start_phase == 1:
        print("\n" + "="*60)
        print("PHASE 1: Latent-Only Training")
        print("="*60)
        
        dataset = TensorDataset(latents)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(trainer.optimizer, T_max=args.phase1_epochs)
        
        for epoch in range(start_epoch, args.phase1_epochs):
            loss, bit_acc = trainer.train_epoch_phase1(dataloader, epoch + 1)
            scheduler.step()
            print(f"Phase1 Epoch {epoch+1}: Loss={loss:.4f}, Bit Acc={bit_acc:.4f}")
            
            if (epoch + 1) % 10 == 0:
                trainer.save_checkpoint(f"{output_dir}/phase1_epoch{epoch+1}.pt", epoch+1, 
                                        {'loss': loss, 'bit_acc': bit_acc}, phase=1)
        
        trainer.save_checkpoint(f"{output_dir}/phase1_final.pt", args.phase1_epochs,
                               {'loss': loss, 'bit_acc': bit_acc}, phase=1)
        print("Phase 1 complete!")
        start_epoch = 0
    
    # ===== PHASE 2a: Precompute roundtrip =====
    print("\n" + "="*60)
    print("PHASE 2a: Precomputing VAE Roundtrip")
    print("="*60)
    
    roundtrip_cache = f"{output_dir}/roundtrip_cache_{args.phase2_samples}.pt"
    if args.recompute_roundtrip and os.path.exists(roundtrip_cache):
        os.remove(roundtrip_cache)
    
    z_orig, watermarks, z_roundtrip = trainer.precompute_roundtrip_data(
        latents, 
        num_samples=args.phase2_samples,
        batch_size=4,
        cache_path=roundtrip_cache
    )
    print(f"Roundtrip data ready: {len(z_orig)} samples")
    
    # ===== PHASE 2b: Fast training =====
    print("\n" + "="*60)
    print("PHASE 2b: Fast Training on Cached Roundtrip Data")
    print("="*60)
    
    # Lower learning rate for fine-tuning
    for param_group in trainer.optimizer.param_groups:
        param_group['lr'] = config.get('lr', 1e-4) * 0.5
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(trainer.optimizer, T_max=args.phase2_epochs)
    
    best_rt_acc = 0.0
    
    for epoch in range(start_epoch, args.phase2_epochs):
        metrics = trainer.train_epoch_phase2_fast(z_orig, watermarks, z_roundtrip, epoch + 1, 
                                                   batch_size=args.batch_size)
        scheduler.step()
        
        print(f"Phase2 Epoch {epoch+1}: Loss={metrics['loss']:.4f}, "
              f"RT Acc={metrics['bit_acc_roundtrip']:.4f}, Clean Acc={metrics['bit_acc_clean']:.4f}")
        
        if metrics['bit_acc_roundtrip'] > best_rt_acc:
            best_rt_acc = metrics['bit_acc_roundtrip']
            trainer.save_checkpoint(f"{output_dir}/best_roundtrip.pt", epoch+1, metrics, phase=2)
            print(f"  New best roundtrip accuracy: {best_rt_acc:.4f}")
        
        if (epoch + 1) % args.eval_interval == 0:
            eval_metrics = trainer.evaluate_roundtrip(latents, n_samples=100)
            print(f"  Fresh eval: Clean={eval_metrics['bit_acc_clean']:.4f}, "
                  f"Roundtrip={eval_metrics['bit_acc_roundtrip']:.4f}")
        
        if (epoch + 1) % 10 == 0:
            trainer.save_checkpoint(f"{output_dir}/phase2_epoch{epoch+1}.pt", epoch+1, metrics, phase=2)
    
    trainer.save_checkpoint(f"{output_dir}/final.pt", args.phase2_epochs, metrics, phase=2)
    
    # ===== FINAL EVALUATION =====
    print("\n" + "="*60)
    print("FINAL EVALUATION (fresh VAE roundtrip)")
    print("="*60)
    
    final_metrics = trainer.evaluate_roundtrip(latents, n_samples=500)
    print(f"Clean Bit Accuracy:     {final_metrics['bit_acc_clean']:.4f}")
    print(f"Roundtrip Bit Accuracy: {final_metrics['bit_acc_roundtrip']:.4f}")
    
    print(f"\nResults saved to: {output_dir}")
    print("Done!")


if __name__ == "__main__":
    main()

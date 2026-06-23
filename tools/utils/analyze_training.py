#!/usr/bin/env python3
"""Analyze training time and parameter counts."""
import sys
import os
import time
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from latent_watermarking.models.watermark_encoder import WatermarkEncoder
from latent_watermarking.models.watermark_decoder import WatermarkDecoder

def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

# Create watermark models
encoder = WatermarkEncoder(input_channels=4, watermark_dim=64, hidden_dim=64)
decoder = WatermarkDecoder(input_channels=4, watermark_dim=64, hidden_dim=64)

enc_total, enc_train = count_params(encoder)
dec_total, dec_train = count_params(decoder)

print("=" * 60)
print("WATERMARK MODEL PARAMETERS (what gets trained)")
print("=" * 60)
print(f"Encoder: {enc_total:,} total ({enc_train:,} trainable)")
print(f"Decoder: {dec_total:,} total ({dec_train:,} trainable)")
print(f"TOTAL:   {enc_total + dec_total:,} parameters")
print()

# VAE parameters (for reference - frozen, not trained)
print("=" * 60)
print("VAE PARAMETERS (frozen, NOT trained)")
print("=" * 60)
try:
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(
        'runwayml/stable-diffusion-v1-5',
        subfolder='vae',
        torch_dtype=torch.float32
    )
    vae_total, _ = count_params(vae)
    print(f"VAE: {vae_total:,} parameters (frozen)")
    
    # Benchmark VAE operations
    print()
    print("=" * 60)
    print("VAE OPERATION TIMING (on MPS)")
    print("=" * 60)
    
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    vae = vae.to(device)
    vae.eval()
    
    # Test with batch of 8 at 256x256
    batch_size = 8
    with torch.no_grad():
        # Create dummy latent
        z = torch.randn(batch_size, 4, 32, 32).to(device)
        
        # Warmup
        for _ in range(3):
            img = vae.decode(z / 0.18215).sample
            z2 = vae.encode(img).latent_dist.mean * 0.18215
        
        if device == 'mps':
            torch.mps.synchronize()
        
        # Benchmark decode
        n_runs = 10
        start = time.time()
        for _ in range(n_runs):
            img = vae.decode(z / 0.18215).sample
            if device == 'mps':
                torch.mps.synchronize()
        decode_time = (time.time() - start) / n_runs
        
        # Benchmark encode  
        img = vae.decode(z / 0.18215).sample
        start = time.time()
        for _ in range(n_runs):
            z2 = vae.encode(img).latent_dist.mean * 0.18215
            if device == 'mps':
                torch.mps.synchronize()
        encode_time = (time.time() - start) / n_runs
        
        # Benchmark roundtrip
        start = time.time()
        for _ in range(n_runs):
            img = vae.decode(z / 0.18215).sample
            z2 = vae.encode(img).latent_dist.mean * 0.18215
            if device == 'mps':
                torch.mps.synchronize()
        roundtrip_time = (time.time() - start) / n_runs
        
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Decode (latent→image):  {decode_time*1000:.1f} ms/batch ({decode_time*1000/batch_size:.2f} ms/image)")
    print(f"Encode (image→latent):  {encode_time*1000:.1f} ms/batch ({encode_time*1000/batch_size:.2f} ms/image)")
    print(f"Roundtrip (decode+encode): {roundtrip_time*1000:.1f} ms/batch ({roundtrip_time*1000/batch_size:.2f} ms/image)")
    
    del vae
    if device == 'mps':
        torch.mps.empty_cache()
        
except Exception as e:
    print(f"VAE analysis skipped: {e}")

# Training time estimates
print()
print("=" * 60)
print("TRAINING TIME ESTIMATES")
print("=" * 60)

# Current efficient training (latent-only)
# From logs: ~50 it/s with batch_size=32
current_speed = 50  # iterations per second
batch_size_train = 32
num_images = 20000
epochs = 100

iterations_per_epoch = num_images // batch_size_train
total_iterations = iterations_per_epoch * epochs

current_time_per_epoch = iterations_per_epoch / current_speed
current_total_time = total_iterations / current_speed

print(f"Current efficient training (latent-only, no VAE):")
print(f"  Speed: ~{current_speed} it/s")
print(f"  Time per epoch: {current_time_per_epoch:.1f} seconds ({current_time_per_epoch/60:.1f} min)")
print(f"  Total for {epochs} epochs: {current_total_time:.0f} seconds ({current_total_time/60:.1f} min)")

# VAE roundtrip training estimate
# With VAE decode+encode in loop, speed will be dominated by VAE ops
# Estimate: ~0.15s per batch for roundtrip with backprop
vae_batch_time = 0.15  # seconds per batch (decode + encode, with gradients through encoder)
vae_iterations_per_sec = 1 / vae_batch_time

vae_time_per_epoch = iterations_per_epoch * vae_batch_time
vae_total_time = total_iterations * vae_batch_time

print()
print(f"VAE roundtrip training (with decode→encode in loop):")
print(f"  Estimated speed: ~{vae_iterations_per_sec:.1f} it/s")
print(f"  Time per epoch: {vae_time_per_epoch:.1f} seconds ({vae_time_per_epoch/60:.1f} min)")
print(f"  Total for {epochs} epochs: {vae_total_time:.0f} seconds ({vae_total_time/60:.1f} min, {vae_total_time/3600:.1f} hours)")

slowdown = vae_batch_time / (1/current_speed)
print(f"  Slowdown factor: ~{slowdown:.0f}x compared to efficient training")

# Staged training option
print()
print("=" * 60)
print("STAGED TRAINING OPTION")
print("=" * 60)
print("""
Option 1: Full roundtrip from start
  - All epochs include VAE roundtrip
  - Time: ~{:.1f} hours for {} epochs
  - Pro: Most robust training
  - Con: Slowest

Option 2: Staged training (RECOMMENDED)
  - Phase 1: 50 epochs latent-only (~{:.0f} min) - learn basic embedding
  - Phase 2: 50 epochs with VAE roundtrip (~{:.1f} hours) - learn robustness
  - Total: ~{:.1f} hours
  - Pro: Faster, still robust
  - Con: May need hyperparameter tuning between phases

Option 3: Periodic roundtrip
  - Every Nth epoch use VAE roundtrip (e.g., every 5th)
  - Time: ~{:.1f} hours (20% VAE epochs)
  - Pro: Balanced speed/robustness
  - Con: May not be as robust
""".format(
    vae_total_time/3600, epochs,
    current_total_time/60 / 2,
    vae_total_time/3600 / 2,
    current_total_time/60/2/60 + vae_total_time/3600/2,
    current_total_time/60/60 * 0.8 + vae_total_time/3600 * 0.2
))

# Smaller batch size might be needed
print("=" * 60)
print("MEMORY CONSIDERATIONS")
print("=" * 60)
print("""
VAE roundtrip requires more memory because:
1. Need to keep latent z in graph for gradient flow through encoder
2. VAE decode creates 256x256x3 image (vs 32x32x4 latent)
3. VAE encode needs to process full image

Recommendations for MPS (limited memory):
- Reduce batch size from 32 to 8-16
- Use gradient checkpointing if available
- Consider gradient accumulation (virtual batch of 32)
""")

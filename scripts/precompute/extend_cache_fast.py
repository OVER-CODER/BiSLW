#!/usr/bin/env python3
"""
Fast cache extension with optimizations:
- Larger batch size
- Float16 VAE
- Better memory management
"""

import os
import sys
import torch
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from latent_watermarking.models.latent_split import LatentSplitter
from latent_watermarking.models.recombination import LatentRecombiner
from latent_watermarking.models.watermark_encoder import WatermarkEncoder
from latent_watermarking.models.watermark_decoder import WatermarkDecoder


def main():
    import argparse
    import yaml
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--existing-cache', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output-cache', type=str, required=True)
    parser.add_argument('--target-samples', type=int, default=20000)
    parser.add_argument('--batch-size', type=int, default=8, help='Larger = faster but more memory')
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    args = parser.parse_args()
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Batch size: {args.batch_size}")
    
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Load existing cache
    print(f"Loading existing cache: {args.existing_cache}")
    existing = torch.load(args.existing_cache, map_location='cpu', weights_only=False)
    existing_z_orig = existing['z_orig']
    existing_watermarks = existing['watermarks']
    existing_z_roundtrip = existing['z_roundtrip']
    existing_count = len(existing_z_orig)
    print(f"Existing samples: {existing_count}")
    
    needed = args.target_samples - existing_count
    if needed <= 0:
        print("Nothing to do.")
        return
    
    print(f"Need to compute {needed} more samples")
    
    # Load VAE with float16 for speed
    print("Loading VAE (float16 for speed)...")
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(
        'runwayml/stable-diffusion-v1-5',
        subfolder='vae',
        torch_dtype=torch.float16  # Float16 for faster compute
    ).to(device)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False
    
    scaling_factor = 0.18215
    
    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    
    w_dim = config.get('w_dim', 32)
    splitter = LatentSplitter(mode=config.get('latent_split', 'dct')).to(device)
    recombiner = LatentRecombiner(mode=config.get('latent_split', 'dct')).to(device)
    encoder_l = WatermarkEncoder(watermark_dim=w_dim).to(device)
    encoder_h = WatermarkEncoder(watermark_dim=w_dim).to(device)
    
    encoder_l.load_state_dict(checkpoint['encoder_l'])
    encoder_h.load_state_dict(checkpoint['encoder_h'])
    encoder_l.eval()
    encoder_h.eval()
    
    alpha_l = checkpoint.get('alpha_l', 0.02)
    alpha_h = checkpoint.get('alpha_h', 0.01)
    
    # Load latents
    num_images = config.get('num_images', 20000)
    image_size = config.get('image_size', 256)
    latent_path = f"cache/latents_{num_images}_{image_size}.pt"
    
    print(f"Loading latents: {latent_path}")
    data = torch.load(latent_path, map_location='cpu', weights_only=False)
    latents = data['latents']
    
    new_indices = torch.arange(existing_count, min(existing_count + needed, len(latents)))
    actual_needed = len(new_indices)
    
    print(f"Computing {actual_needed} new roundtrip samples...")
    
    new_z_orig = []
    new_watermarks = []
    new_z_roundtrip = []
    
    batch_size = args.batch_size
    num_batches = (actual_needed + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for i in tqdm(range(0, actual_needed, batch_size), desc="Computing roundtrip", total=num_batches):
            idx = new_indices[i:i+batch_size]
            z_batch = latents[idx].to(device)
            B = z_batch.shape[0]
            
            # Generate watermarks
            w_batch = torch.randn(B, w_dim, device=device)
            
            # Embed watermark (float32)
            z_l, z_h = splitter(z_batch)
            z_l_wm = encoder_l(z_l, w_batch, alpha=alpha_l)
            z_h_wm = encoder_h(z_h, w_batch, alpha=alpha_h)
            z_wm = recombiner(z_l_wm, z_h_wm)
            
            # VAE roundtrip (float16 for speed)
            z_wm_f16 = z_wm.half()
            z_scaled = z_wm_f16 / scaling_factor
            img = vae.decode(z_scaled).sample
            z_rt = vae.encode(img).latent_dist.mean * scaling_factor
            z_rt = z_rt.float()  # Back to float32 for storage
            
            new_z_orig.append(z_batch.cpu())
            new_watermarks.append(w_batch.cpu())
            new_z_roundtrip.append(z_rt.cpu())
            
            # Memory cleanup every 50 batches
            if (i // batch_size) % 50 == 0 and device.type == 'mps':
                torch.mps.empty_cache()
    
    # Unload VAE
    del vae
    if device.type == 'mps':
        torch.mps.empty_cache()
    print("VAE unloaded")
    
    # Concatenate
    new_z_orig = torch.cat(new_z_orig, dim=0)
    new_watermarks = torch.cat(new_watermarks, dim=0)
    new_z_roundtrip = torch.cat(new_z_roundtrip, dim=0)
    
    # Merge
    merged_z_orig = torch.cat([existing_z_orig, new_z_orig], dim=0)
    merged_watermarks = torch.cat([existing_watermarks, new_watermarks], dim=0)
    merged_z_roundtrip = torch.cat([existing_z_roundtrip, new_z_roundtrip], dim=0)
    
    print(f"Merged: {len(merged_z_orig)} total samples")
    
    # Save
    os.makedirs(os.path.dirname(args.output_cache) if os.path.dirname(args.output_cache) else '.', exist_ok=True)
    torch.save({
        'z_orig': merged_z_orig,
        'watermarks': merged_watermarks,
        'z_roundtrip': merged_z_roundtrip
    }, args.output_cache)
    
    print(f"Saved to: {args.output_cache}")
    print("Done!")


if __name__ == "__main__":
    main()

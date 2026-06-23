#!/usr/bin/env python3
"""
Fast precomputation of attacked latents.
Uses batch processing and float16 for speed.
~15-20 min for 5000 samples with 4 attacks.
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import io
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from latent_watermarking.models.latent_split import LatentSplitter
from latent_watermarking.models.recombination import LatentRecombiner
from latent_watermarking.models.watermark_encoder import WatermarkEncoder


def jpeg_attack_batch(images, quality=70):
    """Simulated JPEG - downsample + upsample."""
    B, C, H, W = images.shape
    scale = max(0.3, quality / 100)
    h_s, w_s = max(8, int(H * scale)), max(8, int(W * scale))
    down = F.interpolate(images, size=(h_s, w_s), mode='bilinear', align_corners=False)
    up = F.interpolate(down, size=(H, W), mode='bilinear', align_corners=False)
    blend = quality / 100
    return blend * images + (1 - blend) * up


def noise_attack_batch(images, sigma=0.05):
    return (images + torch.randn_like(images) * sigma).clamp(-1, 1)


def blur_attack_batch(images, kernel_size=5):
    if kernel_size % 2 == 0:
        kernel_size += 1
    sigma = kernel_size / 3
    device = images.device
    x = torch.arange(kernel_size, dtype=images.dtype, device=device) - kernel_size // 2
    k1d = torch.exp(-x**2 / (2 * sigma**2))
    k1d = k1d / k1d.sum()
    k2d = k1d.unsqueeze(1) @ k1d.unsqueeze(0)
    kernel = k2d.unsqueeze(0).unsqueeze(0).expand(3, 1, -1, -1)
    return F.conv2d(images, kernel, padding=kernel_size // 2, groups=3)


ATTACKS = [
    ('clean', None),
    ('jpeg_70', lambda x: jpeg_attack_batch(x, 70)),
    ('jpeg_50', lambda x: jpeg_attack_batch(x, 50)),
    ('noise_0.05', lambda x: noise_attack_batch(x, 0.05)),
    ('blur_5', lambda x: blur_attack_batch(x, 5)),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, help='Model checkpoint')
    parser.add_argument('--cache', default='cache/roundtrip_20000.pt')
    parser.add_argument('--output', default='cache/attacked_5000.pt')
    parser.add_argument('--samples', type=int, default=5000)
    parser.add_argument('--batch-size', type=int, default=8)
    args = parser.parse_args()
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load VAE in float16 for speed
    print("Loading VAE (float16)...")
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(
        'runwayml/stable-diffusion-v1-5',
        subfolder='vae',
        torch_dtype=torch.float16
    ).to(device)
    vae.eval()
    scaling = 0.18215
    
    # Load watermark model
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    config = ckpt.get('config', {})
    w_dim = config.get('w_dim', 32)
    
    splitter = LatentSplitter(mode='dct').to(device)
    recombiner = LatentRecombiner(mode='dct').to(device)
    encoder_l = WatermarkEncoder(watermark_dim=w_dim).to(device)
    encoder_h = WatermarkEncoder(watermark_dim=w_dim).to(device)
    
    encoder_l.load_state_dict(ckpt['encoder_l'])
    encoder_h.load_state_dict(ckpt['encoder_h'])
    encoder_l.eval()
    encoder_h.eval()
    
    alpha_l = ckpt.get('alpha_l', 0.02)
    alpha_h = ckpt.get('alpha_h', 0.01)
    
    @torch.no_grad()
    def embed(z, w):
        z_l, z_h = splitter(z)
        z_l_wm = encoder_l(z_l, w, alpha=alpha_l)
        z_h_wm = encoder_h(z_h, w, alpha=alpha_h)
        return recombiner(z_l_wm, z_h_wm)
    
    # Load existing cache
    print(f"Loading cache: {args.cache}")
    cache = torch.load(args.cache, map_location='cpu', weights_only=False)
    z_orig = cache['z_orig'][:args.samples]
    watermarks = cache['watermarks'][:args.samples]
    
    n = len(z_orig)
    print(f"Processing {n} samples with {len(ATTACKS)} attack types")
    print(f"Attacks: {[name for name, _ in ATTACKS]}")
    
    # Initialize result storage
    result = {
        'z_orig': z_orig,
        'watermarks': watermarks,
    }
    for name, _ in ATTACKS:
        result[f'z_{name}'] = []
    
    # Process in batches
    n_batches = (n + args.batch_size - 1) // args.batch_size
    
    for batch_idx in tqdm(range(n_batches), desc="Precomputing"):
        start = batch_idx * args.batch_size
        end = min(start + args.batch_size, n)
        
        z_batch = z_orig[start:end].to(device)
        w_batch = watermarks[start:end].to(device)
        
        # Embed watermark
        z_wm = embed(z_batch, w_batch)
        
        # Decode to image (float16 for speed)
        img_wm = vae.decode(z_wm.half() / scaling).sample
        
        # Apply each attack and re-encode
        for name, attack_fn in ATTACKS:
            if attack_fn is not None:
                img_att = attack_fn(img_wm)
            else:
                img_att = img_wm
            
            # Encode back
            z_att = vae.encode(img_att).latent_dist.mean * scaling
            result[f'z_{name}'].append(z_att.float().cpu())
        
        if device.type == 'mps':
            torch.mps.empty_cache()
    
    # Concatenate results
    for name, _ in ATTACKS:
        result[f'z_{name}'] = torch.cat(result[f'z_{name}'], dim=0)
        print(f"  {name}: {result[f'z_{name}'].shape}")
    
    # Save
    torch.save(result, args.output)
    size_mb = os.path.getsize(args.output) / 1e6
    print(f"\nSaved: {args.output} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Precompute attacked latents for fast attack-aware training.
Only needs to run once, then training is fast.
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
from latent_watermarking.models.watermark_decoder import WatermarkDecoder


def jpeg_attack(img, quality=70):
    """JPEG compression on single image tensor."""
    img_np = ((img.squeeze(0).permute(1, 2, 0).cpu() + 1) / 2 * 255).clamp(0, 255).numpy().astype(np.uint8)
    pil_img = Image.fromarray(img_np, mode='RGB')
    buffer = io.BytesIO()
    pil_img.save(buffer, format='JPEG', quality=quality)
    buffer.seek(0)
    compressed = Image.open(buffer).convert('RGB')
    img_out = torch.from_numpy(np.array(compressed)).float() / 255.0
    return (img_out.permute(2, 0, 1) * 2 - 1).unsqueeze(0)


def gaussian_noise_attack(img, sigma=0.05):
    return (img + torch.randn_like(img) * sigma).clamp(-1, 1)


def gaussian_blur_attack(img, kernel_size=5):
    if kernel_size % 2 == 0:
        kernel_size += 1
    sigma = kernel_size / 3
    x = torch.arange(kernel_size, dtype=img.dtype) - kernel_size // 2
    kernel_1d = torch.exp(-x**2 / (2 * sigma**2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = kernel_1d.unsqueeze(1) * kernel_1d.unsqueeze(0)
    kernel = kernel_2d.unsqueeze(0).unsqueeze(0).expand(3, 1, -1, -1)
    pad = kernel_size // 2
    return F.conv2d(img, kernel, padding=pad, groups=3)


ATTACKS = {
    'jpeg_70': lambda x: jpeg_attack(x, 70),
    'jpeg_50': lambda x: jpeg_attack(x, 50),
    'noise_0.05': lambda x: gaussian_noise_attack(x, 0.05),
    'blur_5': lambda x: gaussian_blur_attack(x, 5),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--cache', type=str, default='cache/roundtrip_20000.pt')
    parser.add_argument('--output', type=str, default='cache/attacked_10000.pt')
    parser.add_argument('--samples', type=int, default=10000)
    parser.add_argument('--batch-size', type=int, default=4)
    args = parser.parse_args()
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load VAE
    print("Loading VAE...")
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
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
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
    
    def embed(z, w):
        z_low, z_high = splitter(z)
        z_low_wm = encoder_l(z_low, w, alpha=alpha_l)
        z_high_wm = encoder_h(z_high, w, alpha=alpha_h)
        return recombiner(z_low_wm, z_high_wm)
    
    # Load existing cache
    print(f"Loading cache: {args.cache}")
    cache = torch.load(args.cache, map_location='cpu', weights_only=False)
    z_orig = cache['z_orig'][:args.samples]
    watermarks = cache['watermarks'][:args.samples]
    z_roundtrip = cache['z_roundtrip'][:args.samples]  # Clean roundtrip
    
    n = len(z_orig)
    print(f"Processing {n} samples")
    
    # Precompute attacked latents
    attacked_latents = {name: [] for name in ATTACKS}
    
    for i in tqdm(range(n), desc="Precomputing attacks"):
        z = z_orig[i:i+1].to(device)
        w = watermarks[i:i+1].to(device)
        
        # Embed watermark
        with torch.no_grad():
            z_wm = embed(z, w)
            
            # Decode to image
            img_wm = vae.decode(z_wm.half() / scaling).sample.float()
            
            # Apply each attack and re-encode
            for name, attack_fn in ATTACKS.items():
                img_att = attack_fn(img_wm.cpu()).to(device)
                z_att = vae.encode(img_att.half()).latent_dist.mean.float() * scaling
                attacked_latents[name].append(z_att.cpu())
        
        if device.type == 'mps' and i % 100 == 0:
            torch.mps.empty_cache()
    
    # Stack results
    result = {
        'z_orig': z_orig,
        'watermarks': watermarks,
        'z_clean': z_roundtrip,  # Clean roundtrip from cache
    }
    for name in ATTACKS:
        result[f'z_{name}'] = torch.cat(attacked_latents[name], dim=0)
        print(f"  {name}: {result[f'z_{name}'].shape}")
    
    # Save
    torch.save(result, args.output)
    print(f"\nSaved to {args.output}")
    print(f"Size: {os.path.getsize(args.output) / 1e9:.2f} GB")


if __name__ == "__main__":
    main()

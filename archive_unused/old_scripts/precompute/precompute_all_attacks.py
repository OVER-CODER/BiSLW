#!/usr/bin/env python3
"""
Precompute ALL attacked latents for fast training.

This runs once and saves attacked versions of each latent.
Then training just needs latent-space operations (100x faster).

Precomputes for each sample:
- Clean VAE roundtrip
- Center Crop 0.1
- Random Crop 0.1  
- Resize 0.7
- Rotation 15
- Blur
- Contrast 2.0
- Brightness 2.0
- JPEG 70
- Combined
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

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.dirname(project_root))

from latent_watermarking.models.latent_split import LatentSplitter
from latent_watermarking.models.recombination import LatentRecombiner
from latent_watermarking.models.watermark_encoder import WatermarkEncoder


# ============================================================
# ATTACK FUNCTIONS (same as training script)
# ============================================================

def real_jpeg_attack(images, quality=70):
    """Real JPEG compression using PIL."""
    device = images.device
    results = []
    for i in range(images.shape[0]):
        img = images[i].cpu()
        img_np = ((img.permute(1, 2, 0) + 1) / 2 * 255).clamp(0, 255).numpy().astype(np.uint8)
        pil_img = Image.fromarray(img_np)
        
        buffer = io.BytesIO()
        pil_img.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        compressed = Image.open(buffer).convert('RGB')
        
        img_out = torch.from_numpy(np.array(compressed)).float() / 255.0
        img_out = img_out.permute(2, 0, 1) * 2 - 1
        results.append(img_out)
    
    return torch.stack(results).to(device)


def center_crop_attack(images, crop_ratio=0.1):
    B, C, H, W = images.shape
    crop_h = int(H * crop_ratio)
    crop_w = int(W * crop_ratio)
    cropped = images[:, :, crop_h:H-crop_h, crop_w:W-crop_w]
    return F.interpolate(cropped, size=(H, W), mode='bilinear', align_corners=False)


def random_crop_attack(images, crop_ratio=0.1):
    B, C, H, W = images.shape
    crop_h = int(H * crop_ratio * 2)
    crop_w = int(W * crop_ratio * 2)
    top = torch.randint(0, crop_h + 1, (1,)).item() if crop_h > 0 else 0
    left = torch.randint(0, crop_w + 1, (1,)).item() if crop_w > 0 else 0
    new_h, new_w = H - crop_h, W - crop_w
    cropped = images[:, :, top:top+new_h, left:left+new_w]
    return F.interpolate(cropped, size=(H, W), mode='bilinear', align_corners=False)


def resize_attack(images, scale=0.7):
    B, C, H, W = images.shape
    h_small = max(1, int(H * scale))
    w_small = max(1, int(W * scale))
    down = F.interpolate(images, size=(h_small, w_small), mode='bilinear', align_corners=False)
    return F.interpolate(down, size=(H, W), mode='bilinear', align_corners=False)


def rotation_attack(images, angle=15.0):
    B, C, H, W = images.shape
    angle_rad = angle * np.pi / 180
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    theta = torch.tensor([[cos_a, -sin_a, 0], [sin_a, cos_a, 0]], 
                         dtype=images.dtype, device=images.device).unsqueeze(0).expand(B, -1, -1)
    grid = F.affine_grid(theta, images.size(), align_corners=False)
    return F.grid_sample(images, grid, mode='bilinear', padding_mode='reflection', align_corners=False)


def gaussian_blur_attack(images, kernel_size=5):
    if kernel_size % 2 == 0:
        kernel_size += 1
    sigma = kernel_size / 3
    x = torch.arange(kernel_size, device=images.device, dtype=images.dtype) - kernel_size // 2
    kernel_1d = torch.exp(-x**2 / (2 * sigma**2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = kernel_1d.unsqueeze(1) * kernel_1d.unsqueeze(0)
    kernel = kernel_2d.unsqueeze(0).unsqueeze(0).expand(3, 1, -1, -1)
    return F.conv2d(images, kernel, padding=kernel_size // 2, groups=3)


def contrast_attack(images, factor=2.0):
    img_01 = (images + 1) / 2
    mean = img_01.mean(dim=(2, 3), keepdim=True)
    adjusted = (img_01 - mean) * factor + mean
    return (adjusted.clamp(0, 1) * 2 - 1)


def brightness_attack(images, factor=2.0):
    img_01 = (images + 1) / 2
    shift = (factor - 1) * 0.3
    return ((img_01 + shift).clamp(0, 1) * 2 - 1)


def combined_attack(images):
    result = images
    result = resize_attack(result, 0.85)
    result = gaussian_blur_attack(result, 3)
    result = contrast_attack(result, 1.2)
    return result


# All attacks to precompute
ATTACKS = [
    ('clean', None),
    ('center_crop', lambda x: center_crop_attack(x, 0.1)),
    ('random_crop', lambda x: random_crop_attack(x, 0.1)),
    ('resize', lambda x: resize_attack(x, 0.7)),
    ('rotation', lambda x: rotation_attack(x, 15.0)),
    ('blur', lambda x: gaussian_blur_attack(x, 5)),
    ('contrast', lambda x: contrast_attack(x, 2.0)),
    ('brightness', lambda x: brightness_attack(x, 2.0)),
    ('jpeg', lambda x: real_jpeg_attack(x, 70)),
    ('combined', combined_attack),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint for embedding')
    parser.add_argument('--cache', type=str, default='cache/roundtrip_20000.pt', help='Input cache')
    parser.add_argument('--output', type=str, default='cache/attacked_latents.pt', help='Output cache')
    parser.add_argument('--samples', type=int, default=5000, help='Number of samples')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size for VAE')
    parser.add_argument('--start-idx', type=int, default=0, help='Start index in source data')
    args = parser.parse_args()
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load VAE
    print("Loading VAE...")
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(
        'runwayml/stable-diffusion-v1-5',
        subfolder='vae',
        torch_dtype=torch.float32
    ).to(device)
    vae.eval()
    scaling = 0.18215
    
    # Load checkpoint for watermark embedding
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
        z_l, z_h = splitter(z)
        z_l_wm = encoder_l(z_l, w, alpha=alpha_l)
        z_h_wm = encoder_h(z_h, w, alpha=alpha_h)
        return recombiner(z_l_wm, z_h_wm)
    
    # Load source data
    print(f"Loading: {args.cache}")
    start = args.start_idx
    end = args.start_idx + args.samples
    
    if os.path.exists(args.cache):
        cache = torch.load(args.cache, map_location='cpu', weights_only=False)
        if 'z_orig' in cache:
            z_all = cache['z_orig'][start:end]
            w_all = cache['watermarks'][start:end]
        else:
            z_all = cache['latents'][start:end]
            w_all = torch.randn(len(z_all), w_dim)
    else:
        cache = torch.load('cache/latents_20000_256.pt', map_location='cpu', weights_only=False)
        z_all = cache['latents'][start:end]
        w_all = torch.randn(len(z_all), w_dim)
    
    # Ensure binary watermarks
    w_all = (w_all > 0).float() * 2 - 1
    
    n = len(z_all)
    print(f"Processing {n} samples (indices {start} to {end})")
    print(f"Attacks: {[name for name, _ in ATTACKS]}")
    
    # Initialize storage for attacked latents
    attacked_z = {name: [] for name, _ in ATTACKS}
    
    # Process in batches
    batch_size = args.batch_size
    
    for i in tqdm(range(0, n, batch_size), desc="Precomputing"):
        end_idx = min(i + batch_size, n)
        z_batch = z_all[i:end_idx].to(device)
        w_batch = w_all[i:end_idx].to(device)
        
        with torch.no_grad():
            # Embed watermark
            z_wm = embed(z_batch, w_batch)
            
            # Decode to image
            img_wm = vae.decode(z_wm / scaling).sample
            
            # Apply each attack and re-encode
            for name, attack_fn in ATTACKS:
                if attack_fn is not None:
                    img_att = attack_fn(img_wm)
                else:
                    img_att = img_wm
                
                # Re-encode
                z_att = vae.encode(img_att).latent_dist.mean * scaling
                attacked_z[name].append(z_att.cpu())
        
        if device.type == 'mps' and i % 50 == 0:
            torch.mps.empty_cache()
    
    # Concatenate results
    result = {
        'z_orig': z_all,
        'watermarks': w_all,
    }
    
    for name, _ in ATTACKS:
        result[f'z_{name}'] = torch.cat(attacked_z[name], dim=0)
        print(f"  {name}: {result[f'z_{name}'].shape}")
    
    # Save
    torch.save(result, args.output)
    file_size = os.path.getsize(args.output) / 1e9
    print(f"\nSaved to: {args.output}")
    print(f"File size: {file_size:.2f} GB")
    print("\nNow run fast training with:")
    print(f"  python3 scripts/training/train_attack_fast_cached.py --cache {args.output}")


if __name__ == "__main__":
    main()

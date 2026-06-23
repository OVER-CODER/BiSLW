#!/usr/bin/env python3
"""Quick evaluation of fine-tuned model."""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.dirname(project_root))

from latent_watermarking.models.latent_split import LatentSplitter
from latent_watermarking.models.recombination import LatentRecombiner
from latent_watermarking.models.watermark_encoder import WatermarkEncoder
from latent_watermarking.models.watermark_decoder import WatermarkDecoder
from latent_watermarking.models.vae_wrapper import VAEWrapper


def center_crop(images, ratio=0.1):
    B, C, H, W = images.shape
    crop = int(min(H, W) * ratio)
    return images[:, :, crop:H-crop, crop:W-crop]

def random_crop(images, ratio=0.1):
    B, C, H, W = images.shape
    crop = int(min(H, W) * ratio)
    top = torch.randint(0, crop*2, (1,)).item()
    left = torch.randint(0, crop*2, (1,)).item()
    return images[:, :, top:H-crop*2+top, left:W-crop*2+left]

def resize_attack(images, scale=0.7):
    B, C, H, W = images.shape
    small = F.interpolate(images, scale_factor=scale, mode='bilinear', align_corners=False)
    return F.interpolate(small, size=(H, W), mode='bilinear', align_corners=False)

def rotation_attack(images, angle=15):
    angle_rad = torch.tensor(angle * np.pi / 180)
    cos_a, sin_a = torch.cos(angle_rad), torch.sin(angle_rad)
    theta = torch.tensor([[cos_a, -sin_a, 0], [sin_a, cos_a, 0]], dtype=images.dtype).unsqueeze(0)
    theta = theta.expand(images.shape[0], -1, -1).to(images.device)
    grid = F.affine_grid(theta, images.shape, align_corners=False)
    return F.grid_sample(images, grid, align_corners=False, padding_mode='reflection')

def blur_attack(images, kernel_size=5):
    padding = kernel_size // 2
    kernel = torch.ones(1, 1, kernel_size, kernel_size, device=images.device) / (kernel_size ** 2)
    blurred = [F.conv2d(images[:, c:c+1], kernel, padding=padding) for c in range(images.shape[1])]
    return torch.cat(blurred, dim=1)

def contrast_attack(images, factor=2.0):
    mean = images.mean(dim=[2, 3], keepdim=True)
    return ((images - mean) * factor + mean).clamp(-1, 1)

def brightness_attack(images, factor=2.0):
    return (images * factor).clamp(-1, 1)

def jpeg_sim(images, quality=70):
    noise_scale = (100 - quality) / 500
    noise = torch.randn_like(images) * noise_scale
    quantization = 0.02 * (100 - quality) / 30
    quantized = (images / quantization).round() * quantization
    return (0.7 * quantized + 0.3 * images + noise).clamp(-1, 1)

def combined_attack(images):
    x = jpeg_sim(images, quality=80)
    x = resize_attack(x, scale=0.85)
    x = blur_attack(x, kernel_size=3)
    return x


ATTACKS = {
    'None': lambda x: x,
    'C.Crop 0.1': lambda x: center_crop(x, 0.1),
    'R.Crop 0.1': lambda x: random_crop(x, 0.1),
    'Resize 0.7': lambda x: resize_attack(x, 0.7),
    'Rot. 15': lambda x: rotation_attack(x, 15),
    'Blur': lambda x: blur_attack(x, 5),
    'Contr. 2.0': lambda x: contrast_attack(x, 2.0),
    'Bright. 2.0': lambda x: brightness_attack(x, 2.0),
    'JPEG 70': lambda x: jpeg_sim(x, 70),
    'Comb.': combined_attack,
}


def main():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f'Device: {device}')
    
    # Find model
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    
    # Find latest finetune model
    results_dir = os.path.join(project_root, 'results')
    finetune_dirs = [d for d in os.listdir(results_dir) if d.startswith('finetune_efficient')]
    if not finetune_dirs:
        print("No fine-tuned model found!")
        return
    
    latest = sorted(finetune_dirs)[-1]
    ckpt_path = os.path.join(results_dir, latest, 'best.pt')
    print(f'Loading: {ckpt_path}')
    
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    print(f"Epoch: {ckpt['epoch']}")
    print(f"Training metrics: {ckpt['metrics']}")
    
    w_dim = ckpt.get('config', {}).get('w_dim', 32)
    alpha_l = ckpt.get('alpha_l', 0.02)
    alpha_h = ckpt.get('alpha_h', 0.01)
    print(f'alpha_l={alpha_l}, alpha_h={alpha_h}')
    
    # Load models
    splitter = LatentSplitter(mode='dct').to(device)
    recombiner = LatentRecombiner(mode='dct').to(device)
    encoder_l = WatermarkEncoder(watermark_dim=w_dim).to(device)
    encoder_h = WatermarkEncoder(watermark_dim=w_dim).to(device)
    decoder_l = WatermarkDecoder(watermark_dim=w_dim).to(device)
    decoder_h = WatermarkDecoder(watermark_dim=w_dim).to(device)
    
    encoder_l.load_state_dict(ckpt['encoder_l'])
    encoder_h.load_state_dict(ckpt['encoder_h'])
    decoder_l.load_state_dict(ckpt['decoder_l'])
    decoder_h.load_state_dict(ckpt['decoder_h'])
    
    encoder_l.eval()
    encoder_h.eval()
    decoder_l.eval()
    decoder_h.eval()
    
    # Load VAE
    print('Loading VAE...')
    vae = VAEWrapper().to(device)
    
    # Load latents
    latent_path = os.path.join(project_root, 'cache/latents_20000_256.pt')
    latent_data = torch.load(latent_path, map_location='cpu', weights_only=False)
    if isinstance(latent_data, dict):
        latents = latent_data.get('latents', latent_data.get('z_orig', list(latent_data.values())[0]))
    else:
        latents = latent_data
    print(f'Loaded {len(latents)} latents')
    
    # Evaluate
    n_samples = 100
    results = {k: [] for k in ATTACKS.keys()}
    
    print(f'\nEvaluating finetune_efficient on {n_samples} samples...')
    for i in tqdm(range(n_samples)):
        z = latents[i:i+1].to(device)
        w = torch.randn(1, w_dim, device=device)
        w = (w > 0).float() * 2 - 1
        
        # Embed
        z_l, z_h = splitter(z)
        z_l_wm = encoder_l(z_l, w, alpha=alpha_l)
        z_h_wm = encoder_h(z_h, w, alpha=alpha_h)
        z_wm = recombiner(z_l_wm, z_h_wm)
        
        # Decode to image
        with torch.no_grad():
            img_wm = vae.decode(z_wm)
        
        # Test each attack
        for attack_name, attack_fn in ATTACKS.items():
            img_att = attack_fn(img_wm)
            if img_att.shape != img_wm.shape:
                img_att = F.interpolate(img_att, size=img_wm.shape[2:], mode='bilinear', align_corners=False)
            
            z_att = vae.encode(img_att)
            z_l_att, z_h_att = splitter(z_att)
            w_pred_l = decoder_l(z_l_att)
            w_pred_h = decoder_h(z_h_att)
            
            bits_true = (w > 0).float()
            bits_pred = ((w_pred_l + w_pred_h) / 2 > 0).float()
            acc = (bits_true == bits_pred).float().mean().item()
            results[attack_name].append(acc)
        
        if device.type == 'mps':
            torch.mps.empty_cache()
    
    # Print results
    print('\n' + '='*120)
    print('RESULTS: finetune_efficient')
    print('='*120)
    
    print(f"{'Model':<14}|", end='')
    for name in ATTACKS.keys():
        print(f'{name:>12} |', end='')
    print()
    print('-'*140)
    
    print(f"{'finetune_eff':<14}|", end='')
    for name in ATTACKS.keys():
        acc = np.mean(results[name]) * 100
        print(f'{acc:>11.1f} |', end='')
    print()
    print('-'*140)
    
    avg = np.mean([np.mean(v) for v in results.values()]) * 100
    print(f'\nAverage accuracy: {avg:.1f}%')
    
    # Compare with other models
    print('\n' + '='*80)
    print('COMPARISON (from previous run):')
    print('='*80)
    print(f'  efficient   : 68.8%')
    print(f'  fast_staged : 69.7%')
    print(f'  decoder_ft  : 73.4%')
    print(f'  roundtrip   : 72.6%')
    print(f'  lightweight : 76.4%')
    print(f'  finetune_eff: {avg:.1f}%')


if __name__ == '__main__':
    main()

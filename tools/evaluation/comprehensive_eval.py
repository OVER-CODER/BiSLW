#!/usr/bin/env python3
"""
Comprehensive evaluation of ALL models for final comparison.
Evaluates: Quality (PSNR/SSIM), Detection (latent/roundtrip), Robustness (10 attacks)
"""

import os
import sys
import json
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from datetime import datetime
import shutil

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.dirname(project_root))

from latent_watermarking.models.latent_split import LatentSplitter
from latent_watermarking.models.recombination import LatentRecombiner
from latent_watermarking.models.watermark_encoder import WatermarkEncoder
from latent_watermarking.models.watermark_decoder import WatermarkDecoder
from latent_watermarking.models.vae_wrapper import VAEWrapper


# ============================================================
# ALL MODELS TO EVALUATE
# ============================================================
MODELS = {
    'efficient': 'results/efficient_20260222_004718/best_model.pth',
    'fast_staged': 'results/fast_staged_20260222_110232/best_roundtrip.pt',
    'decoder_ft': 'results/decoder_ft_20260222_212319/best.pt',
    'roundtrip': 'results/roundtrip_train_20260222_172359/best_roundtrip.pt',
    'lightweight': 'results/lightweight_20260222_233224/best.pt',
    'finetune_eff': 'results/finetune_efficient_20260225_181255/best.pt',
}

# ============================================================
# ATTACK FUNCTIONS
# ============================================================
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

# ============================================================
# METRIC FUNCTIONS
# ============================================================
def compute_psnr(img1, img2, data_range=2.0):
    mse = ((img1 - img2) ** 2).mean()
    if mse == 0:
        return float('inf')
    return 10 * torch.log10((data_range ** 2) / mse).item()

def compute_ssim(img1, img2, window_size=11, data_range=2.0):
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    mu1 = F.avg_pool2d(img1, window_size, stride=1, padding=window_size//2)
    mu2 = F.avg_pool2d(img2, window_size, stride=1, padding=window_size//2)
    mu1_sq, mu2_sq = mu1**2, mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.avg_pool2d(img1**2, window_size, stride=1, padding=window_size//2) - mu1_sq
    sigma2_sq = F.avg_pool2d(img2**2, window_size, stride=1, padding=window_size//2) - mu2_sq
    sigma12 = F.avg_pool2d(img1*img2, window_size, stride=1, padding=window_size//2) - mu1_mu2
    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2)) / ((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean().item()


def load_model(path, device):
    """Load model from checkpoint."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    w_dim = ckpt.get('config', {}).get('w_dim', 32)
    alpha_l = ckpt.get('alpha_l', 0.02)
    alpha_h = ckpt.get('alpha_h', 0.01)
    
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
    
    for m in [encoder_l, encoder_h, decoder_l, decoder_h]:
        m.eval()
    
    return {
        'splitter': splitter,
        'recombiner': recombiner,
        'encoder_l': encoder_l,
        'encoder_h': encoder_h,
        'decoder_l': decoder_l,
        'decoder_h': decoder_h,
        'alpha_l': alpha_l,
        'alpha_h': alpha_h,
        'w_dim': w_dim,
    }


def evaluate_model(model, vae, latents, n_samples=100, device='cpu'):
    """Comprehensive evaluation of a single model."""
    results = {
        'quality': {'psnr': [], 'ssim': []},
        'detection': {'latent': [], 'roundtrip': []},
        'attacks': {name: [] for name in ATTACKS.keys()},
    }
    
    for i in tqdm(range(n_samples), desc="Evaluating", leave=False):
        z = latents[i:i+1].to(device)
        w = torch.randn(1, model['w_dim'], device=device)
        w = (w > 0).float() * 2 - 1
        
        # Embed watermark
        z_l, z_h = model['splitter'](z)
        z_l_wm = model['encoder_l'](z_l, w, alpha=model['alpha_l'])
        z_h_wm = model['encoder_h'](z_h, w, alpha=model['alpha_h'])
        z_wm = model['recombiner'](z_l_wm, z_h_wm)
        
        # QUALITY: Decode and compute PSNR/SSIM
        with torch.no_grad():
            img_orig = vae.decode(z)
            img_wm = vae.decode(z_wm)
        
        results['quality']['psnr'].append(compute_psnr(img_orig, img_wm))
        results['quality']['ssim'].append(compute_ssim(img_orig, img_wm))
        
        # DETECTION - Latent space (no VAE roundtrip)
        w_pred_l = model['decoder_l'](z_l_wm)
        w_pred_h = model['decoder_h'](z_h_wm)
        bits_true = (w > 0).float()
        bits_pred = ((w_pred_l + w_pred_h) / 2 > 0).float()
        latent_acc = (bits_true == bits_pred).float().mean().item()
        results['detection']['latent'].append(latent_acc)
        
        # DETECTION - Roundtrip (VAE encode-decode)
        z_rt = vae.encode(img_wm)
        z_l_rt, z_h_rt = model['splitter'](z_rt)
        w_pred_l_rt = model['decoder_l'](z_l_rt)
        w_pred_h_rt = model['decoder_h'](z_h_rt)
        bits_pred_rt = ((w_pred_l_rt + w_pred_h_rt) / 2 > 0).float()
        rt_acc = (bits_true == bits_pred_rt).float().mean().item()
        results['detection']['roundtrip'].append(rt_acc)
        
        # ATTACKS
        for attack_name, attack_fn in ATTACKS.items():
            img_att = attack_fn(img_wm)
            if img_att.shape != img_wm.shape:
                img_att = F.interpolate(img_att, size=img_wm.shape[2:], mode='bilinear', align_corners=False)
            
            z_att = vae.encode(img_att)
            z_l_att, z_h_att = model['splitter'](z_att)
            w_pred_l_att = model['decoder_l'](z_l_att)
            w_pred_h_att = model['decoder_h'](z_h_att)
            bits_pred_att = ((w_pred_l_att + w_pred_h_att) / 2 > 0).float()
            att_acc = (bits_true == bits_pred_att).float().mean().item()
            results['attacks'][attack_name].append(att_acc)
        
        if device.type == 'mps':
            torch.mps.empty_cache()
    
    # Aggregate results
    return {
        'quality': {
            'psnr_mean': np.mean(results['quality']['psnr']),
            'psnr_std': np.std(results['quality']['psnr']),
            'ssim_mean': np.mean(results['quality']['ssim']),
            'ssim_std': np.std(results['quality']['ssim']),
        },
        'detection': {
            'latent_acc': np.mean(results['detection']['latent']),
            'roundtrip_acc': np.mean(results['detection']['roundtrip']),
        },
        'attacks': {name: np.mean(accs) for name, accs in results['attacks'].items()},
        'attacks_avg': np.mean([np.mean(accs) for accs in results['attacks'].values()]),
    }


def main():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load VAE
    print("Loading VAE...")
    vae = VAEWrapper().to(device)
    
    # Load latents
    latent_path = os.path.join(project_root, 'cache/latents_20000_256.pt')
    print(f"Loading latents: {latent_path}")
    latent_data = torch.load(latent_path, map_location='cpu', weights_only=False)
    latents = latent_data['latents']
    print(f"Loaded {len(latents)} latents")
    
    n_samples = 100  # Use 100 samples for comprehensive eval
    all_results = {}
    
    # Evaluate each model
    for name, path in MODELS.items():
        full_path = os.path.join(project_root, path)
        if not os.path.exists(full_path):
            print(f"\nSkipping {name} - not found: {full_path}")
            continue
            
        print(f"\n{'='*60}")
        print(f"Evaluating: {name}")
        print(f"{'='*60}")
        
        model = load_model(full_path, device)
        print(f"  alpha_l={model['alpha_l']}, alpha_h={model['alpha_h']}")
        
        results = evaluate_model(model, vae, latents, n_samples, device)
        results['alpha_l'] = model['alpha_l']
        results['alpha_h'] = model['alpha_h']
        results['path'] = path
        all_results[name] = results
        
        print(f"  PSNR: {results['quality']['psnr_mean']:.2f} dB")
        print(f"  SSIM: {results['quality']['ssim_mean']:.4f}")
        print(f"  Latent Acc: {results['detection']['latent_acc']*100:.1f}%")
        print(f"  Roundtrip Acc: {results['detection']['roundtrip_acc']*100:.1f}%")
        print(f"  Attacks Avg: {results['attacks_avg']*100:.1f}%")
    
    # ============================================================
    # PRINT SUMMARY TABLES
    # ============================================================
    print("\n" + "="*120)
    print("COMPREHENSIVE RESULTS SUMMARY")
    print("="*120)
    
    # Quality Table
    print("\n--- IMAGE QUALITY ---")
    print(f"{'Model':<15} | {'PSNR (dB)':>12} | {'SSIM':>12} | {'alpha_l':>10} | {'alpha_h':>10}")
    print("-"*70)
    for name, r in sorted(all_results.items(), key=lambda x: -x[1]['quality']['psnr_mean']):
        print(f"{name:<15} | {r['quality']['psnr_mean']:>12.2f} | {r['quality']['ssim_mean']:>12.4f} | {r['alpha_l']:>10.4f} | {r['alpha_h']:>10.4f}")
    
    # Detection Table
    print("\n--- DETECTION ACCURACY ---")
    print(f"{'Model':<15} | {'Latent Acc':>12} | {'Roundtrip Acc':>14}")
    print("-"*50)
    for name, r in sorted(all_results.items(), key=lambda x: -x[1]['detection']['latent_acc']):
        print(f"{name:<15} | {r['detection']['latent_acc']*100:>11.1f}% | {r['detection']['roundtrip_acc']*100:>13.1f}%")
    
    # Attacks Table
    print("\n--- ATTACK ROBUSTNESS ---")
    attack_names = list(ATTACKS.keys())
    header = f"{'Model':<12} |"
    for att in attack_names:
        header += f" {att:>10} |"
    header += f" {'AVG':>8} |"
    print(header)
    print("-"*145)
    
    for name, r in sorted(all_results.items(), key=lambda x: -x[1]['attacks_avg']):
        row = f"{name:<12} |"
        for att in attack_names:
            row += f" {r['attacks'][att]*100:>9.1f}% |"
        row += f" {r['attacks_avg']*100:>7.1f}% |"
        print(row)
    
    # Best Model Summary
    print("\n" + "="*80)
    print("BEST MODELS BY CATEGORY")
    print("="*80)
    
    best_psnr = max(all_results.items(), key=lambda x: x[1]['quality']['psnr_mean'])
    best_ssim = max(all_results.items(), key=lambda x: x[1]['quality']['ssim_mean'])
    best_latent = max(all_results.items(), key=lambda x: x[1]['detection']['latent_acc'])
    best_roundtrip = max(all_results.items(), key=lambda x: x[1]['detection']['roundtrip_acc'])
    best_attacks = max(all_results.items(), key=lambda x: x[1]['attacks_avg'])
    
    print(f"  Best PSNR:      {best_psnr[0]} ({best_psnr[1]['quality']['psnr_mean']:.2f} dB)")
    print(f"  Best SSIM:      {best_ssim[0]} ({best_ssim[1]['quality']['ssim_mean']:.4f})")
    print(f"  Best Latent:    {best_latent[0]} ({best_latent[1]['detection']['latent_acc']*100:.1f}%)")
    print(f"  Best Roundtrip: {best_roundtrip[0]} ({best_roundtrip[1]['detection']['roundtrip_acc']*100:.1f}%)")
    print(f"  Best Attacks:   {best_attacks[0]} ({best_attacks[1]['attacks_avg']*100:.1f}%)")
    
    # Save results to JSON
    output_path = os.path.join(project_root, 'best res/comprehensive_results.json')
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {output_path}")
    
    return all_results


if __name__ == '__main__':
    main()

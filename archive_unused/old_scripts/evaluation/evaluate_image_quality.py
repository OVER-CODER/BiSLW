#!/usr/bin/env python3
"""Evaluate image-space quality metrics (PSNR, SSIM) through VAE decode."""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from latent_watermarking.models.latent_split import LatentSplitter
from latent_watermarking.models.recombination import LatentRecombiner
from latent_watermarking.models.watermark_encoder import WatermarkEncoder
from latent_watermarking.models.watermark_decoder import WatermarkDecoder


def compute_psnr(img1, img2, data_range=2.0):
    """Compute PSNR between two images."""
    mse = ((img1 - img2) ** 2).mean()
    if mse == 0:
        return float('inf')
    return 10 * torch.log10((data_range ** 2) / mse).item()


def compute_ssim(img1, img2, window_size=11):
    """Compute SSIM between two images."""
    C1 = (0.01 * 2) ** 2  # data_range = 2
    C2 = (0.03 * 2) ** 2
    
    # Simple global SSIM
    mu1 = img1.mean()
    mu2 = img2.mean()
    sigma1_sq = ((img1 - mu1) ** 2).mean()
    sigma2_sq = ((img2 - mu2) ** 2).mean()
    sigma12 = ((img1 - mu1) * (img2 - mu2)).mean()
    
    ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
           ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim.item()


def main():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load VAE
    print("Loading VAE for image-space metrics...")
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(
        'runwayml/stable-diffusion-v1-5', 
        subfolder='vae'
    ).to(device)
    vae.eval()
    
    # Load model checkpoint - find latest
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, 'results')
    runs = sorted([d for d in os.listdir(results_dir) if d.startswith('efficient_')])
    latest_run = runs[-1] if runs else 'efficient_20260221_162635'
    checkpoint_path = os.path.join(results_dir, latest_run, 'best_model.pth')
    print(f"Loading checkpoint: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint['config']
    
    # Initialize models
    w_dim = config.get('w_dim', 32)
    splitter = LatentSplitter(mode='dct').to(device)
    recombiner = LatentRecombiner(mode='dct').to(device)
    encoder_l = WatermarkEncoder(watermark_dim=w_dim).to(device)
    encoder_h = WatermarkEncoder(watermark_dim=w_dim).to(device)
    
    encoder_l.load_state_dict(checkpoint['encoder_l'])
    encoder_h.load_state_dict(checkpoint['encoder_h'])
    encoder_l.eval()
    encoder_h.eval()
    
    alpha_l = checkpoint['alpha_l']
    alpha_h = checkpoint['alpha_h']
    
    # Load latents
    latent_path = os.path.join(script_dir, 'cache/latents_10000_128.pt')
    print(f"Loading latents: {latent_path}")
    latents = torch.load(latent_path, map_location='cpu')['latents']
    
    print(f"\nLatent statistics:")
    print(f"  Shape: {latents.shape}")
    print(f"  Min: {latents.min():.3f}, Max: {latents.max():.3f}")
    print(f"  Mean: {latents.mean():.3f}, Std: {latents.std():.3f}")
    
    # Compute image-space metrics
    print(f"\nComputing IMAGE-SPACE metrics (VAE decode)...")
    print(f"Alpha L/H: {alpha_l:.3f}/{alpha_h:.3f}")
    
    psnr_list = []
    ssim_list = []
    latent_mse_list = []
    
    num_samples = 20  # Evaluate on 20 samples
    
    with torch.no_grad():
        for i in range(num_samples):
            z = latents[i:i+1].to(device)
            w = torch.randn(1, w_dim, device=device)
            
            # Watermark embedding
            z_low, z_high = splitter(z)
            z_low_wm = encoder_l(z_low, w, alpha=alpha_l)
            z_high_wm = encoder_h(z_high, w, alpha=alpha_h)
            z_wm = recombiner(z_low_wm, z_high_wm)
            
            # Latent-space MSE
            latent_mse = ((z - z_wm) ** 2).mean().item()
            latent_mse_list.append(latent_mse)
            
            # Decode to images (SD VAE uses 0.18215 scaling)
            img_orig = vae.decode(z / 0.18215).sample
            img_wm = vae.decode(z_wm / 0.18215).sample
            
            # Clamp to valid range [-1, 1]
            img_orig = torch.clamp(img_orig, -1, 1)
            img_wm = torch.clamp(img_wm, -1, 1)
            
            # Compute metrics
            psnr = compute_psnr(img_orig, img_wm, data_range=2.0)
            ssim = compute_ssim(img_orig, img_wm)
            
            psnr_list.append(psnr)
            ssim_list.append(ssim)
            
            print(f"  Sample {i+1}: PSNR={psnr:.2f} dB, SSIM={ssim:.4f}")
            
            if device.type == 'mps':
                torch.mps.empty_cache()
    
    # Summary
    print("\n" + "=" * 60)
    print("IMAGE-SPACE QUALITY METRICS")
    print("=" * 60)
    print(f"\nPSNR: {np.mean(psnr_list):.2f} dB +/- {np.std(psnr_list):.2f} dB")
    print(f"SSIM: {np.mean(ssim_list):.4f} +/- {np.std(ssim_list):.4f}")
    print(f"\nLatent MSE: {np.mean(latent_mse_list):.6f}")
    print(f"\nTargets: 40 dB PSNR, 0.91 SSIM")
    
    # Check if targets are met
    avg_psnr = np.mean(psnr_list)
    avg_ssim = np.mean(ssim_list)
    
    print("\n" + "-" * 60)
    if avg_psnr >= 40.0:
        print(f"PSNR target MET: {avg_psnr:.2f} >= 40 dB")
    else:
        print(f"PSNR target NOT MET: {avg_psnr:.2f} < 40 dB")
        print(f"  -> Reduce alpha values to improve quality")
        
    if avg_ssim >= 0.91:
        print(f"SSIM target MET: {avg_ssim:.4f} >= 0.91")
    else:
        print(f"SSIM target NOT MET: {avg_ssim:.4f} < 0.91")
    print("=" * 60)


if __name__ == '__main__':
    main()

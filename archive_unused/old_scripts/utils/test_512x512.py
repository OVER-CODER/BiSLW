#!/usr/bin/env python3
"""Test VAE baseline and watermarking at 512x512 resolution."""

import torch
import numpy as np
from diffusers import AutoencoderKL
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.latent_split import LatentSplitter
from models.recombination import LatentRecombiner
from models.watermark_encoder import WatermarkEncoder
from models.watermark_decoder import WatermarkDecoder


def compute_metrics(img1, img2):
    """Compute PSNR and SSIM."""
    mse = ((img1 - img2) ** 2).mean().item()
    psnr = 10 * np.log10(4.0 / (mse + 1e-10))
    
    C1, C2 = 0.0001, 0.0009
    mu1, mu2 = img1.mean(), img2.mean()
    sigma1_sq = ((img1 - mu1) ** 2).mean()
    sigma2_sq = ((img2 - mu2) ** 2).mean()
    sigma12 = ((img1 - mu1) * (img2 - mu2)).mean()
    ssim = ((2*mu1*mu2 + C1) * (2*sigma12 + C2)) / \
           ((mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2))
    return psnr, ssim.item()


def main():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")
    print("Resolution: 512x512")
    
    print("\nLoading SD v1.5 VAE...")
    vae = AutoencoderKL.from_pretrained(
        'runwayml/stable-diffusion-v1-5', 
        subfolder='vae'
    ).to(device)
    vae.eval()
    
    # Generate synthetic images at 512x512 and encode to latents
    print("\nGenerating test images at 512x512...")
    num_samples = 10
    
    # Create random images (simulating real images)
    torch.manual_seed(42)
    test_images = torch.randn(num_samples, 3, 512, 512).clamp(-1, 1)
    
    print("\n" + "="*60)
    print("TEST 1: BASELINE VAE (No Watermarking)")
    print("="*60)
    
    baseline_psnr = []
    baseline_ssim = []
    
    with torch.no_grad():
        for i in range(num_samples):
            img = test_images[i:i+1].to(device)
            
            # Encode
            z = vae.encode(img).latent_dist.mean * 0.18215
            
            # Decode
            img_recon = vae.decode(z / 0.18215).sample
            img_recon = torch.clamp(img_recon, -1, 1)
            
            psnr, ssim = compute_metrics(img, img_recon)
            baseline_psnr.append(psnr)
            baseline_ssim.append(ssim)
            
            print(f"  Sample {i+1}: PSNR={psnr:.2f} dB, SSIM={ssim:.4f}")
            torch.mps.empty_cache()
    
    print(f"\nBaseline Average: PSNR={np.mean(baseline_psnr):.2f} dB, SSIM={np.mean(baseline_ssim):.4f}")
    
    # Now test with watermarking
    print("\n" + "="*60)
    print("TEST 2: WITH WATERMARKING")
    print("="*60)
    
    # Load trained model
    checkpoint_path = 'results/efficient_20260221_171939/best_model.pth'
    if os.path.exists(checkpoint_path):
        print(f"Loading model: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        config = checkpoint['config']
        
        splitter = LatentSplitter(mode='dct').to(device)
        recombiner = LatentRecombiner(mode='dct').to(device)
        encoder_l = WatermarkEncoder(watermark_dim=config.get('w_dim', 32)).to(device)
        encoder_h = WatermarkEncoder(watermark_dim=config.get('w_dim', 32)).to(device)
        decoder_l = WatermarkDecoder(watermark_dim=config.get('w_dim', 32)).to(device)
        decoder_h = WatermarkDecoder(watermark_dim=config.get('w_dim', 32)).to(device)
        
        encoder_l.load_state_dict(checkpoint['encoder_l'])
        encoder_h.load_state_dict(checkpoint['encoder_h'])
        decoder_l.load_state_dict(checkpoint['decoder_l'])
        decoder_h.load_state_dict(checkpoint['decoder_h'])
        
        encoder_l.eval()
        encoder_h.eval()
        decoder_l.eval()
        decoder_h.eval()
        
        alpha_l = checkpoint['alpha_l']
        alpha_h = checkpoint['alpha_h']
        w_dim = config.get('w_dim', 32)
        
        print(f"Alpha L/H: {alpha_l}/{alpha_h}")
        
        wm_psnr = []
        wm_ssim = []
        bit_acc = []
        
        with torch.no_grad():
            for i in range(num_samples):
                img = test_images[i:i+1].to(device)
                
                # Encode to latent
                z = vae.encode(img).latent_dist.mean * 0.18215
                
                # Generate watermark
                w = torch.randn(1, w_dim, device=device)
                
                # Embed watermark
                z_low, z_high = splitter(z)
                z_low_wm = encoder_l(z_low, w, alpha=alpha_l)
                z_high_wm = encoder_h(z_high, w, alpha=alpha_h)
                z_wm = recombiner(z_low_wm, z_high_wm)
                
                # Decode watermarked
                img_wm = vae.decode(z_wm / 0.18215).sample
                img_wm = torch.clamp(img_wm, -1, 1)
                
                # Quality metrics
                psnr, ssim = compute_metrics(img, img_wm)
                wm_psnr.append(psnr)
                wm_ssim.append(ssim)
                
                # Decode watermark
                z_wm_low, z_wm_high = splitter(z_wm)
                w_pred = (decoder_l(z_wm_low) + decoder_h(z_wm_high)) / 2
                bits_true = (w > 0).float()
                bits_pred = (w_pred > 0).float()
                acc = (bits_true == bits_pred).float().mean().item()
                bit_acc.append(acc)
                
                print(f"  Sample {i+1}: PSNR={psnr:.2f} dB, SSIM={ssim:.4f}, BitAcc={acc:.2%}")
                torch.mps.empty_cache()
        
        print(f"\nWatermarked Average: PSNR={np.mean(wm_psnr):.2f} dB, SSIM={np.mean(wm_ssim):.4f}, BitAcc={np.mean(bit_acc):.2%}")
    else:
        print("No trained model found")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY (512x512 Resolution)")
    print("="*60)
    print(f"Baseline VAE:    PSNR={np.mean(baseline_psnr):.2f} dB, SSIM={np.mean(baseline_ssim):.4f}")
    if 'wm_psnr' in dir():
        print(f"With Watermark:  PSNR={np.mean(wm_psnr):.2f} dB, SSIM={np.mean(wm_ssim):.4f}, BitAcc={np.mean(bit_acc):.2%}")
        print(f"\nQuality Loss from Watermarking: {np.mean(baseline_psnr) - np.mean(wm_psnr):.2f} dB")
    print("="*60)


if __name__ == '__main__':
    main()

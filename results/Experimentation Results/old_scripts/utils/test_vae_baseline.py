#!/usr/bin/env python3
"""Test baseline VAE reconstruction quality (no watermarking)."""

import torch
import numpy as np
from diffusers import AutoencoderKL

def main():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")
    
    print("Loading SD v1.5 VAE...")
    vae = AutoencoderKL.from_pretrained(
        'runwayml/stable-diffusion-v1-5', 
        subfolder='vae'
    ).to(device)
    vae.eval()
    
    # Load precomputed latents
    latents = torch.load('cache/latents_10000_128.pt', map_location='cpu')['latents']
    print(f"Latents shape: {latents.shape}")
    print(f"Latent range: [{latents.min():.3f}, {latents.max():.3f}]")
    
    print("\n" + "="*60)
    print("BASELINE VAE RECONSTRUCTION TEST (No Watermarking)")
    print("="*60)
    print("Testing: z -> decode -> img -> encode -> z' -> decode -> img'")
    print("Comparing: img vs img' (one encode-decode cycle)")
    print("="*60)
    
    psnr_list = []
    ssim_list = []
    
    with torch.no_grad():
        for i in range(20):
            z = latents[i:i+1].to(device)
            
            # Decode latent to image
            img = vae.decode(z / 0.18215).sample
            img = torch.clamp(img, -1, 1)
            
            # Re-encode image back to latent
            z_recon = vae.encode(img).latent_dist.mean * 0.18215
            
            # Decode again
            img_recon = vae.decode(z_recon / 0.18215).sample
            img_recon = torch.clamp(img_recon, -1, 1)
            
            # PSNR between original decode and re-encode-decode
            mse = ((img - img_recon) ** 2).mean().item()
            psnr = 10 * np.log10(4.0 / (mse + 1e-10))
            psnr_list.append(psnr)
            
            # Simple SSIM
            C1, C2 = 0.0001, 0.0009
            mu1, mu2 = img.mean(), img_recon.mean()
            sigma1_sq = ((img - mu1) ** 2).mean()
            sigma2_sq = ((img_recon - mu2) ** 2).mean()
            sigma12 = ((img - mu1) * (img_recon - mu2)).mean()
            ssim = ((2*mu1*mu2 + C1) * (2*sigma12 + C2)) / \
                   ((mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2))
            ssim_list.append(ssim.item())
            
            print(f"  Sample {i+1}: PSNR={psnr:.2f} dB, SSIM={ssim.item():.4f}")
            
            if device.type == 'mps':
                torch.mps.empty_cache()
    
    print("\n" + "="*60)
    print("VAE BASELINE RESULTS (encode-decode cycle)")
    print("="*60)
    print(f"PSNR: {np.mean(psnr_list):.2f} dB +/- {np.std(psnr_list):.2f} dB")
    print(f"SSIM: {np.mean(ssim_list):.4f} +/- {np.std(ssim_list):.4f}")
    print("\nThis is the theoretical MAXIMUM quality.")
    print("Any watermarking will result in LOWER metrics.")
    print("="*60)


if __name__ == '__main__':
    main()

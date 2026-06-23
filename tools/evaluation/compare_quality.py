#!/usr/bin/env python3
"""Compare image quality metrics between models."""

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
from latent_watermarking.models.vae_wrapper import VAEWrapper


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


def main():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f'Device: {device}')
    
    # Load VAE
    print('Loading VAE...')
    vae = VAEWrapper().to(device)
    
    # Load latents
    latent_path = os.path.join(project_root, 'cache/latents_20000_256.pt')
    latent_data = torch.load(latent_path, map_location='cpu', weights_only=False)
    latents = latent_data['latents']
    print(f'Loaded {len(latents)} latents')
    
    # Models to compare
    models = {
        'efficient': os.path.join(project_root, 'results/efficient_20260222_004718/best_model.pth'),
        'lightweight': os.path.join(project_root, 'results/lightweight_20260222_233224/best.pt'),
        'finetune_eff': os.path.join(project_root, 'results/finetune_efficient_20260225_181255/best.pt'),
    }
    
    n_samples = 50
    results = {}
    
    for name, path in models.items():
        print(f'\nEvaluating {name}...')
        ckpt = torch.load(path, map_location=device, weights_only=False)
        w_dim = ckpt.get('config', {}).get('w_dim', 32)
        alpha_l = ckpt.get('alpha_l', 0.02)
        alpha_h = ckpt.get('alpha_h', 0.01)
        
        splitter = LatentSplitter(mode='dct').to(device)
        recombiner = LatentRecombiner(mode='dct').to(device)
        encoder_l = WatermarkEncoder(watermark_dim=w_dim).to(device)
        encoder_h = WatermarkEncoder(watermark_dim=w_dim).to(device)
        
        encoder_l.load_state_dict(ckpt['encoder_l'])
        encoder_h.load_state_dict(ckpt['encoder_h'])
        encoder_l.eval()
        encoder_h.eval()
        
        psnr_list, ssim_list = [], []
        
        for i in tqdm(range(n_samples), desc=name):
            z = latents[i:i+1].to(device)
            w = torch.randn(1, w_dim, device=device)
            w = (w > 0).float() * 2 - 1
            
            z_l, z_h = splitter(z)
            z_l_wm = encoder_l(z_l, w, alpha=alpha_l)
            z_h_wm = encoder_h(z_h, w, alpha=alpha_h)
            z_wm = recombiner(z_l_wm, z_h_wm)
            
            with torch.no_grad():
                img_orig = vae.decode(z)
                img_wm = vae.decode(z_wm)
            
            psnr_list.append(compute_psnr(img_orig, img_wm))
            ssim_list.append(compute_ssim(img_orig, img_wm))
            
            if device.type == 'mps':
                torch.mps.empty_cache()
        
        results[name] = {
            'psnr': np.mean(psnr_list),
            'psnr_std': np.std(psnr_list),
            'ssim': np.mean(ssim_list),
            'ssim_std': np.std(ssim_list),
            'alpha_l': alpha_l,
            'alpha_h': alpha_h,
        }
    
    print('\n' + '='*80)
    print('IMAGE QUALITY COMPARISON')
    print('='*80)
    print(f"{'Model':<15} | {'PSNR (dB)':>15} | {'SSIM':>15} | {'alpha_l':>10} | {'alpha_h':>10}")
    print('-'*80)
    for name, r in results.items():
        psnr_str = f"{r['psnr']:.2f} ± {r['psnr_std']:.2f}"
        ssim_str = f"{r['ssim']:.4f} ± {r['ssim_std']:.4f}"
        print(f"{name:<15} | {psnr_str:>15} | {ssim_str:>15} | {r['alpha_l']:>10.4f} | {r['alpha_h']:>10.4f}")
    print('-'*80)
    
    print('\nSUMMARY:')
    for name, r in results.items():
        print(f"  {name}: PSNR={r['psnr']:.2f} dB, SSIM={r['ssim']:.4f}")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Fast comprehensive evaluation - no VGG/LPIPS/FID to avoid downloads.
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import json
import argparse
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from latent_watermarking.models.latent_split import LatentSplitter
from latent_watermarking.models.recombination import LatentRecombiner
from latent_watermarking.models.watermark_encoder import WatermarkEncoder
from latent_watermarking.models.watermark_decoder import WatermarkDecoder


def compute_psnr(img1, img2, data_range=2.0):
    mse = ((img1 - img2) ** 2).mean()
    if mse == 0:
        return float('inf')
    return 10 * torch.log10((data_range ** 2) / mse).item()


def compute_ssim(img1, img2, window_size=11, data_range=2.0):
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    
    def gaussian_window(size, sigma=1.5):
        coords = torch.arange(size, device=img1.device, dtype=img1.dtype) - size // 2
        gauss = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        return (gauss / gauss.sum()).unsqueeze(1) @ (gauss / gauss.sum()).unsqueeze(0)
    
    window = gaussian_window(window_size).unsqueeze(0).unsqueeze(0).expand(3, 1, -1, -1)
    pad = window_size // 2
    
    mu1 = F.conv2d(img1, window, padding=pad, groups=3)
    mu2 = F.conv2d(img2, window, padding=pad, groups=3)
    sigma1_sq = F.conv2d(img1 * img1, window, padding=pad, groups=3) - mu1 ** 2
    sigma2_sq = F.conv2d(img2 * img2, window, padding=pad, groups=3) - mu2 ** 2
    sigma12 = F.conv2d(img1 * img2, window, padding=pad, groups=3) - mu1 * mu2
    
    ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / ((mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean().item()


# Attacks
def jpeg_attack(images, quality):
    scale = max(0.5, quality / 100)
    B, C, H, W = images.shape
    h_s, w_s = max(1, int(H * scale)), max(1, int(W * scale))
    down = F.interpolate(images, size=(h_s, w_s), mode='bilinear', align_corners=False)
    up = F.interpolate(down, size=(H, W), mode='bilinear', align_corners=False)
    return (quality / 100) * images + (1 - quality / 100) * up


def gaussian_noise_attack(images, sigma):
    return (images + torch.randn_like(images) * sigma).clamp(-1, 1)


def gaussian_blur_attack(images, kernel_size):
    if kernel_size % 2 == 0:
        kernel_size += 1
    sigma = kernel_size / 3
    x = torch.arange(kernel_size, dtype=images.dtype, device=images.device) - kernel_size // 2
    kernel_1d = torch.exp(-x**2 / (2 * sigma**2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel = (kernel_1d.unsqueeze(1) @ kernel_1d.unsqueeze(0)).unsqueeze(0).unsqueeze(0).expand(3, 1, -1, -1)
    return F.conv2d(images, kernel, padding=kernel_size // 2, groups=3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--samples', type=int, default=100)
    args = parser.parse_args()
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Output dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"results/eval_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load VAE
    print("Loading VAE...")
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained('runwayml/stable-diffusion-v1-5', subfolder='vae', torch_dtype=torch.float32).to(device)
    vae.eval()
    scaling = 0.18215
    
    # Load model
    print(f"Loading: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = ckpt.get('config', {})
    w_dim = config.get('w_dim', 32)
    
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
    
    alpha_l = ckpt.get('alpha_l', 0.02)
    alpha_h = ckpt.get('alpha_h', 0.01)
    print(f"Alpha: {alpha_l}/{alpha_h}")
    
    def embed(z, w):
        z_l, z_h = splitter(z)
        z_l_wm = encoder_l(z_l, w, alpha=alpha_l)
        z_h_wm = encoder_h(z_h, w, alpha=alpha_h)
        return recombiner(z_l_wm, z_h_wm)
    
    def extract(z):
        z_l, z_h = splitter(z)
        return decoder_l(z_l), decoder_h(z_h)
    
    def bit_acc(w_true, w_l, w_h):
        bits_true = (w_true > 0).float()
        bits_pred = ((w_l + w_h) / 2 > 0).float()
        return (bits_true == bits_pred).float().mean().item()
    
    # Load test data
    latent_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache/latents_20000_256.pt')
    print(f"Loading latents: {latent_path}")
    latents = torch.load(latent_path, map_location='cpu', weights_only=False)
    if isinstance(latents, dict):
        latents = latents.get('latents', latents.get('z_orig', list(latents.values())[0]))
    latents = latents[:args.samples]
    watermarks = torch.randn(len(latents), w_dim)
    
    results = {}
    
    # =========================================
    # 1. IMAGE QUALITY
    # =========================================
    print("\n" + "="*50)
    print("1. IMAGE QUALITY")
    print("="*50)
    
    psnr_list, ssim_list = [], []
    
    for i in tqdm(range(min(20, len(latents))), desc="Quality"):
        z = latents[i:i+1].to(device)
        w = watermarks[i:i+1].to(device)
        
        with torch.no_grad():
            z_wm = embed(z, w)
            img_orig = vae.decode(z / scaling).sample
            img_wm = vae.decode(z_wm / scaling).sample
        
        psnr_list.append(compute_psnr(img_orig, img_wm))
        ssim_list.append(compute_ssim(img_orig, img_wm))
    
    results['psnr'] = {'mean': np.mean(psnr_list), 'std': np.std(psnr_list)}
    results['ssim'] = {'mean': np.mean(ssim_list), 'std': np.std(ssim_list)}
    
    print(f"  PSNR: {results['psnr']['mean']:.2f} ± {results['psnr']['std']:.2f} dB")
    print(f"  SSIM: {results['ssim']['mean']:.4f} ± {results['ssim']['std']:.4f}")
    
    # =========================================
    # 2. WATERMARK DETECTION
    # =========================================
    print("\n" + "="*50)
    print("2. WATERMARK DETECTION")
    print("="*50)
    
    latent_acc, roundtrip_acc = [], []
    
    for i in tqdm(range(args.samples), desc="Detection"):
        z = latents[i:i+1].to(device)
        w = watermarks[i:i+1].to(device)
        
        with torch.no_grad():
            z_wm = embed(z, w)
            
            # Latent accuracy
            w_l, w_h = extract(z_wm)
            latent_acc.append(bit_acc(w, w_l, w_h))
            
            # VAE roundtrip accuracy
            img_wm = vae.decode(z_wm / scaling).sample
            z_rt = vae.encode(img_wm).latent_dist.mean * scaling
            w_l, w_h = extract(z_rt)
            roundtrip_acc.append(bit_acc(w, w_l, w_h))
        
        if device.type == 'mps' and i % 20 == 0:
            torch.mps.empty_cache()
    
    results['latent_acc'] = {'mean': np.mean(latent_acc), 'std': np.std(latent_acc)}
    results['roundtrip_acc'] = {'mean': np.mean(roundtrip_acc), 'std': np.std(roundtrip_acc)}
    
    print(f"  Latent accuracy: {results['latent_acc']['mean']:.4f} ± {results['latent_acc']['std']:.4f}")
    print(f"  VAE roundtrip:   {results['roundtrip_acc']['mean']:.4f} ± {results['roundtrip_acc']['std']:.4f}")
    
    # =========================================
    # 3. ATTACK ROBUSTNESS
    # =========================================
    print("\n" + "="*50)
    print("3. ATTACK ROBUSTNESS")
    print("="*50)
    
    attacks = [
        ("JPEG-90", lambda x: jpeg_attack(x, 90)),
        ("JPEG-70", lambda x: jpeg_attack(x, 70)),
        ("JPEG-50", lambda x: jpeg_attack(x, 50)),
        ("Noise-0.01", lambda x: gaussian_noise_attack(x, 0.01)),
        ("Noise-0.05", lambda x: gaussian_noise_attack(x, 0.05)),
        ("Blur-3", lambda x: gaussian_blur_attack(x, 3)),
        ("Blur-5", lambda x: gaussian_blur_attack(x, 5)),
    ]
    
    n_attack_samples = min(50, args.samples)
    results['attacks'] = {}
    
    for attack_name, attack_fn in attacks:
        accuracies = []
        
        for i in range(n_attack_samples):
            z = latents[i:i+1].to(device)
            w = watermarks[i:i+1].to(device)
            
            with torch.no_grad():
                z_wm = embed(z, w)
                img_wm = vae.decode(z_wm / scaling).sample
                
                # Apply attack
                img_att = attack_fn(img_wm)
                z_att = vae.encode(img_att).latent_dist.mean * scaling
                
                w_l, w_h = extract(z_att)
                accuracies.append(bit_acc(w, w_l, w_h))
            
            if device.type == 'mps':
                torch.mps.empty_cache()
        
        results['attacks'][attack_name] = {'mean': np.mean(accuracies), 'std': np.std(accuracies)}
        print(f"  {attack_name:12}: {results['attacks'][attack_name]['mean']:.4f}")
    
    # =========================================
    # 4. COMPUTATIONAL
    # =========================================
    print("\n" + "="*50)
    print("4. COMPUTATIONAL")
    print("="*50)
    
    import time
    z_test = latents[0:1].to(device)
    w_test = watermarks[0:1].to(device)
    
    # Warmup
    for _ in range(5):
        _ = embed(z_test, w_test)
        _ = extract(z_test)
    
    # Time encode
    n_timing = 50
    start = time.time()
    for _ in range(n_timing):
        with torch.no_grad():
            _ = embed(z_test, w_test)
    encode_time = (time.time() - start) / n_timing * 1000
    
    # Time decode
    start = time.time()
    for _ in range(n_timing):
        with torch.no_grad():
            _ = extract(z_test)
    decode_time = (time.time() - start) / n_timing * 1000
    
    results['timing'] = {'encode_ms': encode_time, 'decode_ms': decode_time}
    print(f"  Encode: {encode_time:.2f} ms")
    print(f"  Decode: {decode_time:.2f} ms")
    
    # Model size
    total_params = sum(p.numel() for m in [encoder_l, encoder_h, decoder_l, decoder_h] for p in m.parameters())
    results['model_params'] = total_params
    print(f"  Parameters: {total_params:,}")
    
    # =========================================
    # SUMMARY
    # =========================================
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print(f"  PSNR: {results['psnr']['mean']:.2f} dB (target: 40 dB)")
    print(f"  SSIM: {results['ssim']['mean']:.4f} (target: 0.91)")
    print(f"  Latent Acc: {results['latent_acc']['mean']:.4f}")
    print(f"  Roundtrip Acc: {results['roundtrip_acc']['mean']:.4f}")
    print(f"  Avg Attack Acc: {np.mean([v['mean'] for v in results['attacks'].values()]):.4f}")
    print(f"  Output: {output_dir}")
    
    # Save
    with open(f"{output_dir}/results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_dir}/results.json")


if __name__ == "__main__":
    main()

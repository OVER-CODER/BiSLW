#!/usr/bin/env python3
"""
Interpolate weights between two models to find quality/robustness sweet spot.

new_model = α * model_A + (1-α) * model_B

Usage:
  python scripts/evaluation/interpolate_models.py --alpha 0.6
"""

import os
import sys
import argparse
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


# Attack functions
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
    blurred = []
    for c in range(images.shape[1]):
        blurred.append(F.conv2d(images[:, c:c+1], kernel, padding=padding))
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


def interpolate_state_dicts(sd1, sd2, alpha):
    """Interpolate two state dicts: new = alpha * sd1 + (1-alpha) * sd2"""
    result = {}
    for key in sd1.keys():
        if key in sd2:
            result[key] = alpha * sd1[key] + (1 - alpha) * sd2[key]
        else:
            result[key] = sd1[key]
    return result


class InterpolatedModel:
    def __init__(self, ckpt_a, ckpt_b, alpha, device):
        self.device = device
        self.alpha = alpha
        
        # Load both checkpoints
        ckpt_a_data = torch.load(ckpt_a, map_location=device, weights_only=False)
        ckpt_b_data = torch.load(ckpt_b, map_location=device, weights_only=False)
        
        w_dim = ckpt_a_data.get('config', {}).get('w_dim', 32)
        
        # Create models
        self.splitter = LatentSplitter(mode='dct').to(device)
        self.recombiner = LatentRecombiner(mode='dct').to(device)
        self.encoder_l = WatermarkEncoder(watermark_dim=w_dim).to(device)
        self.encoder_h = WatermarkEncoder(watermark_dim=w_dim).to(device)
        self.decoder_l = WatermarkDecoder(watermark_dim=w_dim).to(device)
        self.decoder_h = WatermarkDecoder(watermark_dim=w_dim).to(device)
        
        # Interpolate weights
        self.encoder_l.load_state_dict(interpolate_state_dicts(
            ckpt_a_data['encoder_l'], ckpt_b_data['encoder_l'], alpha))
        self.encoder_h.load_state_dict(interpolate_state_dicts(
            ckpt_a_data['encoder_h'], ckpt_b_data['encoder_h'], alpha))
        self.decoder_l.load_state_dict(interpolate_state_dicts(
            ckpt_a_data['decoder_l'], ckpt_b_data['decoder_l'], alpha))
        self.decoder_h.load_state_dict(interpolate_state_dicts(
            ckpt_a_data['decoder_h'], ckpt_b_data['decoder_h'], alpha))
        
        # Interpolate alphas too
        alpha_l_a = ckpt_a_data.get('alpha_l', 0.02)
        alpha_l_b = ckpt_b_data.get('alpha_l', 0.02)
        alpha_h_a = ckpt_a_data.get('alpha_h', 0.01)
        alpha_h_b = ckpt_b_data.get('alpha_h', 0.01)
        
        self.alpha_l = alpha * alpha_l_a + (1 - alpha) * alpha_l_b
        self.alpha_h = alpha * alpha_h_a + (1 - alpha) * alpha_h_b
        
        self._set_eval_mode()
        
    def _set_eval_mode(self):
        self.encoder_l.eval()
        self.encoder_h.eval()
        self.decoder_l.eval()
        self.decoder_h.eval()
        
    def embed_watermark(self, z, w):
        z_low, z_high = self.splitter(z)
        z_low_wm = self.encoder_l(z_low, w, alpha=self.alpha_l)
        z_high_wm = self.encoder_h(z_high, w, alpha=self.alpha_h)
        return self.recombiner(z_low_wm, z_high_wm)
    
    def extract_watermark(self, z):
        z_low, z_high = self.splitter(z)
        return self.decoder_l(z_low), self.decoder_h(z_high)


def compute_bit_accuracy(w_true, w_pred_l, w_pred_h):
    bits_true = (w_true > 0).float()
    bits_pred = ((w_pred_l + w_pred_h) / 2 > 0).float()
    return (bits_true == bits_pred).float().mean().item()


def compute_psnr(img1, img2):
    mse = F.mse_loss(img1, img2)
    if mse == 0:
        return float('inf')
    return 10 * torch.log10(4 / mse).item()  # images in [-1,1] so max range is 2


def compute_ssim(img1, img2, window_size=11):
    C1, C2 = 0.01**2, 0.03**2
    mu1 = F.avg_pool2d(img1, window_size, stride=1, padding=window_size//2)
    mu2 = F.avg_pool2d(img2, window_size, stride=1, padding=window_size//2)
    
    mu1_sq, mu2_sq, mu1_mu2 = mu1**2, mu2**2, mu1 * mu2
    
    sigma1_sq = F.avg_pool2d(img1**2, window_size, stride=1, padding=window_size//2) - mu1_sq
    sigma2_sq = F.avg_pool2d(img2**2, window_size, stride=1, padding=window_size//2) - mu2_sq
    sigma12 = F.avg_pool2d(img1*img2, window_size, stride=1, padding=window_size//2) - mu1_mu2
    
    ssim_map = ((2*mu1_mu2 + C1) * (2*sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean().item()


@torch.no_grad()
def evaluate_model(model, vae, latents, n_samples=100):
    """Evaluate model on all attacks and quality metrics."""
    
    indices = torch.randperm(len(latents))[:n_samples].tolist()
    
    attack_results = {name: [] for name in ATTACKS.keys()}
    psnr_values = []
    ssim_values = []
    
    for i in tqdm(indices, desc="Evaluating"):
        z = latents[i:i+1].to(model.device)
        w = torch.randn(1, 32, device=model.device)
        w = (w > 0).float() * 2 - 1  # Binary watermark
        
        # Embed watermark
        z_wm = model.embed_watermark(z, w)
        
        # Decode to images for quality metrics and attacks
        img_orig = vae.decode(z)
        img_wm = vae.decode(z_wm)
        
        # Quality metrics
        psnr_values.append(compute_psnr(img_orig, img_wm))
        ssim_values.append(compute_ssim(img_orig, img_wm))
        
        # Attack evaluation
        for attack_name, attack_fn in ATTACKS.items():
            img_att = attack_fn(img_wm)
            
            # Handle size changes from crop attacks
            if img_att.shape != img_wm.shape:
                img_att = F.interpolate(img_att, size=img_wm.shape[2:], mode='bilinear', align_corners=False)
            
            z_att = vae.encode(img_att)
            w_pred_l, w_pred_h = model.extract_watermark(z_att)
            acc = compute_bit_accuracy(w, w_pred_l, w_pred_h)
            attack_results[attack_name].append(acc)
    
    return {
        'psnr': np.mean(psnr_values),
        'ssim': np.mean(ssim_values),
        'attacks': {k: np.mean(v) * 100 for k, v in attack_results.items()}
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', type=float, nargs='+', default=[0.3, 0.5, 0.7],
                       help='Interpolation alpha(s): new = α*lightweight + (1-α)*efficient')
    parser.add_argument('--n_samples', type=int, default=100)
    parser.add_argument('--latents', type=str, 
                       default='cache/latents_20000_256.pt')
    args = parser.parse_args()
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 
                         'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Model paths
    lightweight_path = "best res/lightweight_20260222_233224/best.pt"
    efficient_path = "best res/efficient_20260222_004718/best_model.pth"
    
    # Load VAE
    print("Loading VAE...")
    vae = VAEWrapper()
    vae = vae.to(device)
    
    # Load latents
    print(f"Loading latents: {args.latents}")
    latents_data = torch.load(args.latents, map_location='cpu', weights_only=True)
    latents = latents_data['latents'] if isinstance(latents_data, dict) else latents_data
    
    # Test multiple alpha values
    results = {}
    
    for alpha in args.alpha:
        print(f"\n{'='*60}")
        print(f"Testing α = {alpha} (lightweight:{alpha*100:.0f}% + efficient:{(1-alpha)*100:.0f}%)")
        print('='*60)
        
        model = InterpolatedModel(lightweight_path, efficient_path, alpha, device)
        print(f"  Interpolated alpha_l={model.alpha_l:.4f}, alpha_h={model.alpha_h:.4f}")
        
        metrics = evaluate_model(model, vae, latents, args.n_samples)
        results[alpha] = metrics
        
        print(f"\n  Quality: PSNR={metrics['psnr']:.2f} dB, SSIM={metrics['ssim']:.4f}")
        print(f"  Attacks:")
        for attack, acc in metrics['attacks'].items():
            print(f"    {attack:12s}: {acc:.1f}%")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY: Interpolation Results")
    print("="*80)
    print(f"{'Alpha':>8s} | {'PSNR (dB)':>10s} | {'SSIM':>8s} | {'Clean Acc':>10s} | {'Avg Attack':>10s}")
    print("-"*60)
    
    for alpha, metrics in results.items():
        avg_attack = np.mean([v for k, v in metrics['attacks'].items() if k != 'None'])
        print(f"{alpha:>8.2f} | {metrics['psnr']:>10.2f} | {metrics['ssim']:>8.4f} | "
              f"{metrics['attacks']['None']:>9.1f}% | {avg_attack:>9.1f}%")
    
    # Find best balanced model
    print("\nRecommendation:")
    best_alpha = None
    best_score = 0
    for alpha, metrics in results.items():
        # Score: normalize PSNR (target 37), SSIM (target 0.9), and avg accuracy
        avg_attack = np.mean([v for k, v in metrics['attacks'].items() if k != 'None'])
        psnr_score = min(metrics['psnr'] / 37, 1.0)
        ssim_score = min(metrics['ssim'] / 0.9, 1.0)
        acc_score = avg_attack / 75  # Normalize to ~75% target
        
        combined = psnr_score * 0.3 + ssim_score * 0.2 + acc_score * 0.5
        if combined > best_score:
            best_score = combined
            best_alpha = alpha
    
    print(f"  Best balanced α = {best_alpha}")
    print(f"  PSNR: {results[best_alpha]['psnr']:.2f} dB")
    print(f"  SSIM: {results[best_alpha]['ssim']:.4f}")
    print(f"  Avg attack accuracy: {np.mean([v for k, v in results[best_alpha]['attacks'].items() if k != 'None']):.1f}%")


if __name__ == '__main__':
    main()

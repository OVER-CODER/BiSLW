#!/usr/bin/env python3
"""
Ablation study on effect of image quality loss weights.
Evaluates PSNR, SSIM, LPIPS, SIFID, and bit accuracy for different lambda_latent values.
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from models.latent_split import LatentSplitter
from models.recombination import LatentRecombiner
from models.watermark_encoder import WatermarkEncoder
from models.watermark_decoder import WatermarkDecoder


class LPIPS(nn.Module):
    """Simplified LPIPS using VGG features."""
    def __init__(self, device):
        super().__init__()
        from torchvision import models
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features[:23].to(device)
        vgg.eval()
        for p in vgg.parameters():
            p.requires_grad = False
        self.vgg = vgg
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
    def forward(self, x, y):
        # Normalize from [-1,1] to ImageNet range
        x = (x + 1) / 2
        y = (y + 1) / 2
        x = (x - self.mean.to(x.device)) / self.std.to(x.device)
        y = (y - self.mean.to(y.device)) / self.std.to(y.device)
        
        fx = self.vgg(x)
        fy = self.vgg(y)
        return F.mse_loss(fx, fy)


def compute_sifid(feat1, feat2):
    """Simplified FID between feature batches."""
    mu1, mu2 = feat1.mean(0), feat2.mean(0)
    return ((mu1 - mu2) ** 2).sum().item()


def load_vae(device):
    """Load VAE."""
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(
        'runwayml/stable-diffusion-v1-5',
        subfolder='vae',
        torch_dtype=torch.float32
    ).to(device)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False
    return vae


def decode_latent(vae, z, scaling_factor=0.18215):
    with torch.no_grad():
        return vae.decode(z / scaling_factor).sample


def apply_attack(images, attack_name, vae, device):
    """Apply attack to images."""
    if attack_name == "None":
        return images
    
    B, C, H, W = images.shape
    
    if attack_name == "Comb.":
        # Combined attack: JPEG + slight rotation + contrast
        # JPEG
        scale = 0.7
        h_s, w_s = max(8, int(H * scale)), max(8, int(W * scale))
        images = F.interpolate(images, (h_s, w_s), mode='bilinear', align_corners=False)
        images = F.interpolate(images, (H, W), mode='bilinear', align_corners=False)
        # Contrast
        images = torch.clamp(images * 1.3, -1, 1)
    
    return images


def train_and_evaluate(lambda_latent, latents, vae, device, config):
    """Train model with specific lambda_latent and evaluate."""
    w_dim = config['w_dim']
    epochs = config['epochs']
    batch_size = config['batch_size']
    alpha_l = config['alpha_l']
    alpha_h = config['alpha_h']
    
    # Initialize models
    splitter = LatentSplitter(mode='dct').to(device)
    recombiner = LatentRecombiner(mode='dct').to(device)
    encoder_l = WatermarkEncoder(watermark_dim=w_dim).to(device)
    encoder_h = WatermarkEncoder(watermark_dim=w_dim).to(device)
    decoder_l = WatermarkDecoder(watermark_dim=w_dim).to(device)
    decoder_h = WatermarkDecoder(watermark_dim=w_dim).to(device)
    
    params = (
        list(encoder_l.parameters()) + list(encoder_h.parameters()) +
        list(decoder_l.parameters()) + list(decoder_h.parameters())
    )
    optimizer = torch.optim.AdamW(params, lr=2e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    n_batches = len(latents) // batch_size
    
    # Training
    for epoch in range(epochs):
        encoder_l.train()
        encoder_h.train()
        decoder_l.train()
        decoder_h.train()
        
        indices = torch.randperm(len(latents))
        
        for b in range(n_batches):
            idx = indices[b * batch_size:(b + 1) * batch_size]
            z = latents[idx].to(device)
            w = torch.randn(batch_size, w_dim, device=device)
            
            # Forward
            z_low, z_high = splitter(z)
            z_low_wm = encoder_l(z_low, w, alpha=alpha_l)
            z_high_wm = encoder_h(z_high, w, alpha=alpha_h)
            z_wm = recombiner(z_low_wm, z_high_wm)
            
            z_wm_low, z_wm_high = splitter(z_wm)
            w_pred_l = decoder_l(z_wm_low)
            w_pred_h = decoder_h(z_wm_high)
            
            # Losses
            loss_w = F.mse_loss(w_pred_l, w) + F.mse_loss(w_pred_h, w)
            loss_cons = F.mse_loss(w_pred_l, w_pred_h)
            loss_latent = F.mse_loss(z_wm, z)
            
            loss = loss_w + 0.3 * loss_cons + lambda_latent * loss_latent
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
        
        scheduler.step()
    
    # Evaluation
    encoder_l.eval()
    encoder_h.eval()
    decoder_l.eval()
    decoder_h.eval()
    
    lpips_fn = LPIPS(device)
    
    n_eval = config['n_eval']
    eval_indices = torch.randperm(len(latents))[:n_eval]
    
    psnr_list, ssim_list, lpips_list = [], [], []
    acc_none_list, acc_comb_list = [], []
    orig_feats, wm_feats = [], []
    
    with torch.no_grad():
        for idx in eval_indices:
            z = latents[idx:idx+1].to(device)
            w = torch.randn(1, w_dim, device=device)
            
            # Watermark
            z_low, z_high = splitter(z)
            z_low_wm = encoder_l(z_low, w, alpha=alpha_l)
            z_high_wm = encoder_h(z_high, w, alpha=alpha_h)
            z_wm = recombiner(z_low_wm, z_high_wm)
            
            # Decode to images
            img_orig = decode_latent(vae, z)
            img_wm = decode_latent(vae, z_wm)
            
            # Quality metrics
            img_orig_np = ((img_orig[0] + 1) / 2).clamp(0, 1).cpu().numpy()
            img_wm_np = ((img_wm[0] + 1) / 2).clamp(0, 1).cpu().numpy()
            
            mse = np.mean((img_orig_np - img_wm_np) ** 2)
            psnr = 10 * np.log10(1.0 / (mse + 1e-10))
            psnr_list.append(psnr)
            
            # SSIM (simplified)
            mu1 = img_orig_np.mean()
            mu2 = img_wm_np.mean()
            var1 = img_orig_np.var()
            var2 = img_wm_np.var()
            cov = np.mean((img_orig_np - mu1) * (img_wm_np - mu2))
            c1, c2 = 0.01**2, 0.03**2
            ssim = ((2*mu1*mu2 + c1) * (2*cov + c2)) / ((mu1**2 + mu2**2 + c1) * (var1 + var2 + c2))
            ssim_list.append(ssim)
            
            # LPIPS
            lpips_val = lpips_fn(img_orig, img_wm).item()
            lpips_list.append(lpips_val)
            
            # Store features for SIFID
            orig_feats.append(lpips_fn.vgg(((img_orig + 1) / 2 - lpips_fn.mean.to(device)) / lpips_fn.std.to(device)).flatten())
            wm_feats.append(lpips_fn.vgg(((img_wm + 1) / 2 - lpips_fn.mean.to(device)) / lpips_fn.std.to(device)).flatten())
            
            # Bit accuracy - None
            z_wm_low, z_wm_high = splitter(z_wm)
            w_pred_l = decoder_l(z_wm_low)
            w_pred_h = decoder_h(z_wm_high)
            w_pred = (w_pred_l + w_pred_h) / 2
            bits_true = (w > 0).float()
            bits_pred = (w_pred > 0).float()
            acc_none = (bits_true == bits_pred).float().mean().item()
            acc_none_list.append(acc_none)
            
            # Bit accuracy - Combined attack
            img_attacked = apply_attack(img_wm, "Comb.", vae, device)
            z_attacked = vae.encode(img_attacked).latent_dist.mean * 0.18215
            z_att_low, z_att_high = splitter(z_attacked)
            w_pred_l = decoder_l(z_att_low)
            w_pred_h = decoder_h(z_att_high)
            w_pred = (w_pred_l + w_pred_h) / 2
            bits_pred = (w_pred > 0).float()
            acc_comb = (bits_true == bits_pred).float().mean().item()
            acc_comb_list.append(acc_comb)
    
    # SIFID
    orig_feats = torch.stack(orig_feats)
    wm_feats = torch.stack(wm_feats)
    sifid = compute_sifid(orig_feats, wm_feats)
    
    return {
        'psnr': {'mean': float(np.mean(psnr_list)), 'std': float(np.std(psnr_list))},
        'ssim': {'mean': float(np.mean(ssim_list)), 'std': float(np.std(ssim_list))},
        'lpips': {'mean': float(np.mean(lpips_list)), 'std': float(np.std(lpips_list))},
        'sifid': float(sifid),
        'acc_none': {'mean': float(np.mean(acc_none_list)), 'std': float(np.std(acc_none_list))},
        'acc_comb': {'mean': float(np.mean(acc_comb_list)), 'std': float(np.std(acc_comb_list))}
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--n_eval', type=int, default=50)
    parser.add_argument('--latents', type=str, default='cache/latents_1000_256.pt')
    parser.add_argument('--output', type=str, default='results/ablation_loss_weights.json')
    args = parser.parse_args()
    
    os.chdir(PROJECT_ROOT)
    
    # Device
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Device: {device}")
    
    # Load data
    print(f"Loading latents from {args.latents}")
    latent_data = torch.load(args.latents, map_location='cpu', weights_only=False)
    latents = latent_data['latents'] if isinstance(latent_data, dict) else latent_data
    
    # Load VAE
    vae = load_vae(device)
    
    # Lambda values to test
    lambda_values = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
    
    config = {
        'w_dim': 32,
        'epochs': args.epochs,
        'batch_size': 32,
        'alpha_l': 0.1,
        'alpha_h': 0.05,
        'n_eval': args.n_eval
    }
    
    results = {
        'config': config,
        'lambda_latent_values': {}
    }
    
    for lambda_val in lambda_values:
        print(f"\n{'='*60}")
        print(f"Training with lambda_latent = {lambda_val}")
        print(f"{'='*60}")
        
        metrics = train_and_evaluate(lambda_val, latents, vae, device, config)
        results['lambda_latent_values'][str(lambda_val)] = metrics
        
        print(f"PSNR: {metrics['psnr']['mean']:.2f} ± {metrics['psnr']['std']:.2f}")
        print(f"SSIM: {metrics['ssim']['mean']:.4f} ± {metrics['ssim']['std']:.4f}")
        print(f"LPIPS: {metrics['lpips']['mean']:.4f} ± {metrics['lpips']['std']:.4f}")
        print(f"SIFID: {metrics['sifid']:.6f}")
        print(f"Bit Acc (None): {metrics['acc_none']['mean']*100:.1f}%")
        print(f"Bit Acc (Comb): {metrics['acc_comb']['mean']*100:.1f}%")
    
    results['timestamp'] = datetime.now().isoformat()
    
    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")
    
    # Print summary table
    print("\n" + "="*80)
    print("ABLATION STUDY: Effect of Image Quality Loss Weight (lambda_latent)")
    print("="*80)
    print(f"{'λ_latent':<10} {'PSNR↑':<12} {'SSIM↑':<12} {'LPIPS↓':<12} {'SIFID↓':<12} {'Acc(None)':<12} {'Acc(Comb)':<12}")
    print("-"*80)
    for lv in lambda_values:
        m = results['lambda_latent_values'][str(lv)]
        print(f"{lv:<10} {m['psnr']['mean']:.2f}±{m['psnr']['std']:.1f}   "
              f"{m['ssim']['mean']:.3f}±{m['ssim']['std']:.2f}  "
              f"{m['lpips']['mean']:.4f}±{m['lpips']['std']:.3f} "
              f"{m['sifid']:.5f}      "
              f"{m['acc_none']['mean']*100:.1f}%        "
              f"{m['acc_comb']['mean']*100:.1f}%")


if __name__ == '__main__':
    main()

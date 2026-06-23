#!/usr/bin/env python3
"""
Quick Regeneration Robustness Analysis using pre-trained models.
"""

import os
import sys
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.latent_split import LatentSplitter
from models.recombination import LatentRecombiner
from models.watermark_encoder import WatermarkEncoder
from models.watermark_decoder import WatermarkDecoder


def compute_bit_accuracy(extracted, target):
    extracted_bits = (extracted > 0).float()
    target_bits = (target > 0).float()
    return (extracted_bits == target_bits).float().mean(dim=-1)


def get_alpha_schedule(num_timesteps=1000, beta_start=0.00085, beta_end=0.012):
    betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_timesteps) ** 2
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    return alphas_cumprod


def add_noise(x, t, alphas_cumprod, noise=None):
    if noise is None:
        noise = torch.randn_like(x)
    alpha_t = alphas_cumprod[t].view(-1, 1, 1, 1).to(x.device)
    sqrt_alpha_t = torch.sqrt(alpha_t)
    sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)
    noisy_x = sqrt_alpha_t * x + sqrt_one_minus_alpha_t * noise
    return noisy_x, noise


def load_pretrained_models(device, checkpoint_path, w_dim=32):
    """Load pre-trained models from checkpoint."""
    print(f"Loading pre-trained models from {checkpoint_path}")
    
    splitter = LatentSplitter(mode='dct').to(device)
    recombiner = LatentRecombiner(mode='dct').to(device)
    encoder_l = WatermarkEncoder(watermark_dim=w_dim).to(device)
    encoder_h = WatermarkEncoder(watermark_dim=w_dim).to(device)
    decoder_l = WatermarkDecoder(watermark_dim=w_dim).to(device)
    decoder_h = WatermarkDecoder(watermark_dim=w_dim).to(device)
    
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    encoder_l.load_state_dict(ckpt['encoder_l'])
    encoder_h.load_state_dict(ckpt['encoder_h'])
    decoder_l.load_state_dict(ckpt['decoder_l'])
    decoder_h.load_state_dict(ckpt['decoder_h'])
    
    alpha_l = ckpt.get('alpha_l', 0.1)
    alpha_h = ckpt.get('alpha_h', 0.05)
    
    print(f"  Loaded from epoch {ckpt['epoch']}, alpha_l={alpha_l}, alpha_h={alpha_h}")
    
    models = {
        'splitter': splitter,
        'recombiner': recombiner,
        'encoder_l': encoder_l,
        'encoder_h': encoder_h,
        'decoder_l': decoder_l,
        'decoder_h': decoder_h
    }
    return models, alpha_l, alpha_h


def embed_watermark(models, z, w, alpha_l=0.1, alpha_h=0.05):
    z_low, z_high = models['splitter'](z)
    z_low_wm = models['encoder_l'](z_low, w, alpha=alpha_l)
    z_high_wm = models['encoder_h'](z_high, w, alpha=alpha_h)
    return models['recombiner'](z_low_wm, z_high_wm)


def extract_watermark(models, z):
    z_low, z_high = models['splitter'](z)
    w_l = models['decoder_l'](z_low)
    w_h = models['decoder_h'](z_high)
    return (w_l + w_h) / 2


def simulate_regeneration(z, timestep, alphas_cumprod, device):
    """Simulate diffusion regeneration (add noise, partial recovery)."""
    noise = torch.randn_like(z)
    z_noisy, _ = add_noise(z, timestep, alphas_cumprod.to(device), noise)
    
    alpha_t = alphas_cumprod[timestep].to(device)
    recovery_quality = alpha_t ** 0.5
    z_recovered = recovery_quality * z + (1 - recovery_quality) * z_noisy
    
    return z_recovered


def main():
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # Load pre-trained model
    checkpoint_path = Path(__file__).parent.parent.parent / 'best res' / 'efficient_20260222_004718' / 'best_model.pth'
    models, alpha_l, alpha_h = load_pretrained_models(device, checkpoint_path)
    
    for m in models.values():
        m.eval()
    
    # Parameters
    n_samples = 200
    w_dim = 32
    timesteps = [0, 25, 50, 75, 100, 150, 200, 250, 300, 350, 400, 450, 500]
    
    alphas_cumprod = get_alpha_schedule()
    
    output_dir = Path(__file__).parent.parent.parent / 'results' / 'regeneration_analysis'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nRunning regeneration analysis with {n_samples} samples...")
    
    results = {}
    
    with torch.no_grad():
        z = torch.randn(n_samples, 4, 64, 64, device=device)
        w = torch.randn(n_samples, w_dim, device=device)
        
        # Embed watermark
        z_wm = embed_watermark(models, z, w, alpha_l, alpha_h)
        
        for t in timesteps:
            if t == 0:
                z_test = z_wm
            else:
                z_test = simulate_regeneration(z_wm, t, alphas_cumprod, device)
            
            w_extracted = extract_watermark(models, z_test)
            accuracies = compute_bit_accuracy(w_extracted, w)
            
            results[t] = {
                'accuracy': accuracies.mean().item(),
                'std': accuracies.std().item()
            }
            print(f"  Timestep {t:3d}: {accuracies.mean().item()*100:.1f}% ± {accuracies.std().item()*100:.1f}%")
    
    # Plot
    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'legend.fontsize': 10,
        'figure.figsize': (7, 5)
    })
    
    ts = sorted(results.keys())
    accs = [results[t]['accuracy'] * 100 for t in ts]
    stds = [results[t]['std'] * 100 for t in ts]
    
    fig, ax = plt.subplots(figsize=(7, 5))
    
    ax.plot(ts, accs, 'b-', linewidth=2, marker='o', markersize=6, label='BiSLW')
    ax.fill_between(ts, 
                    [a - s for a, s in zip(accs, stds)],
                    [a + s for a, s in zip(accs, stds)],
                    alpha=0.2, color='blue')
    
    ax.axhline(y=50, color='r', linestyle='--', linewidth=1.5, label='Random Baseline (50%)')
    
    ax.set_xlabel('Regeneration Timestep')
    ax.set_ylabel('Bit Accuracy (%)')
    ax.set_title('Watermark Robustness to Diffusion Regeneration')
    ax.set_xlim([0, max(ts)])
    ax.set_ylim([45, 105])
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'regeneration_robustness.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'regeneration_robustness.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save JSON
    save_results = {
        'config': {
            'w_dim': w_dim,
            'n_samples': n_samples,
            'alpha_l': alpha_l,
            'alpha_h': alpha_h,
            'checkpoint': str(checkpoint_path)
        },
        'results': {str(k): v for k, v in results.items()},
        'timestamp': datetime.now().isoformat()
    }
    
    with open(output_dir / 'regeneration_robustness.json', 'w') as f:
        json.dump(save_results, f, indent=2)
    
    print(f"\nSaved plot to {output_dir}/regeneration_robustness.png")
    print(f"Saved results to {output_dir}/regeneration_robustness.json")
    
    # Summary
    print("\n" + "="*50)
    print("Regeneration Robustness Summary")
    print("="*50)
    print(f"{'Timestep':>10} | {'Accuracy':>10} | {'Std':>8}")
    print("-"*35)
    for t in ts:
        print(f"{t:>10} | {results[t]['accuracy']*100:>9.1f}% | {results[t]['std']*100:>7.1f}%")


if __name__ == '__main__':
    main()

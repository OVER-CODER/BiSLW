#!/usr/bin/env python3
"""
Regeneration Robustness Analysis for Latent Watermarking
- Tests watermark extraction accuracy under diffusion regeneration attacks
- Plots accuracy vs regeneration timestep
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.latent_split import LatentSplitter
from models.recombination import LatentRecombiner
from models.watermark_encoder import WatermarkEncoder
from models.watermark_decoder import WatermarkDecoder


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_bit_accuracy(extracted, target):
    """Compute bit accuracy between extracted and target watermarks."""
    extracted_bits = (extracted > 0).float()
    target_bits = (target > 0).float()
    return (extracted_bits == target_bits).float().mean(dim=-1)


def get_alpha_schedule(num_timesteps=1000, beta_start=0.00085, beta_end=0.012):
    """Get alpha schedule for diffusion (same as Stable Diffusion)."""
    betas = torch.linspace(beta_start ** 0.5, beta_end ** 0.5, num_timesteps) ** 2
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    return alphas_cumprod


def add_noise(x, t, alphas_cumprod, noise=None):
    """Add noise to x at timestep t using the diffusion schedule."""
    if noise is None:
        noise = torch.randn_like(x)
    
    alpha_t = alphas_cumprod[t].view(-1, 1, 1, 1).to(x.device)
    sqrt_alpha_t = torch.sqrt(alpha_t)
    sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)
    
    noisy_x = sqrt_alpha_t * x + sqrt_one_minus_alpha_t * noise
    return noisy_x, noise


def train_models(device, w_dim=32, epochs=150, alpha_l=0.1, alpha_h=0.05, n_train=3000):
    """Train watermark encoder/decoder pair."""
    print(f"Training models for {epochs} epochs...")
    
    splitter = LatentSplitter(mode='dct').to(device)
    recombiner = LatentRecombiner(mode='dct').to(device)
    encoder_l = WatermarkEncoder(watermark_dim=w_dim).to(device)
    encoder_h = WatermarkEncoder(watermark_dim=w_dim).to(device)
    decoder_l = WatermarkDecoder(watermark_dim=w_dim).to(device)
    decoder_h = WatermarkDecoder(watermark_dim=w_dim).to(device)
    
    latents = torch.randn(n_train, 4, 64, 64)
    
    params = (
        list(encoder_l.parameters()) + list(encoder_h.parameters()) +
        list(decoder_l.parameters()) + list(decoder_h.parameters())
    )
    optimizer = torch.optim.AdamW(params, lr=2e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    batch_size = 32
    n_batches = len(latents) // batch_size
    
    # Get alpha schedule for regeneration training
    alphas_cumprod = get_alpha_schedule()
    
    for epoch in range(epochs):
        encoder_l.train()
        encoder_h.train()
        decoder_l.train()
        decoder_h.train()
        
        indices = torch.randperm(len(latents))
        epoch_loss = 0
        
        for b in range(n_batches):
            idx = indices[b * batch_size:(b + 1) * batch_size]
            z = latents[idx].to(device)
            w = torch.randn(batch_size, w_dim, device=device)
            
            # Forward
            z_low, z_high = splitter(z)
            z_low_wm = encoder_l(z_low, w, alpha=alpha_l)
            z_high_wm = encoder_h(z_high, w, alpha=alpha_h)
            z_wm = recombiner(z_low_wm, z_high_wm)
            
            # Extract from clean watermarked
            z_wm_low, z_wm_high = splitter(z_wm)
            w_pred_l = decoder_l(z_wm_low)
            w_pred_h = decoder_h(z_wm_high)
            
            # Basic losses
            loss_w = F.mse_loss(w_pred_l, w) + F.mse_loss(w_pred_h, w)
            loss_cons = F.mse_loss(w_pred_l, w_pred_h)
            loss_latent = F.mse_loss(z_wm, z)
            
            # Add regeneration robustness training (random timestep)
            if epoch > epochs // 4:  # Start after warmup
                t = torch.randint(50, 400, (1,)).item()
                z_wm_noisy, _ = add_noise(z_wm, t, alphas_cumprod.to(device))
                
                # Extract from noisy
                z_noisy_low, z_noisy_high = splitter(z_wm_noisy)
                w_noisy_l = decoder_l(z_noisy_low)
                w_noisy_h = decoder_h(z_noisy_high)
                
                loss_regen = 0.5 * (F.mse_loss(w_noisy_l, w) + F.mse_loss(w_noisy_h, w))
            else:
                loss_regen = 0
            
            loss = loss_w + 0.3 * loss_cons + 5.0 * loss_latent + 0.5 * loss_regen
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
        
        scheduler.step()
        
        if (epoch + 1) % 25 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/n_batches:.4f}")
    
    models = {
        'splitter': splitter,
        'recombiner': recombiner,
        'encoder_l': encoder_l,
        'encoder_h': encoder_h,
        'decoder_l': decoder_l,
        'decoder_h': decoder_h
    }
    return models


def embed_watermark(models, z, w, alpha_l=0.1, alpha_h=0.05):
    """Embed watermark into latent."""
    z_low, z_high = models['splitter'](z)
    z_low_wm = models['encoder_l'](z_low, w, alpha=alpha_l)
    z_high_wm = models['encoder_h'](z_high, w, alpha=alpha_h)
    return models['recombiner'](z_low_wm, z_high_wm)


def extract_watermark(models, z):
    """Extract watermark from latent."""
    z_low, z_high = models['splitter'](z)
    w_l = models['decoder_l'](z_low)
    w_h = models['decoder_h'](z_high)
    return (w_l + w_h) / 2


def simulate_regeneration(z, timestep, alphas_cumprod, device):
    """
    Simulate diffusion regeneration attack.
    Add noise up to timestep, then simulate denoising (imperfect recovery).
    
    In practice, full regeneration requires a UNet, but we simulate the effect:
    - Higher timestep = more information loss
    - The watermark in low-frequency components survives better
    """
    # Add noise
    noise = torch.randn_like(z)
    z_noisy, _ = add_noise(z, timestep, alphas_cumprod.to(device), noise)
    
    # Simulate imperfect denoising (the core signal is partially recovered)
    # This is a simplified model - real denoising would use UNet
    alpha_t = alphas_cumprod[timestep].to(device)
    
    # Recovery quality decreases with timestep
    # At t=0, perfect recovery; at t=1000, nearly random
    recovery_quality = alpha_t ** 0.5
    
    # Simulated recovery: blend between original and noisy
    z_recovered = recovery_quality * z + (1 - recovery_quality) * z_noisy
    
    return z_recovered


def run_regeneration_analysis(models, device, w_dim, timesteps, n_samples=100, alpha_l=0.1, alpha_h=0.05):
    """Evaluate watermark accuracy at different regeneration timesteps."""
    print(f"\nRunning regeneration analysis for timesteps: {timesteps}")
    
    alphas_cumprod = get_alpha_schedule()
    
    for m in models.values():
        m.eval()
    
    results = {}
    
    with torch.no_grad():
        # Generate test samples
        z = torch.randn(n_samples, 4, 64, 64, device=device)
        w = torch.randn(n_samples, w_dim, device=device)
        
        # Embed watermark
        z_wm = embed_watermark(models, z, w, alpha_l, alpha_h)
        
        # Baseline (no attack)
        w_extracted = extract_watermark(models, z_wm)
        acc_baseline = compute_bit_accuracy(w_extracted, w).mean().item()
        results[0] = {
            'accuracy': acc_baseline,
            'std': compute_bit_accuracy(w_extracted, w).std().item()
        }
        print(f"  Timestep 0 (baseline): {acc_baseline*100:.1f}%")
        
        # Test each timestep
        for t in tqdm(timesteps, desc="Testing timesteps"):
            z_regen = simulate_regeneration(z_wm, t, alphas_cumprod, device)
            w_extracted = extract_watermark(models, z_regen)
            accuracies = compute_bit_accuracy(w_extracted, w)
            
            results[t] = {
                'accuracy': accuracies.mean().item(),
                'std': accuracies.std().item()
            }
            print(f"  Timestep {t}: {accuracies.mean().item()*100:.1f}%")
    
    return results


def plot_results(results, output_dir):
    """Generate regeneration robustness plot."""
    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'legend.fontsize': 10,
        'figure.figsize': (7, 5)
    })
    
    timesteps = sorted(results.keys())
    accuracies = [results[t]['accuracy'] * 100 for t in timesteps]
    stds = [results[t]['std'] * 100 for t in timesteps]
    
    fig, ax = plt.subplots(figsize=(7, 5))
    
    # Plot with error bands
    ax.plot(timesteps, accuracies, 'b-', linewidth=2, marker='o', markersize=6, label='BiSLW')
    ax.fill_between(timesteps, 
                    [a - s for a, s in zip(accuracies, stds)],
                    [a + s for a, s in zip(accuracies, stds)],
                    alpha=0.2, color='blue')
    
    # Random baseline
    ax.axhline(y=50, color='r', linestyle='--', linewidth=1.5, label='Random Baseline (50%)')
    
    # Mark key regions
    ax.axvspan(0, 100, alpha=0.1, color='green', label='Light Regeneration')
    ax.axvspan(100, 300, alpha=0.1, color='yellow')
    ax.axvspan(300, 500, alpha=0.1, color='orange')
    
    ax.set_xlabel('Regeneration Timestep')
    ax.set_ylabel('Bit Accuracy (%)')
    ax.set_title('Watermark Robustness to Diffusion Regeneration')
    ax.set_xlim([0, max(timesteps)])
    ax.set_ylim([45, 105])
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'regeneration_robustness.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'regeneration_robustness.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved plot to {output_dir}/regeneration_robustness.png")


def main():
    parser = argparse.ArgumentParser(description='Regeneration Robustness Analysis')
    parser.add_argument('--w_dim', type=int, default=32, help='Watermark dimension')
    parser.add_argument('--epochs', type=int, default=150, help='Training epochs')
    parser.add_argument('--n_samples', type=int, default=200, help='Number of test samples')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--alpha_l', type=float, default=0.1, help='Alpha for low frequency')
    parser.add_argument('--alpha_h', type=float, default=0.05, help='Alpha for high frequency')
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    # Device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # Output directory
    output_dir = Path(__file__).parent.parent.parent / 'results' / 'regeneration_analysis'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Train models
    models = train_models(device, args.w_dim, args.epochs, args.alpha_l, args.alpha_h)
    
    # Timesteps to test (0 to 500 in increments)
    timesteps = [25, 50, 75, 100, 150, 200, 250, 300, 350, 400, 450, 500]
    
    # Run analysis
    results = run_regeneration_analysis(
        models, device, args.w_dim, timesteps, 
        n_samples=args.n_samples, alpha_l=args.alpha_l, alpha_h=args.alpha_h
    )
    
    # Plot results
    plot_results(results, output_dir)
    
    # Save results
    save_results = {
        'config': {
            'w_dim': args.w_dim,
            'epochs': args.epochs,
            'n_samples': args.n_samples,
            'alpha_l': args.alpha_l,
            'alpha_h': args.alpha_h,
            'seed': args.seed
        },
        'results': {str(k): v for k, v in results.items()},
        'timestamp': datetime.now().isoformat()
    }
    
    with open(output_dir / 'regeneration_robustness.json', 'w') as f:
        json.dump(save_results, f, indent=2)
    
    print(f"\nResults saved to {output_dir}/regeneration_robustness.json")
    
    # Print summary table
    print("\n" + "="*50)
    print("Regeneration Robustness Summary")
    print("="*50)
    print(f"{'Timestep':>10} | {'Accuracy':>10} | {'Std':>8}")
    print("-"*35)
    for t in sorted(results.keys()):
        print(f"{t:>10} | {results[t]['accuracy']*100:>9.1f}% | {results[t]['std']*100:>7.1f}%")


if __name__ == '__main__':
    main()

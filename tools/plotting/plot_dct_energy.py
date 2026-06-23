#!/usr/bin/env python3
"""
Plot DCT energy distribution before and after watermark embedding.
Shows energy decreasing with frequency, comparing original vs watermarked.
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from models.latent_split import LatentSplitter
from models.recombination import LatentRecombiner
from models.watermark_encoder import WatermarkEncoder
from models.watermark_decoder import WatermarkDecoder


def load_vae(device):
    """Load VAE for encoding images to latents."""
    print("Loading VAE...")
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


def encode_image(vae, img, scaling_factor=0.18215):
    """Encode image to latent."""
    with torch.no_grad():
        latent = vae.encode(img).latent_dist.mean
        return latent * scaling_factor


def load_models(checkpoint_path, device, w_dim=32):
    """Load trained watermark models."""
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    splitter = LatentSplitter(mode='dct').to(device)
    recombiner = LatentRecombiner(mode='dct').to(device)
    encoder_l = WatermarkEncoder(watermark_dim=w_dim).to(device)
    encoder_h = WatermarkEncoder(watermark_dim=w_dim).to(device)
    
    encoder_l.load_state_dict(checkpoint['encoder_l'])
    encoder_h.load_state_dict(checkpoint['encoder_h'])
    
    alpha_l = checkpoint.get('alpha_l', 0.3)
    alpha_h = checkpoint.get('alpha_h', 0.15)
    
    encoder_l.eval()
    encoder_h.eval()
    
    return {
        'splitter': splitter,
        'recombiner': recombiner,
        'encoder_l': encoder_l,
        'encoder_h': encoder_h,
        'alpha_l': alpha_l,
        'alpha_h': alpha_h
    }


def load_sample_images(sample_dir, n_images=5, size=512, device='cpu'):
    """Load sample images."""
    extensions = ['*.png', '*.jpg', '*.jpeg']
    image_files = []
    for ext in extensions:
        image_files.extend(list(Path(sample_dir).glob(ext)))
    
    image_files = sorted(image_files)[:n_images]
    images = []
    
    for img_path in image_files:
        print(f"  Loading: {img_path.name}")
        img = Image.open(img_path).convert('RGB')
        w, h = img.size
        min_dim = min(w, h)
        left = (w - min_dim) // 2
        top = (h - min_dim) // 2
        img = img.crop((left, top, left + min_dim, top + min_dim))
        img = img.resize((size, size), Image.LANCZOS)
        
        img_np = np.array(img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)
        img_tensor = img_tensor * 2 - 1
        images.append(img_tensor)
    
    return torch.stack(images).to(device)


def compute_dct_energy_by_frequency(z_dct, n_bins=32):
    """
    Compute energy at different frequency bands.
    Frequency is determined by distance from DC (top-left corner).
    
    Args:
        z_dct: DCT coefficients tensor (B, C, H, W)
        n_bins: Number of frequency bins
    
    Returns:
        frequencies: Array of frequency values (normalized 0-1)
        energies: Array of average energy at each frequency
    """
    B, C, H, W = z_dct.shape
    
    # Create frequency map (distance from DC corner)
    y_coords = torch.arange(H, device=z_dct.device).float()
    x_coords = torch.arange(W, device=z_dct.device).float()
    yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
    
    # Normalize distances to [0, 1]
    max_dist = np.sqrt(H**2 + W**2)
    freq_map = torch.sqrt(yy**2 + xx**2) / max_dist
    
    # Compute energy (squared magnitude)
    energy = (z_dct ** 2).mean(dim=(0, 1))  # Average over batch and channels
    
    # Bin by frequency
    bin_edges = np.linspace(0, 1, n_bins + 1)
    frequencies = (bin_edges[:-1] + bin_edges[1:]) / 2
    energies = np.zeros(n_bins)
    
    freq_map_np = freq_map.cpu().numpy()
    energy_np = energy.cpu().numpy()
    
    for i in range(n_bins):
        mask = (freq_map_np >= bin_edges[i]) & (freq_map_np < bin_edges[i+1])
        if mask.sum() > 0:
            energies[i] = energy_np[mask].mean()
    
    return frequencies, energies


def generate_dct_energy_plot(
    checkpoint_path='best res/efficient_20260222_004718/best_model.pth',
    sample_dir='sample_images/hq',
    output_dir='results/qualitative_results',
    w_dim=32,
    seed=42,
    n_samples=5
):
    """Generate DCT energy distribution plot."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Device
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # Load models
    models = load_models(checkpoint_path, device, w_dim)
    vae = load_vae(device)
    
    # Load sample images
    sample_path = Path(PROJECT_ROOT) / sample_dir
    print(f"Loading {n_samples} images from {sample_dir}...")
    images = load_sample_images(sample_path, n_images=n_samples, device=device)
    
    # Encode to latent space
    print("Encoding images to latent space...")
    z = encode_image(vae, images)
    
    # Split into frequency components (this gives us DCT representation)
    print("Computing DCT decomposition...")
    z_low, z_high = models['splitter'](z)
    
    # Full DCT representation (combine for analysis)
    z_dct_original = z_low + z_high
    
    # Generate watermark and embed
    print("Embedding watermarks...")
    w = (torch.randn(n_samples, w_dim, device=device) > 0).float() * 2 - 1
    
    with torch.no_grad():
        z_low_wm = models['encoder_l'](z_low, w, alpha=models['alpha_l'])
        z_high_wm = models['encoder_h'](z_high, w, alpha=models['alpha_h'])
    
    z_dct_watermarked = z_low_wm + z_high_wm
    
    # Compute energy distributions
    print("Computing energy distributions...")
    freq_orig, energy_orig = compute_dct_energy_by_frequency(z_dct_original)
    freq_wm, energy_wm = compute_dct_energy_by_frequency(z_dct_watermarked)
    
    # Log scale for better visualization
    energy_orig_log = np.log10(energy_orig + 1e-10)
    energy_wm_log = np.log10(energy_wm + 1e-10)
    
    # Create figure with paper styling
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'legend.fontsize': 10,
    })
    
    fig, ax = plt.subplots(figsize=(6, 4.5))
    
    # Plot energy curves
    ax.plot(freq_orig, energy_orig_log, 'b-', linewidth=2.5, label='Original', alpha=0.9)
    ax.plot(freq_wm, energy_wm_log, 'r--', linewidth=2.5, label='Watermarked', alpha=0.9)
    
    # Fill area under curves for visual effect
    ax.fill_between(freq_orig, energy_orig_log, alpha=0.2, color='blue')
    ax.fill_between(freq_wm, energy_wm_log, alpha=0.2, color='red')
    
    ax.set_xlabel('Normalized Frequency', fontsize=12, fontweight='bold')
    ax.set_ylabel(r'$\log_{10}$ Energy', fontsize=12, fontweight='bold')
    ax.set_title('DCT Energy Distribution', fontsize=14, fontweight='bold', pad=10)
    
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim([0, 1])
    
    # Add frequency labels
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(['DC', 'Low', 'Mid', 'High', 'Nyquist'])
    
    plt.tight_layout()
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, 'dct_energy_distribution.pdf'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(output_dir, 'dct_energy_distribution.png'), dpi=150, bbox_inches='tight')
    print(f"Saved: dct_energy_distribution.pdf/png")
    plt.close(fig)
    
    # Also create a difference plot
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    
    energy_diff = energy_wm - energy_orig
    energy_diff_pct = (energy_diff / (energy_orig + 1e-10)) * 100
    
    colors = ['green' if d >= 0 else 'red' for d in energy_diff_pct]
    ax2.bar(freq_orig, energy_diff_pct, width=0.025, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax2.set_xlabel('Normalized Frequency', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Energy Change (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Watermark Energy Perturbation', fontsize=14, fontweight='bold', pad=10)
    ax2.grid(True, alpha=0.3, linestyle='--', axis='y')
    ax2.set_xlim([0, 1])
    ax2.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax2.set_xticklabels(['DC', 'Low', 'Mid', 'High', 'Nyquist'])
    
    plt.tight_layout()
    fig2.savefig(os.path.join(output_dir, 'dct_energy_perturbation.pdf'), dpi=300, bbox_inches='tight')
    fig2.savefig(os.path.join(output_dir, 'dct_energy_perturbation.png'), dpi=150, bbox_inches='tight')
    print(f"Saved: dct_energy_perturbation.pdf/png")
    plt.close(fig2)
    
    # Print statistics
    print("\n=== Energy Statistics ===")
    print(f"Original total energy: {energy_orig.sum():.4f}")
    print(f"Watermarked total energy: {energy_wm.sum():.4f}")
    print(f"Energy change: {((energy_wm.sum() - energy_orig.sum()) / energy_orig.sum()) * 100:.2f}%")
    
    # Low vs high frequency energy
    mid_idx = len(freq_orig) // 2
    low_freq_orig = energy_orig[:mid_idx].sum()
    high_freq_orig = energy_orig[mid_idx:].sum()
    low_freq_wm = energy_wm[:mid_idx].sum()
    high_freq_wm = energy_wm[mid_idx:].sum()
    
    print(f"\nLow-freq energy (original): {low_freq_orig:.4f}")
    print(f"Low-freq energy (watermarked): {low_freq_wm:.4f}")
    print(f"High-freq energy (original): {high_freq_orig:.4f}")
    print(f"High-freq energy (watermarked): {high_freq_wm:.4f}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='best res/efficient_20260222_004718/best_model.pth')
    parser.add_argument('--sample_dir', type=str, default='sample_images/hq')
    parser.add_argument('--output_dir', type=str, default='results/qualitative_results')
    parser.add_argument('--w_dim', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_samples', type=int, default=5)
    args = parser.parse_args()
    
    generate_dct_energy_plot(
        checkpoint_path=args.checkpoint,
        sample_dir=args.sample_dir,
        output_dir=args.output_dir,
        w_dim=args.w_dim,
        seed=args.seed,
        n_samples=args.n_samples
    )

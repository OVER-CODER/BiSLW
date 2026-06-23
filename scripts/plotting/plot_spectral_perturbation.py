#!/usr/bin/env python3
"""
Generate spectral perturbation heatmaps showing |ΔZ_low| and |ΔZ_high|.

Expected patterns:
- Low freq: perturbations concentrated near center (DC components)
- High freq: perturbations distributed at edges
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
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


def load_sample_image(sample_dir, size=512, device='cpu'):
    """Load a sample image."""
    extensions = ['*.png', '*.jpg', '*.jpeg']
    image_files = []
    for ext in extensions:
        image_files.extend(list(Path(sample_dir).glob(ext)))
    
    if not image_files:
        raise ValueError(f"No images found in {sample_dir}")
    
    img_path = sorted(image_files)[0]
    print(f"Loading: {img_path.name}")
    
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
    
    return img_tensor.unsqueeze(0).to(device)


def generate_spectral_heatmaps(
    checkpoint_path='best res/efficient_20260222_004718/best_model.pth',
    sample_dir='sample_images/hq',
    output_dir='results/qualitative_results',
    w_dim=32,
    seed=42
):
    """Generate spectral perturbation heatmaps."""
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
    
    # Load sample image
    sample_path = Path(PROJECT_ROOT) / sample_dir
    image = load_sample_image(sample_path, device=device)
    
    # Encode to latent
    print("Encoding image to latent space...")
    z = encode_image(vae, image)
    
    # Split into low and high frequency
    print("Splitting latent into frequency components...")
    z_low, z_high = models['splitter'](z)
    
    # Generate watermark
    w = (torch.randn(1, w_dim, device=device) > 0).float() * 2 - 1
    
    # Embed watermark
    print("Embedding watermark...")
    with torch.no_grad():
        z_low_wm = models['encoder_l'](z_low, w, alpha=models['alpha_l'])
        z_high_wm = models['encoder_h'](z_high, w, alpha=models['alpha_h'])
    
    # Compute perturbations |ΔZ_low| and |ΔZ_high|
    delta_z_low = (z_low_wm - z_low).abs()
    delta_z_high = (z_high_wm - z_high).abs()
    
    # Combine into full DCT perturbation (since z_low and z_high use disjoint masks)
    delta_z_full = delta_z_low + delta_z_high
    
    # Average across channels for visualization
    delta_low_map = delta_z_low[0].mean(dim=0).cpu().numpy()
    delta_high_map = delta_z_high[0].mean(dim=0).cpu().numpy()
    delta_full_map = delta_z_full[0].mean(dim=0).cpu().numpy()
    
    # Log scale for better visualization
    delta_full_log = np.log1p(delta_full_map * 100)
    delta_low_log = np.log1p(delta_low_map * 100)
    delta_high_log = np.log1p(delta_high_map * 100)
    
    # Normalize full map
    delta_full_norm = delta_full_log / (delta_full_log.max() + 1e-8)
    delta_low_norm = delta_low_log / (delta_low_log.max() + 1e-8)
    delta_high_norm = delta_high_log / (delta_high_log.max() + 1e-8)
    
    # Compute radial frequency profile (distance from DC)
    H, W = delta_full_map.shape
    cy, cx = 0, 0  # DC is at top-left corner
    y_coords, x_coords = np.ogrid[:H, :W]
    freq_dist = np.sqrt((y_coords - cy)**2 + (x_coords - cx)**2)
    max_dist = np.sqrt(H**2 + W**2)
    
    # Bin by frequency distance and compute mean perturbation
    n_bins = 32
    bin_edges = np.linspace(0, max_dist, n_bins + 1)
    radial_profile_full = []
    radial_profile_low = []
    radial_profile_high = []
    bin_centers = []
    
    for i in range(n_bins):
        mask = (freq_dist >= bin_edges[i]) & (freq_dist < bin_edges[i+1])
        if mask.sum() > 0:
            radial_profile_full.append(delta_full_map[mask].mean())
            radial_profile_low.append(delta_low_map[mask].mean())
            radial_profile_high.append(delta_high_map[mask].mean())
            bin_centers.append((bin_edges[i] + bin_edges[i+1]) / 2)
    
    radial_profile_full = np.array(radial_profile_full)
    radial_profile_low = np.array(radial_profile_low)
    radial_profile_high = np.array(radial_profile_high)
    bin_centers = np.array(bin_centers)
    # Normalize bin centers to [0, 1]
    bin_centers_norm = bin_centers / max_dist
    
    print("Creating visualizations...")
    
    # Create figure with paper styling
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
    })
    
    # Use 'inferno' colormap for better visibility
    cmap = 'inferno'
    
    # Figure 1: Full combined DCT perturbation |ΔZ|
    fig1, ax1 = plt.subplots(figsize=(4.5, 4))
    im1 = ax1.imshow(delta_full_norm, cmap=cmap, interpolation='bilinear', origin='upper')
    ax1.set_title(r'$|\Delta Z|$ (Full DCT Perturbation)', fontsize=13, fontweight='bold', pad=10)
    ax1.set_xlabel('Horizontal frequency', fontsize=11)
    ax1.set_ylabel('Vertical frequency', fontsize=11)
    cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label('Perturbation (log scale)', fontsize=10)
    ax1.set_xticks([0, delta_full_norm.shape[1]-1])
    ax1.set_xticklabels(['DC', 'High'])
    ax1.set_yticks([0, delta_full_norm.shape[0]-1])
    ax1.set_yticklabels(['DC', 'High'])
    
    plt.tight_layout()
    fig1.savefig(os.path.join(output_dir, 'spectral_perturbation_full.pdf'), dpi=300, bbox_inches='tight')
    fig1.savefig(os.path.join(output_dir, 'spectral_perturbation_full.png'), dpi=150, bbox_inches='tight')
    print(f"Saved: spectral_perturbation_full.pdf/png")
    plt.close(fig1)
    
    # Figure 2: Radial frequency profile
    fig2, ax2 = plt.subplots(figsize=(5, 3.5))
    ax2.plot(bin_centers_norm, radial_profile_low / radial_profile_full.max(), 
             'b-', linewidth=2, label=r'Low-freq encoder ($\Delta Z_{low}$)')
    ax2.plot(bin_centers_norm, radial_profile_high / radial_profile_full.max(), 
             'r-', linewidth=2, label=r'High-freq encoder ($\Delta Z_{high}$)')
    ax2.fill_between(bin_centers_norm, 0, radial_profile_low / radial_profile_full.max(), 
                     alpha=0.3, color='blue')
    ax2.fill_between(bin_centers_norm, 0, radial_profile_high / radial_profile_full.max(), 
                     alpha=0.3, color='red')
    ax2.axvline(x=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.7, label='Nyquist/2')
    ax2.set_xlabel('Normalized frequency (distance from DC)', fontsize=11)
    ax2.set_ylabel('Perturbation magnitude', fontsize=11)
    ax2.set_title('Radial Frequency Profile', fontsize=13, fontweight='bold', pad=10)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, None)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig2.savefig(os.path.join(output_dir, 'spectral_radial_profile.pdf'), dpi=300, bbox_inches='tight')
    fig2.savefig(os.path.join(output_dir, 'spectral_radial_profile.png'), dpi=150, bbox_inches='tight')
    print(f"Saved: spectral_radial_profile.pdf/png")
    plt.close(fig2)
    
    # Figure 3: Side-by-side low/high with smooth interpolation
    fig3, axes = plt.subplots(1, 2, figsize=(9, 4))
    
    im1 = axes[0].imshow(delta_low_norm, cmap=cmap, interpolation='bilinear', origin='upper')
    axes[0].set_title(r'$|\Delta Z_{low}|$', fontsize=14, fontweight='bold', pad=10)
    axes[0].set_xlabel('Horizontal frequency', fontsize=11)
    axes[0].set_ylabel('Vertical frequency', fontsize=11)
    cbar1 = plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    axes[0].set_xticks([0, delta_low_norm.shape[1]-1])
    axes[0].set_xticklabels(['DC', 'High'])
    axes[0].set_yticks([0, delta_low_norm.shape[0]-1])
    axes[0].set_yticklabels(['DC', 'High'])
    
    im2 = axes[1].imshow(delta_high_norm, cmap=cmap, interpolation='bilinear', origin='upper')
    axes[1].set_title(r'$|\Delta Z_{high}|$', fontsize=14, fontweight='bold', pad=10)
    axes[1].set_xlabel('Horizontal frequency', fontsize=11)
    axes[1].set_ylabel('Vertical frequency', fontsize=11)
    cbar2 = plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
    axes[1].set_xticks([0, delta_high_norm.shape[1]-1])
    axes[1].set_xticklabels(['DC', 'High'])
    axes[1].set_yticks([0, delta_high_norm.shape[0]-1])
    axes[1].set_yticklabels(['DC', 'High'])
    
    plt.tight_layout()
    fig3.savefig(os.path.join(output_dir, 'spectral_perturbation_combined.pdf'), dpi=300, bbox_inches='tight')
    fig3.savefig(os.path.join(output_dir, 'spectral_perturbation_combined.png'), dpi=150, bbox_inches='tight')
    print(f"Saved: spectral_perturbation_combined.pdf/png")
    plt.close(fig3)
    
    # Print statistics
    print("\n=== Perturbation Statistics (DCT domain) ===")
    print(f"|ΔZ_low| - Mean: {delta_low_map.mean():.6f}, Max: {delta_low_map.max():.6f}")
    print(f"|ΔZ_high| - Mean: {delta_high_map.mean():.6f}, Max: {delta_high_map.max():.6f}")
    print(f"|ΔZ_full| - Mean: {delta_full_map.mean():.6f}, Max: {delta_full_map.max():.6f}")
    
    # Analyze spatial distribution in DCT domain
    h, w = delta_low_map.shape
    h_cut, w_cut = h // 2, w // 2
    
    print(f"\nRadial profile peak frequencies:")
    low_peak_idx = np.argmax(radial_profile_low)
    high_peak_idx = np.argmax(radial_profile_high)
    print(f"  Low-freq encoder peak at normalized freq: {bin_centers_norm[low_peak_idx]:.3f}")
    print(f"  High-freq encoder peak at normalized freq: {bin_centers_norm[high_peak_idx]:.3f}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='best res/efficient_20260222_004718/best_model.pth')
    parser.add_argument('--sample_dir', type=str, default='sample_images/hq')
    parser.add_argument('--output_dir', type=str, default='results/spectral_analysis')
    parser.add_argument('--w_dim', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    generate_spectral_heatmaps(
        checkpoint_path=args.checkpoint,
        sample_dir=args.sample_dir,
        output_dir=args.output_dir,
        w_dim=args.w_dim,
        seed=args.seed
    )

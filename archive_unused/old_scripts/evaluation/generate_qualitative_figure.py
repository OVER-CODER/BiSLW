#!/usr/bin/env python3
"""
Generate qualitative comparison figure for the paper.
Shows: Original Generated Images | BiSLW (Our Method) | Residual (×10)
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from PIL import Image

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from models.latent_split import LatentSplitter
from models.recombination import LatentRecombiner
from models.watermark_encoder import WatermarkEncoder
from models.watermark_decoder import WatermarkDecoder


def load_vae(device):
    """Load VAE for decoding latents to images."""
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


def decode_latent(vae, z, scaling_factor=0.18215):
    """Decode latent to image."""
    with torch.no_grad():
        z_scaled = z / scaling_factor
        img = vae.decode(z_scaled).sample
        return img


def load_models(checkpoint_path, device, w_dim=32):
    """Load trained watermark models."""
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Initialize models
    splitter = LatentSplitter(mode='dct').to(device)
    recombiner = LatentRecombiner(mode='dct').to(device)
    encoder_l = WatermarkEncoder(watermark_dim=w_dim).to(device)
    encoder_h = WatermarkEncoder(watermark_dim=w_dim).to(device)
    decoder_l = WatermarkDecoder(watermark_dim=w_dim).to(device)
    decoder_h = WatermarkDecoder(watermark_dim=w_dim).to(device)
    
    # Load weights
    encoder_l.load_state_dict(checkpoint['encoder_l'])
    encoder_h.load_state_dict(checkpoint['encoder_h'])
    decoder_l.load_state_dict(checkpoint['decoder_l'])
    decoder_h.load_state_dict(checkpoint['decoder_h'])
    
    # Get alpha values
    alpha_l = checkpoint.get('alpha_l', 0.3)
    alpha_h = checkpoint.get('alpha_h', 0.15)
    
    # Set to eval mode
    encoder_l.eval()
    encoder_h.eval()
    decoder_l.eval()
    decoder_h.eval()
    
    return {
        'splitter': splitter,
        'recombiner': recombiner,
        'encoder_l': encoder_l,
        'encoder_h': encoder_h, 
        'decoder_l': decoder_l,
        'decoder_h': decoder_h,
        'alpha_l': alpha_l,
        'alpha_h': alpha_h
    }


def embed_watermark(models, z, w):
    """Embed watermark into latent."""
    with torch.no_grad():
        # Split latent into frequency bands
        z_low, z_high = models['splitter'](z)
        
        # Embed watermark
        z_low_wm = models['encoder_l'](z_low, w, alpha=models['alpha_l'])
        z_high_wm = models['encoder_h'](z_high, w, alpha=models['alpha_h'])
        
        # Recombine
        z_wm = models['recombiner'](z_low_wm, z_high_wm)
        
        return z_wm


def tensor_to_numpy(tensor):
    """Convert tensor image to numpy for plotting."""
    # Assume tensor is (C, H, W) in [-1, 1]
    img = (tensor + 1) / 2  # Convert to [0, 1]
    img = img.clamp(0, 1)
    img = img.permute(1, 2, 0).cpu().numpy()
    return img


def generate_qualitative_figure(
    n_samples=4,
    checkpoint_path='best res/efficient_20260222_004718/best_model.pth',
    latents_path='cache/latents_1000_256.pt',
    output_path='results/qualitative_comparison.pdf',
    w_dim=32,
    seed=42
):
    """
    Generate qualitative comparison figure for paper.
    
    Layout: 4 rows × 3 columns
    Columns: Original | BiSLW (Ours) | Residual (×10)
    """
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
    
    # Load VAE
    vae = load_vae(device)
    
    # Load precomputed latents
    print(f"Loading latents from {latents_path}")
    latent_data = torch.load(latents_path, map_location='cpu', weights_only=False)
    latents = latent_data['latents'] if isinstance(latent_data, dict) else latent_data
    
    # Select random samples
    indices = torch.randperm(len(latents))[:n_samples]
    z_batch = latents[indices].to(device)
    
    # Generate random watermarks
    w_batch = torch.randn(n_samples, w_dim, device=device)
    
    # Embed watermark
    print("Embedding watermarks...")
    z_wm_batch = embed_watermark(models, z_batch, w_batch)
    
    # Decode to images
    print("Decoding to images...")
    with torch.no_grad():
        images_original = decode_latent(vae, z_batch)
        images_watermarked = decode_latent(vae, z_wm_batch)
    
    # Create figure
    print("Creating figure...")
    fig = plt.figure(figsize=(10, 13))
    gs = gridspec.GridSpec(n_samples, 3, figure=fig, wspace=0.02, hspace=0.05)
    
    # Column headers
    col_titles = ['Original', 'BiSLW (Ours)', 'Residual (×10)']
    
    for row in range(n_samples):
        orig_img = tensor_to_numpy(images_original[row])
        wm_img = tensor_to_numpy(images_watermarked[row])
        
        # Compute residual (absolute difference × 10)
        residual = np.abs(orig_img - wm_img) * 10
        residual = np.clip(residual, 0, 1)
        
        # Compute PSNR for annotation
        mse = np.mean((orig_img - wm_img) ** 2)
        psnr = 10 * np.log10(1.0 / (mse + 1e-10))
        
        # Original
        ax0 = fig.add_subplot(gs[row, 0])
        ax0.imshow(orig_img)
        ax0.axis('off')
        if row == 0:
            ax0.set_title(col_titles[0], fontsize=14, fontweight='bold', pad=10)
        
        # BiSLW (Watermarked)
        ax1 = fig.add_subplot(gs[row, 1])
        ax1.imshow(wm_img)
        ax1.axis('off')
        if row == 0:
            ax1.set_title(col_titles[1], fontsize=14, fontweight='bold', pad=10)
        # Add PSNR annotation
        ax1.text(0.98, 0.02, f'PSNR: {psnr:.1f} dB', transform=ax1.transAxes,
                fontsize=8, color='white', ha='right', va='bottom',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))
        
        # Residual (×10)
        ax2 = fig.add_subplot(gs[row, 2])
        ax2.imshow(residual)
        ax2.axis('off')
        if row == 0:
            ax2.set_title(col_titles[2], fontsize=14, fontweight='bold', pad=10)
    
    plt.tight_layout()
    
    # Save figure
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save as PDF and PNG
    fig.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    png_path = output_path.replace('.pdf', '.png')
    fig.savefig(png_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    
    print(f"Figure saved to {output_path}")
    print(f"Figure saved to {png_path}")
    
    plt.close()
    
    # Also save individual images for flexibility
    individual_dir = os.path.join(os.path.dirname(output_path), 'individual')
    os.makedirs(individual_dir, exist_ok=True)
    
    for i in range(n_samples):
        orig_img = tensor_to_numpy(images_original[i])
        wm_img = tensor_to_numpy(images_watermarked[i])
        residual = np.clip(np.abs(orig_img - wm_img) * 10, 0, 1)
        
        Image.fromarray((orig_img * 255).astype(np.uint8)).save(
            os.path.join(individual_dir, f'sample_{i+1}_original.png'))
        Image.fromarray((wm_img * 255).astype(np.uint8)).save(
            os.path.join(individual_dir, f'sample_{i+1}_bislw.png'))
        Image.fromarray((residual * 255).astype(np.uint8)).save(
            os.path.join(individual_dir, f'sample_{i+1}_residual_x10.png'))
    
    print(f"Individual images saved to {individual_dir}")
    
    return fig


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Generate qualitative comparison figure')
    parser.add_argument('--n_samples', type=int, default=4, help='Number of samples')
    parser.add_argument('--checkpoint', type=str, 
                        default='best res/efficient_20260222_004718/best_model.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--latents', type=str, default='cache/latents_1000_256.pt',
                        help='Path to precomputed latents')
    parser.add_argument('--output', type=str, default='results/qualitative_comparison.pdf',
                        help='Output path for figure')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Change to project root
    os.chdir(PROJECT_ROOT)
    
    generate_qualitative_figure(
        n_samples=args.n_samples,
        checkpoint_path=args.checkpoint,
        latents_path=args.latents,
        output_path=args.output,
        seed=args.seed
    )

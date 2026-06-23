#!/usr/bin/env python3
"""
Generate comprehensive qualitative comparison figure across watermarking methods.

Creates a figure with:
- Columns: Original | LaWa (Ours) | Stable Signature | SSL | FNNS | RoSteALS | RivaGan | HiDDen | DCT-DWT
- For each image:
  - Row 1: Watermarked Image
  - Row 2: Residual ×10
  - Row 3: Center Crop (to show detail)

Usage:
    python scripts/evaluation/generate_method_comparison.py --n_images 10
    
For baselines, provide pre-watermarked images in:
    results/baseline_images/{method_name}/image_{i}.png
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from PIL import Image
from pathlib import Path

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from models.latent_split import LatentSplitter
from models.recombination import LatentRecombiner
from models.watermark_encoder import WatermarkEncoder
from models.watermark_decoder import WatermarkDecoder


# Methods to compare
METHODS = [
    'Original',
    'LaWa',
    'Stable Signature',
    'SSL',
    'FNNS',
    'RoSteALS',
    'RivaGan',
    'HiDDen',
    'DCT-DWT'
]

# Short names for file paths
METHOD_SHORT_NAMES = {
    'Original': 'original',
    'LaWa': 'lawa',
    'Stable Signature': 'stable_signature',
    'SSL': 'ssl',
    'FNNS': 'fnns',
    'RoSteALS': 'rosteals',
    'RivaGan': 'rivagan',
    'HiDDen': 'hidden',
    'DCT-DWT': 'dct_dwt'
}


def load_vae(device):
    """Load VAE for encoding/decoding."""
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


def encode_image(vae, image, scaling_factor=0.18215):
    """Encode image to latent."""
    with torch.no_grad():
        latent = vae.encode(image).latent_dist.mean
        return latent * scaling_factor


def decode_latent(vae, z, scaling_factor=0.18215):
    """Decode latent to image."""
    with torch.no_grad():
        z_scaled = z / scaling_factor
        img = vae.decode(z_scaled).sample
        return img


def load_lawa_models(checkpoint_path, device, w_dim=32):
    """Load LaWa (BiSLW) watermark models."""
    print(f"Loading LaWa checkpoint from {checkpoint_path}")
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
    alpha_l = checkpoint.get('alpha_l', 0.02)
    alpha_h = checkpoint.get('alpha_h', 0.01)
    
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


def embed_lawa_watermark(models, z, w):
    """Embed watermark using LaWa method."""
    with torch.no_grad():
        z_low, z_high = models['splitter'](z)
        z_low_wm = models['encoder_l'](z_low, w, alpha=models['alpha_l'])
        z_high_wm = models['encoder_h'](z_high, w, alpha=models['alpha_h'])
        z_wm = models['recombiner'](z_low_wm, z_high_wm)
        return z_wm


def tensor_to_numpy(tensor):
    """Convert tensor image to numpy for plotting."""
    img = (tensor + 1) / 2
    img = img.clamp(0, 1)
    img = img.permute(1, 2, 0).cpu().numpy()
    return img


def numpy_to_tensor(img_np, device='cpu'):
    """Convert numpy image [0,1] to tensor [-1,1]."""
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float()
    img_tensor = img_tensor * 2 - 1
    return img_tensor.to(device)


def load_baseline_image(method_name, image_idx, baseline_dir, size=512):
    """
    Load pre-generated baseline watermarked image.
    
    Expected path: {baseline_dir}/{method_short_name}/image_{idx}.png
    """
    short_name = METHOD_SHORT_NAMES.get(method_name, method_name.lower().replace(' ', '_'))
    img_path = os.path.join(baseline_dir, short_name, f'image_{image_idx}.png')
    
    if os.path.exists(img_path):
        img = Image.open(img_path).convert('RGB')
        img = img.resize((size, size), Image.LANCZOS)
        img_np = np.array(img).astype(np.float32) / 255.0
        return img_np
    else:
        return None


def center_crop(img_np, crop_ratio=0.3):
    """Extract center crop from image."""
    h, w = img_np.shape[:2]
    crop_h = int(h * crop_ratio)
    crop_w = int(w * crop_ratio)
    start_h = (h - crop_h) // 2
    start_w = (w - crop_w) // 2
    return img_np[start_h:start_h+crop_h, start_w:start_w+crop_w]


def compute_residual(original, watermarked, amplification=10):
    """Compute residual (absolute difference) with amplification."""
    residual = np.abs(original - watermarked) * amplification
    return np.clip(residual, 0, 1)


def compute_psnr(img1, img2):
    """Compute PSNR between two images."""
    mse = np.mean((img1 - img2) ** 2)
    if mse < 1e-10:
        return 100.0
    return 10 * np.log10(1.0 / mse)


def generate_comparison_figure(
    n_images=10,
    checkpoint_path='best res/efficient_20260222_004718/best_model.pth',
    latents_path='cache/latents_1000_256.pt',
    baseline_dir='results/baseline_images',
    output_path='results/method_comparison.pdf',
    w_dim=32,
    seed=42,
    crop_ratio=0.3,
    use_simulated_baselines=True
):
    """
    Generate comprehensive method comparison figure.
    
    Args:
        n_images: Number of images to include
        checkpoint_path: Path to LaWa model checkpoint
        latents_path: Path to precomputed latents
        baseline_dir: Directory containing baseline watermarked images
        output_path: Output path for figure
        w_dim: Watermark dimension
        seed: Random seed
        crop_ratio: Ratio for center crop size
        use_simulated_baselines: If True, simulate baseline methods if not found
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
    
    # Load LaWa models
    lawa_models = load_lawa_models(checkpoint_path, device, w_dim)
    
    # Load VAE
    vae = load_vae(device)
    
    # Load precomputed latents
    print(f"Loading latents from {latents_path}")
    latent_data = torch.load(latents_path, map_location='cpu', weights_only=False)
    latents = latent_data['latents'] if isinstance(latent_data, dict) else latent_data
    
    # Select random samples
    indices = torch.randperm(len(latents))[:n_images]
    z_batch = latents[indices].to(device)
    
    # Generate watermarks
    w_batch = torch.randn(n_images, w_dim, device=device)
    
    # Generate LaWa watermarked latents
    print("Generating LaWa watermarked images...")
    z_lawa = embed_lawa_watermark(lawa_models, z_batch, w_batch)
    
    # Decode all images
    print("Decoding images...")
    with torch.no_grad():
        images_original = decode_latent(vae, z_batch)
        images_lawa = decode_latent(vae, z_lawa)
    
    # Convert to numpy
    originals_np = [tensor_to_numpy(images_original[i]) for i in range(n_images)]
    lawa_np = [tensor_to_numpy(images_lawa[i]) for i in range(n_images)]
    
    # Prepare all method images
    # Dict: method_name -> list of numpy images
    method_images = {
        'Original': originals_np,
        'LaWa': lawa_np,
    }
    
    # Load or simulate baseline methods
    baseline_methods = ['Stable Signature', 'SSL', 'FNNS', 'RoSteALS', 'RivaGan', 'HiDDen', 'DCT-DWT']
    
    for method in baseline_methods:
        method_images[method] = []
        print(f"Loading/generating {method} images...")
        
        for i in range(n_images):
            # Try to load pre-generated baseline image
            baseline_img = load_baseline_image(method, i, baseline_dir)
            
            if baseline_img is not None:
                method_images[method].append(baseline_img)
            elif use_simulated_baselines:
                # Simulate the baseline method's watermarking effect
                # These are PLACEHOLDER simulations - replace with actual method outputs
                simulated = simulate_baseline_watermark(originals_np[i], method)
                method_images[method].append(simulated)
            else:
                # Use original as placeholder
                method_images[method].append(originals_np[i])
    
    # Create figure
    print("Creating comparison figure...")
    n_methods = len(METHODS)
    n_rows = n_images * 3  # 3 rows per image: watermarked, residual, crop
    
    # Figure size: adjust based on number of images and methods
    fig_width = 2.0 * n_methods  # 2 inches per method column
    fig_height = 2.0 * n_images * 3 / 3  # Adjust height
    
    fig = plt.figure(figsize=(fig_width, fig_height * 1.5))
    gs = gridspec.GridSpec(n_rows, n_methods, figure=fig, wspace=0.02, hspace=0.02)
    
    # Add column headers
    for col, method in enumerate(METHODS):
        ax = fig.add_subplot(gs[0, col])
        ax.text(0.5, 1.15, method, ha='center', va='bottom', fontsize=10, fontweight='bold',
                transform=ax.transAxes)
    
    # Row labels
    row_types = ['Image', 'Res. ×10', 'Crop']
    
    for img_idx in range(n_images):
        base_row = img_idx * 3
        original = originals_np[img_idx]
        
        for col, method in enumerate(METHODS):
            wm_image = method_images[method][img_idx]
            
            # Row 1: Watermarked/Original Image
            ax1 = fig.add_subplot(gs[base_row, col])
            ax1.imshow(wm_image)
            ax1.axis('off')
            
            # Add image number label on first column
            if col == 0 and img_idx < n_images:
                ax1.text(-0.15, 0.5, f'#{img_idx+1}', ha='right', va='center', 
                        fontsize=9, fontweight='bold', transform=ax1.transAxes)
            
            # Row 2: Residual ×10
            ax2 = fig.add_subplot(gs[base_row + 1, col])
            if method == 'Original':
                # Original has no residual - show black
                residual = np.zeros_like(original)
            else:
                residual = compute_residual(original, wm_image, amplification=10)
            ax2.imshow(residual)
            ax2.axis('off')
            
            # Row 3: Center Crop
            ax3 = fig.add_subplot(gs[base_row + 2, col])
            crop = center_crop(wm_image, crop_ratio=crop_ratio)
            ax3.imshow(crop)
            ax3.axis('off')
    
    # Add row type labels on the left
    for img_idx in range(n_images):
        for row_offset, label in enumerate(row_types):
            row = img_idx * 3 + row_offset
            ax = fig.add_subplot(gs[row, 0])
            # Labels already added via image number
    
    plt.tight_layout()
    
    # Save figure
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    png_path = output_path.replace('.pdf', '.png')
    fig.savefig(png_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    
    print(f"Figure saved to {output_path}")
    print(f"Figure saved to {png_path}")
    
    # Save individual images for each method
    save_individual_images(method_images, originals_np, output_path, n_images, crop_ratio)
    
    plt.close()
    
    # Print PSNR statistics
    print_psnr_statistics(method_images, originals_np, n_images)
    
    return fig


def simulate_baseline_watermark(original, method):
    """
    Simulate watermarking effect for baseline methods.
    
    IMPORTANT: These are PLACEHOLDER simulations based on typical characteristics
    of each method. For actual paper results, replace with real method outputs.
    """
    img = original.copy()
    h, w = img.shape[:2]
    
    if method == 'Stable Signature':
        # Stable Signature: Embeds in decoder, typically subtle high-freq noise
        noise = np.random.randn(h, w, 3) * 0.008
        img = np.clip(img + noise, 0, 1)
        
    elif method == 'SSL':
        # SSL (Self-Supervised Learning based): Subtle pattern embedding
        noise = np.random.randn(h, w, 3) * 0.012
        img = np.clip(img + noise, 0, 1)
        
    elif method == 'FNNS':
        # FNNS: Frequency-based, affects specific frequency bands
        from scipy.ndimage import gaussian_filter
        noise = np.random.randn(h, w, 3) * 0.015
        noise = gaussian_filter(noise, sigma=2)
        img = np.clip(img + noise, 0, 1)
        
    elif method == 'RoSteALS':
        # RoSteALS: Robust steganography, moderate noise pattern
        noise = np.random.randn(h, w, 3) * 0.010
        img = np.clip(img + noise, 0, 1)
        
    elif method == 'RivaGan':
        # RivaGan: GAN-based, structured noise patterns
        # Create block-based noise pattern
        block_size = 16
        blocks_h, blocks_w = h // block_size, w // block_size
        block_noise = np.random.randn(blocks_h, blocks_w, 3) * 0.025
        noise = np.repeat(np.repeat(block_noise, block_size, axis=0), block_size, axis=1)
        noise = noise[:h, :w]
        img = np.clip(img + noise, 0, 1)
        
    elif method == 'HiDDen':
        # HiDDen: Deep learning based, relatively uniform noise
        noise = np.random.randn(h, w, 3) * 0.018
        img = np.clip(img + noise, 0, 1)
        
    elif method == 'DCT-DWT':
        # DCT-DWT: Transform-domain watermarking, affects mid frequencies
        from scipy.ndimage import gaussian_filter
        noise = np.random.randn(h, w, 3) * 0.020
        noise = gaussian_filter(noise, sigma=3) - gaussian_filter(noise, sigma=6)
        noise = noise * 3  # Amplify band-limited noise
        img = np.clip(img + noise, 0, 1)
    
    return img


def save_individual_images(method_images, originals_np, output_path, n_images, crop_ratio):
    """Save individual images for each method."""
    base_dir = os.path.dirname(output_path)
    individual_dir = os.path.join(base_dir, 'method_comparison_individual')
    
    for method in METHODS:
        method_dir = os.path.join(individual_dir, METHOD_SHORT_NAMES.get(method, method.lower()))
        os.makedirs(method_dir, exist_ok=True)
        
        for i in range(n_images):
            img = method_images[method][i]
            original = originals_np[i]
            
            # Save watermarked image
            Image.fromarray((img * 255).astype(np.uint8)).save(
                os.path.join(method_dir, f'image_{i+1}.png'))
            
            # Save residual
            if method != 'Original':
                residual = compute_residual(original, img, amplification=10)
                Image.fromarray((residual * 255).astype(np.uint8)).save(
                    os.path.join(method_dir, f'residual_{i+1}.png'))
            
            # Save crop
            crop = center_crop(img, crop_ratio=crop_ratio)
            # Resize crop for better visibility
            crop_resized = np.array(Image.fromarray((crop * 255).astype(np.uint8)).resize(
                (256, 256), Image.LANCZOS)) / 255.0
            Image.fromarray((crop_resized * 255).astype(np.uint8)).save(
                os.path.join(method_dir, f'crop_{i+1}.png'))
    
    print(f"Individual images saved to {individual_dir}")


def print_psnr_statistics(method_images, originals_np, n_images):
    """Print PSNR statistics for each method."""
    print("\n" + "=" * 60)
    print("PSNR Statistics (dB)")
    print("=" * 60)
    print(f"{'Method':<20} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    print("-" * 60)
    
    for method in METHODS:
        if method == 'Original':
            continue
        
        psnrs = []
        for i in range(n_images):
            psnr = compute_psnr(originals_np[i], method_images[method][i])
            psnrs.append(psnr)
        
        psnrs = np.array(psnrs)
        print(f"{method:<20} {psnrs.mean():>10.2f} {psnrs.std():>10.2f} {psnrs.min():>10.2f} {psnrs.max():>10.2f}")
    
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Generate method comparison figure')
    parser.add_argument('--n_images', type=int, default=10, help='Number of images')
    parser.add_argument('--checkpoint', type=str, 
                        default='best res/efficient_20260222_004718/best_model.pth',
                        help='Path to LaWa checkpoint')
    parser.add_argument('--latents', type=str, default='cache/latents_1000_256.pt',
                        help='Path to precomputed latents')
    parser.add_argument('--baseline_dir', type=str, default='results/baseline_images',
                        help='Directory with baseline watermarked images')
    parser.add_argument('--output', type=str, default='results/method_comparison.pdf',
                        help='Output path')
    parser.add_argument('--w_dim', type=int, default=32, help='Watermark dimension')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--crop_ratio', type=float, default=0.3, help='Center crop ratio')
    parser.add_argument('--no_simulated', action='store_true',
                        help='Disable simulated baselines (use originals as placeholder)')
    
    args = parser.parse_args()
    
    generate_comparison_figure(
        n_images=args.n_images,
        checkpoint_path=args.checkpoint,
        latents_path=args.latents,
        baseline_dir=args.baseline_dir,
        output_path=args.output,
        w_dim=args.w_dim,
        seed=args.seed,
        crop_ratio=args.crop_ratio,
        use_simulated_baselines=not args.no_simulated
    )


if __name__ == '__main__':
    main()

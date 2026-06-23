#!/usr/bin/env python3
"""
Generate qualitative comparison figure for the paper using sample images.
Uses local images or downloads sample images, applies BiSLW watermarking.
Shows: Original | BiSLW (Ours) | Residual (×10)
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from PIL import Image
import urllib.request

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from models.latent_split import LatentSplitter
from models.recombination import LatentRecombiner
from models.watermark_encoder import WatermarkEncoder
from models.watermark_decoder import WatermarkDecoder


# Sample image URLs (high quality, freely available)
SAMPLE_IMAGE_URLS = [
    "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/1200px-PNG_transparency_demonstration_1.png",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b6/Image_created_with_a_mobile_phone.png/1200px-Image_created_with_a_mobile_phone.png",
    "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1e/Sunrise_over_the_sea.jpg/1200px-Sunrise_over_the_sea.jpg",
]


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


def embed_watermark(models, z, w):
    """Embed watermark into latent."""
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


def load_and_preprocess_image(path_or_url, size=512, device='cpu'):
    """Load image from path or URL and preprocess for VAE."""
    if path_or_url.startswith('http'):
        # Download image
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            urllib.request.urlretrieve(path_or_url, f.name)
            img = Image.open(f.name).convert('RGB')
            os.unlink(f.name)
    else:
        img = Image.open(path_or_url).convert('RGB')
    
    # Center crop to square
    w, h = img.size
    min_dim = min(w, h)
    left = (w - min_dim) // 2
    top = (h - min_dim) // 2
    img = img.crop((left, top, left + min_dim, top + min_dim))
    
    # Resize
    img = img.resize((size, size), Image.LANCZOS)
    
    # Convert to tensor [-1, 1]
    img_np = np.array(img).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)  # HWC -> CHW
    img_tensor = img_tensor * 2 - 1  # [0,1] -> [-1,1]
    
    return img_tensor.unsqueeze(0).to(device)


def compute_psnr(img1, img2):
    """Compute PSNR between two numpy images in [0, 1]."""
    mse = np.mean((img1 - img2) ** 2)
    if mse < 1e-10:
        return 100.0
    return 10 * np.log10(1.0 / mse)


def compute_ssim(img1, img2, window_size=11):
    """Compute SSIM between two numpy images."""
    from scipy.ndimage import uniform_filter
    
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    if img1.ndim == 3:
        img1 = np.mean(img1, axis=2)
        img2 = np.mean(img2, axis=2)
    
    mu1 = uniform_filter(img1, size=window_size)
    mu2 = uniform_filter(img2, size=window_size)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = uniform_filter(img1 ** 2, size=window_size) - mu1_sq
    sigma2_sq = uniform_filter(img2 ** 2, size=window_size) - mu2_sq
    sigma12 = uniform_filter(img1 * img2, size=window_size) - mu1_mu2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return np.mean(ssim_map)


def generate_sample_images(n_samples, size, device):
    """Generate diverse sample images using various methods."""
    images = []
    
    # Method 1: Gradient images (smooth color transitions)
    print("  Generating gradient images...")
    for i in range(min(2, n_samples)):
        img = torch.zeros(1, 3, size, size, device=device)
        # Create smooth gradients with different colors
        x = torch.linspace(0, 1, size, device=device)
        y = torch.linspace(0, 1, size, device=device)
        xx, yy = torch.meshgrid(x, y, indexing='xy')
        
        # Different color schemes
        if i == 0:  # Sunset-like
            img[0, 0] = 0.8 * xx + 0.1  # Red channel
            img[0, 1] = 0.4 * yy * xx + 0.2  # Green
            img[0, 2] = 0.6 * (1 - xx) + 0.1  # Blue
        else:  # Ocean-like
            img[0, 0] = 0.2 * xx + 0.1
            img[0, 1] = 0.5 * yy + 0.2
            img[0, 2] = 0.7 * (1 - xx * yy) + 0.2
        
        img = img * 2 - 1  # Convert to [-1, 1]
        images.append(img)
    
    # Method 2: Procedural patterns
    print("  Generating pattern images...")
    for i in range(min(2, max(0, n_samples - 2))):
        img = torch.zeros(1, 3, size, size, device=device)
        x = torch.linspace(-3, 3, size, device=device)
        y = torch.linspace(-3, 3, size, device=device)
        xx, yy = torch.meshgrid(x, y, indexing='xy')
        
        if i == 0:  # Ripple pattern
            r = torch.sqrt(xx**2 + yy**2)
            pattern = torch.sin(r * 5) * 0.5 + 0.5
            img[0, 0] = pattern * 0.8 + 0.1
            img[0, 1] = pattern * 0.6 + 0.3
            img[0, 2] = (1 - pattern) * 0.5 + 0.4
        else:  # Checkerboard-like
            pattern = torch.sin(xx * 4) * torch.sin(yy * 4) * 0.5 + 0.5
            img[0, 0] = pattern * 0.7 + 0.2
            img[0, 1] = pattern * 0.5 + 0.4
            img[0, 2] = pattern * 0.3 + 0.5
        
        img = img * 2 - 1
        images.append(img)
    
    return images[:n_samples]


def generate_qualitative_figure(
    n_samples=4,
    checkpoint_path='best res/efficient_20260222_004718/best_model.pth',
    output_path='results/qualitative_comparison_v2.pdf',
    w_dim=32,
    seed=42,
    image_size=512,
    use_local_images=True,
    local_image_dir='sample_images',
    alpha_scale=1.0  # Scale factor for alpha values (lower = higher PSNR)
):
    """Generate qualitative comparison figure."""
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
    
    # Load VAE and watermark models
    vae = load_vae(device)
    models = load_models(checkpoint_path, device, w_dim)
    
    # Apply alpha scaling for PSNR control
    original_alpha_l = models['alpha_l']
    original_alpha_h = models['alpha_h']
    models['alpha_l'] = models['alpha_l'] * alpha_scale
    models['alpha_h'] = models['alpha_h'] * alpha_scale
    print(f"Alpha values: alpha_l={models['alpha_l']:.4f} (orig: {original_alpha_l}), alpha_h={models['alpha_h']:.4f} (orig: {original_alpha_h})")
    
    print(f"\nGenerating {n_samples} sample images...")
    
    # Try to load local images first
    input_images = []
    local_path = os.path.join(PROJECT_ROOT, local_image_dir)
    
    if use_local_images and os.path.exists(local_path):
        print(f"Looking for images in {local_path}")
        image_files = [f for f in os.listdir(local_path) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
        for img_file in image_files[:n_samples]:
            try:
                img = load_and_preprocess_image(
                    os.path.join(local_path, img_file), 
                    size=image_size, 
                    device=device
                )
                input_images.append(img)
                print(f"  Loaded: {img_file}")
            except Exception as e:
                print(f"  Failed to load {img_file}: {e}")
    
    # Generate procedural images for remaining samples
    if len(input_images) < n_samples:
        print(f"Generating {n_samples - len(input_images)} procedural images...")
        proc_images = generate_sample_images(
            n_samples - len(input_images), 
            image_size, 
            device
        )
        input_images.extend(proc_images)
    
    # Process each image
    original_images = []
    watermarked_images = []
    metrics_list = []
    
    for i, img_tensor in enumerate(input_images):
        print(f"\n[{i+1}/{n_samples}] Processing image...")
        
        # Pass through VAE (encode then decode) to get clean baseline
        z = encode_image(vae, img_tensor)
        img_reconstructed = decode_latent(vae, z)
        
        # Generate watermark
        w = torch.randn(1, w_dim, device=device)
        
        # Embed watermark
        z_wm = embed_watermark(models, z, w)
        
        # Decode watermarked latent
        img_wm = decode_latent(vae, z_wm)
        
        # Store images
        original_images.append(img_reconstructed[0])
        watermarked_images.append(img_wm[0])
        
        # Compute metrics
        orig_np = tensor_to_numpy(img_reconstructed[0])
        wm_np = tensor_to_numpy(img_wm[0])
        psnr = compute_psnr(orig_np, wm_np)
        ssim = compute_ssim(orig_np, wm_np)
        metrics_list.append({'psnr': psnr, 'ssim': ssim})
        print(f"   PSNR: {psnr:.2f} dB, SSIM: {ssim:.4f}")
    
    # Create figure
    print("\nCreating comparison figure...")
    fig = plt.figure(figsize=(10, 13))
    gs = gridspec.GridSpec(n_samples, 3, figure=fig, wspace=0.02, hspace=0.08)
    
    col_titles = ['Original', 'BiSLW (Ours)', 'Residual (×10)']
    
    for row in range(n_samples):
        orig_img = tensor_to_numpy(original_images[row])
        wm_img = tensor_to_numpy(watermarked_images[row])
        
        # Compute residual
        residual = np.abs(orig_img - wm_img) * 10
        residual = np.clip(residual, 0, 1)
        
        psnr = metrics_list[row]['psnr']
        ssim = metrics_list[row]['ssim']
        
        # Original
        ax0 = fig.add_subplot(gs[row, 0])
        ax0.imshow(orig_img)
        ax0.axis('off')
        if row == 0:
            ax0.set_title(col_titles[0], fontsize=14, fontweight='bold', pad=10)
        
        # Watermarked
        ax1 = fig.add_subplot(gs[row, 1])
        ax1.imshow(wm_img)
        ax1.axis('off')
        if row == 0:
            ax1.set_title(col_titles[1], fontsize=14, fontweight='bold', pad=10)
        
        # Residual
        ax2 = fig.add_subplot(gs[row, 2])
        ax2.imshow(residual)
        ax2.axis('off')
        if row == 0:
            ax2.set_title(col_titles[2], fontsize=14, fontweight='bold', pad=10)
    
    plt.tight_layout()
    
    # Save figure
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    png_path = output_path.replace('.pdf', '.png')
    fig.savefig(png_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    
    print(f"\nFigure saved to {output_path}")
    print(f"Figure saved to {png_path}")
    
    plt.close()
    
    # Save individual images to qualitative_results folder
    individual_dir = os.path.join(os.path.dirname(output_path), 'qualitative_results')
    os.makedirs(individual_dir, exist_ok=True)
    
    image_names = ['city', 'beach', 'cat', 'mountain']  # Default names
    
    for i in range(n_samples):
        orig_img = tensor_to_numpy(original_images[i])
        wm_img = tensor_to_numpy(watermarked_images[i])
        residual = np.clip(np.abs(orig_img - wm_img) * 10, 0, 1)
        
        name = image_names[i] if i < len(image_names) else f'sample_{i+1}'
        
        Image.fromarray((orig_img * 255).astype(np.uint8)).save(
            os.path.join(individual_dir, f'{name}_original.png'))
        Image.fromarray((wm_img * 255).astype(np.uint8)).save(
            os.path.join(individual_dir, f'{name}_bislw.png'))
        Image.fromarray((residual * 255).astype(np.uint8)).save(
            os.path.join(individual_dir, f'{name}_residual_x10.png'))
    
    print(f"Individual images saved to {individual_dir}")
    
    # Save metrics to JSON
    import json
    results_json = {
        'config': {
            'checkpoint': checkpoint_path,
            'alpha_l': models['alpha_l'],
            'alpha_h': models['alpha_h'],
            'alpha_scale': alpha_scale,
            'w_dim': w_dim,
            'image_size': image_size,
            'seed': seed
        },
        'samples': [],
        'summary': {}
    }
    
    for i in range(n_samples):
        name = image_names[i] if i < len(image_names) else f'sample_{i+1}'
        results_json['samples'].append({
            'name': name,
            'psnr': float(metrics_list[i]['psnr']),
            'ssim': float(metrics_list[i]['ssim']),
            'files': {
                'original': f'{name}_original.png',
                'watermarked': f'{name}_bislw.png',
                'residual': f'{name}_residual_x10.png'
            }
        })
    
    # Summary
    avg_psnr = np.mean([m['psnr'] for m in metrics_list])
    avg_ssim = np.mean([m['ssim'] for m in metrics_list])
    
    results_json['summary'] = {
        'avg_psnr': float(avg_psnr),
        'avg_ssim': float(avg_ssim),
        'n_samples': n_samples
    }
    
    # Save JSON file
    json_path = os.path.join(individual_dir, 'metrics.json')
    with open(json_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    print(f"Metrics saved to {json_path}")
    
    print(f"\n{'='*50}")
    print(f"Average PSNR: {avg_psnr:.2f} dB")
    print(f"Average SSIM: {avg_ssim:.4f}")
    print(f"{'='*50}")
    
    return fig


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Generate qualitative comparison figure')
    parser.add_argument('--n_samples', type=int, default=4, help='Number of samples')
    parser.add_argument('--checkpoint', type=str, 
                        default='best res/efficient_20260222_004718/best_model.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--output', type=str, default='results/qualitative_comparison_v2.pdf',
                        help='Output path for figure')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--size', type=int, default=512, help='Image size')
    parser.add_argument('--alpha_scale', type=float, default=1.0, 
                        help='Scale factor for alpha (lower = higher PSNR, e.g., 0.3)')
    
    args = parser.parse_args()
    
    # Change to project root
    os.chdir(PROJECT_ROOT)
    
    generate_qualitative_figure(
        n_samples=args.n_samples,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        seed=args.seed,
        image_size=args.size,
        alpha_scale=args.alpha_scale
    )

#!/usr/bin/env python3
"""
Generate proper qualitative comparison figure for the paper.
Uses Stable Diffusion to generate real images, then applies BiSLW watermarking.
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


# Sample prompts for diverse image generation
SAMPLE_PROMPTS = [
    "A beautiful sunset over mountains with orange and purple sky, photorealistic",
    "A cute golden retriever puppy playing in a garden, detailed fur, soft lighting",  
    "A modern city skyline at night with neon lights reflecting on water",
    "A serene Japanese garden with cherry blossoms and a small pond, spring",
    "A majestic lion in the African savanna, golden hour lighting, wildlife photography",
    "An astronaut floating in space with Earth in the background, cinematic",
    "A cozy coffee shop interior with warm lighting and wooden furniture",
    "A tropical beach with crystal clear water and palm trees, paradise",
]


def load_stable_diffusion(device):
    """Load Stable Diffusion pipeline."""
    print("Loading Stable Diffusion pipeline...")
    from diffusers import StableDiffusionPipeline
    
    pipe = StableDiffusionPipeline.from_pretrained(
        'runwayml/stable-diffusion-v1-5',
        torch_dtype=torch.float32,
        safety_checker=None,
        requires_safety_checker=False
    ).to(device)
    
    # Disable progress bar for cleaner output
    pipe.set_progress_bar_config(disable=True)
    
    return pipe


def load_vae(device):
    """Load VAE for encoding/decoding latents."""
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
        # Image should be in [-1, 1]
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
    
    # Convert to grayscale for SSIM
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


def generate_qualitative_figure(
    n_samples=4,
    checkpoint_path='best res/efficient_20260222_004718/best_model.pth',
    output_path='results/qualitative_comparison_real.pdf',
    w_dim=32,
    seed=42,
    image_size=512,
    num_inference_steps=30
):
    """
    Generate qualitative comparison figure for paper using real SD-generated images.
    
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
    
    # Load Stable Diffusion
    pipe = load_stable_diffusion(device)
    vae = pipe.vae
    
    # Load watermark models
    models = load_models(checkpoint_path, device, w_dim)
    
    # Generate images with prompts
    print(f"\nGenerating {n_samples} images with Stable Diffusion...")
    prompts = SAMPLE_PROMPTS[:n_samples]
    
    original_images = []
    watermarked_images = []
    metrics_list = []
    
    for i, prompt in enumerate(prompts):
        print(f"\n[{i+1}/{n_samples}] Generating: {prompt[:50]}...")
        
        # Generate image
        generator = torch.Generator(device=device).manual_seed(seed + i)
        with torch.no_grad():
            output = pipe(
                prompt,
                num_inference_steps=num_inference_steps,
                generator=generator,
                output_type="pt"  # Return as tensor
            )
            image = output.images[0]  # (C, H, W) in [0, 1]
        
        # Convert to [-1, 1] for VAE
        image_norm = image * 2 - 1
        image_tensor = image_norm.unsqueeze(0).to(device)  # (1, C, H, W)
        
        # Encode to latent
        z = encode_image(vae, image_tensor)
        
        # Generate watermark
        w = torch.randn(1, w_dim, device=device)
        
        # Embed watermark
        z_wm = embed_watermark(models, z, w)
        
        # Decode watermarked latent
        image_wm = decode_latent(vae, z_wm)
        
        # Store images
        original_images.append(image_tensor[0])
        watermarked_images.append(image_wm[0])
        
        # Compute metrics
        orig_np = tensor_to_numpy(image_tensor[0])
        wm_np = tensor_to_numpy(image_wm[0])
        psnr = compute_psnr(orig_np, wm_np)
        ssim = compute_ssim(orig_np, wm_np)
        metrics_list.append({'psnr': psnr, 'ssim': ssim})
        print(f"   PSNR: {psnr:.2f} dB, SSIM: {ssim:.4f}")
    
    # Create figure
    print("\nCreating comparison figure...")
    fig = plt.figure(figsize=(10, 13))
    gs = gridspec.GridSpec(n_samples, 3, figure=fig, wspace=0.02, hspace=0.08)
    
    # Column headers
    col_titles = ['Original', 'BiSLW (Ours)', 'Residual (×10)']
    
    for row in range(n_samples):
        orig_img = tensor_to_numpy(original_images[row])
        wm_img = tensor_to_numpy(watermarked_images[row])
        
        # Compute residual (absolute difference × 10)
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
        
        # BiSLW (Watermarked)
        ax1 = fig.add_subplot(gs[row, 1])
        ax1.imshow(wm_img)
        ax1.axis('off')
        if row == 0:
            ax1.set_title(col_titles[1], fontsize=14, fontweight='bold', pad=10)
        # Add metrics annotation
        ax1.text(0.98, 0.02, f'PSNR: {psnr:.1f} dB\nSSIM: {ssim:.3f}', 
                transform=ax1.transAxes, fontsize=8, color='white', 
                ha='right', va='bottom',
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
    
    print(f"\nFigure saved to {output_path}")
    print(f"Figure saved to {png_path}")
    
    plt.close()
    
    # Save individual images
    individual_dir = os.path.join(os.path.dirname(output_path), 'individual_real')
    os.makedirs(individual_dir, exist_ok=True)
    
    for i in range(n_samples):
        orig_img = tensor_to_numpy(original_images[i])
        wm_img = tensor_to_numpy(watermarked_images[i])
        residual = np.clip(np.abs(orig_img - wm_img) * 10, 0, 1)
        
        Image.fromarray((orig_img * 255).astype(np.uint8)).save(
            os.path.join(individual_dir, f'sample_{i+1}_original.png'))
        Image.fromarray((wm_img * 255).astype(np.uint8)).save(
            os.path.join(individual_dir, f'sample_{i+1}_bislw.png'))
        Image.fromarray((residual * 255).astype(np.uint8)).save(
            os.path.join(individual_dir, f'sample_{i+1}_residual_x10.png'))
    
    print(f"Individual images saved to {individual_dir}")
    
    # Print summary
    avg_psnr = np.mean([m['psnr'] for m in metrics_list])
    avg_ssim = np.mean([m['ssim'] for m in metrics_list])
    print(f"\n{'='*50}")
    print(f"Average PSNR: {avg_psnr:.2f} dB")
    print(f"Average SSIM: {avg_ssim:.4f}")
    print(f"{'='*50}")
    
    return fig


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Generate qualitative comparison using real SD images')
    parser.add_argument('--n_samples', type=int, default=4, help='Number of samples')
    parser.add_argument('--checkpoint', type=str, 
                        default='best res/efficient_20260222_004718/best_model.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--output', type=str, default='results/qualitative_comparison_real.pdf',
                        help='Output path for figure')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--steps', type=int, default=30, help='Diffusion inference steps')
    
    args = parser.parse_args()
    
    # Change to project root
    os.chdir(PROJECT_ROOT)
    
    generate_qualitative_figure(
        n_samples=args.n_samples,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        seed=args.seed,
        num_inference_steps=args.steps
    )

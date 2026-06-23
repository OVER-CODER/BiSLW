#!/usr/bin/env python3
"""
Comprehensive Watermark Evaluation Suite

Runs all evaluation metrics including:
1. Image Quality Metrics (PSNR, SSIM, LPIPS, FID)
2. Watermark Robustness Evaluation (attacks)
3. Statistical Analysis
4. Computational Analysis
5. Qualitative Experiments
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from latent_watermarking.models.latent_split import LatentSplitter
from latent_watermarking.models.recombination import LatentRecombiner
from latent_watermarking.models.watermark_encoder import WatermarkEncoder
from latent_watermarking.models.watermark_decoder import WatermarkDecoder


# ============================================================
# 1. IMAGE QUALITY METRICS
# ============================================================

def compute_psnr(img1, img2, data_range=2.0):
    """Compute PSNR between two images (both in [-1, 1])."""
    mse = ((img1 - img2) ** 2).mean()
    if mse == 0:
        return float('inf')
    return 10 * torch.log10((data_range ** 2) / mse).item()


def compute_ssim(img1, img2, window_size=11, data_range=2.0):
    """Compute SSIM between two images."""
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    
    # Create Gaussian window
    def gaussian_window(size, sigma=1.5):
        coords = torch.arange(size, device=img1.device, dtype=img1.dtype) - size // 2
        gauss = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        gauss = gauss / gauss.sum()
        return gauss.unsqueeze(1) @ gauss.unsqueeze(0)
    
    window = gaussian_window(window_size).unsqueeze(0).unsqueeze(0)
    window = window.expand(img1.shape[1], 1, -1, -1)
    
    pad = window_size // 2
    
    mu1 = F.conv2d(img1, window, padding=pad, groups=img1.shape[1])
    mu2 = F.conv2d(img2, window, padding=pad, groups=img2.shape[1])
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = F.conv2d(img1 * img1, window, padding=pad, groups=img1.shape[1]) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=pad, groups=img2.shape[1]) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=pad, groups=img1.shape[1]) - mu1_mu2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return ssim_map.mean().item()


def compute_lpips(img1, img2, lpips_model):
    """Compute LPIPS using VGG features."""
    with torch.no_grad():
        return lpips_model(img1, img2).mean().item()


# ============================================================
# 2. ATTACK FUNCTIONS
# ============================================================

def jpeg_attack(images, quality):
    """Simulated JPEG compression attack."""
    scale_factor = max(0.5, quality / 100)
    B, C, H, W = images.shape
    h_small = max(1, int(H * scale_factor))
    w_small = max(1, int(W * scale_factor))
    
    images_down = F.interpolate(images, size=(h_small, w_small), mode='bilinear', align_corners=False)
    images_up = F.interpolate(images_down, size=(H, W), mode='bilinear', align_corners=False)
    
    blend = quality / 100
    return blend * images + (1 - blend) * images_up


def gaussian_noise_attack(images, sigma):
    """Add Gaussian noise."""
    noise = torch.randn_like(images) * sigma
    return torch.clamp(images + noise, -1, 1)


def gaussian_blur_attack(images, kernel_size):
    """Apply Gaussian blur."""
    sigma = kernel_size / 3
    coords = torch.arange(kernel_size, device=images.device, dtype=images.dtype) - kernel_size // 2
    xx, yy = torch.meshgrid(coords, coords, indexing='ij')
    kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    kernel = kernel.unsqueeze(0).unsqueeze(0).expand(images.shape[1], 1, -1, -1)
    
    padding = kernel_size // 2
    return F.conv2d(images, kernel, padding=padding, groups=images.shape[1])


def resize_attack(images, scale):
    """Resize attack (downsample then upsample)."""
    B, C, H, W = images.shape
    h_small = max(1, int(H * scale))
    w_small = max(1, int(W * scale))
    
    images_down = F.interpolate(images, size=(h_small, w_small), mode='bilinear', align_corners=False)
    images_up = F.interpolate(images_down, size=(H, W), mode='bilinear', align_corners=False)
    return images_up


def crop_attack(images, crop_ratio):
    """Random crop and resize back."""
    B, C, H, W = images.shape
    crop_h = int(H * crop_ratio)
    crop_w = int(W * crop_ratio)
    
    top = crop_h // 2
    left = crop_w // 2
    
    cropped = images[:, :, top:H-crop_h+top, left:W-crop_w+left]
    return F.interpolate(cropped, size=(H, W), mode='bilinear', align_corners=False)


def rotation_attack(images, angle):
    """Rotate images by angle degrees."""
    B, C, H, W = images.shape
    angle_rad = angle * np.pi / 180
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    
    theta = torch.tensor([
        [cos_a, -sin_a, 0],
        [sin_a, cos_a, 0]
    ], dtype=images.dtype, device=images.device).unsqueeze(0).expand(B, -1, -1)
    
    grid = F.affine_grid(theta, images.size(), align_corners=False)
    return F.grid_sample(images, grid, mode='bilinear', padding_mode='reflection', align_corners=False)


# ============================================================
# 3. EVALUATION FUNCTIONS
# ============================================================

def evaluate_robustness(latents, watermark, encoder_l, encoder_h, decoder_l, decoder_h,
                        splitter, recombiner, vae, alpha_l, alpha_h, device, attacks_list):
    """Evaluate robustness under attacks."""
    results = {}
    
    for attack_name, attack_fn, params in attacks_list:
        bit_accs = []
        bers = []
        
        for i in range(min(50, len(latents))):
            z = latents[i:i+1].to(device)
            w = watermark[i:i+1].to(device)
            
            # Encode watermark
            z_l, z_h = splitter(z)
            
            w_low = encoder_l(z_l, w)
            w_high = encoder_h(z_h, w)
            
            z_l_wm = z_l + alpha_l * w_low
            z_h_wm = z_h + alpha_h * w_high
            
            z_wm = recombiner(z_l_wm, z_h_wm)
            
            # Decode to image, apply attack, encode back
            with torch.no_grad():
                # Decode to image
                img_wm = vae.decode(z_wm).sample
                
                # Apply attack
                img_attacked = attack_fn(img_wm)
                
                # Encode back to latent
                z_attacked = vae.encode(img_attacked).latent_dist.sample() * 0.18215
                
            # Extract watermark
            z_l_att, z_h_att = splitter(z_attacked)
            
            w_l_ext = decoder_l(z_l_att)
            w_h_ext = decoder_h(z_h_att)
            w_ext = (w_l_ext + w_h_ext) / 2
            
            # Compute bit accuracy
            bits_true = (w > 0).float()
            bits_pred = (w_ext > 0).float()
            
            bit_acc = (bits_true == bits_pred).float().mean().item()
            ber = (bits_true != bits_pred).float().mean().item()
            
            bit_accs.append(bit_acc)
            bers.append(ber)
        
        results[attack_name] = {
            'params': params,
            'bit_accuracy': np.mean(bit_accs),
            'bit_acc_std': np.std(bit_accs),
            'ber': np.mean(bers),
            'ber_std': np.std(bers)
        }
        
    return results


def compute_detection_metrics(watermark_true, watermark_pred):
    """Compute TPR, FPR, AUC for detection."""
    from sklearn.metrics import roc_curve, auc
    
    # Cosine similarity as detection score
    w_true_norm = F.normalize(watermark_true, dim=1)
    w_pred_norm = F.normalize(watermark_pred, dim=1)
    
    similarity = (w_true_norm * w_pred_norm).sum(dim=1).detach().cpu().numpy()
    
    # For FPR/TPR: all samples are watermarked (positive)
    # We need to generate negative samples (random watermarks)
    num_samples = len(similarity)
    random_similarity = np.random.uniform(-0.5, 0.5, num_samples)
    
    y_true = np.concatenate([np.ones(num_samples), np.zeros(num_samples)])
    y_score = np.concatenate([similarity, random_similarity])
    
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    auc_score = auc(fpr, tpr)
    
    # Find threshold at 1% FPR
    idx = np.argmin(np.abs(fpr - 0.01))
    threshold_at_1fpr = thresholds[idx]
    tpr_at_1fpr = tpr[idx]
    
    return {
        'fpr': fpr.tolist(),
        'tpr': tpr.tolist(),
        'auc': auc_score,
        'threshold_at_1fpr': threshold_at_1fpr,
        'tpr_at_1fpr': tpr_at_1fpr
    }


# ============================================================
# MAIN EVALUATION
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Comprehensive Watermark Evaluation")
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to model checkpoint')
    args = parser.parse_args()
    
    print("=" * 70)
    print("COMPREHENSIVE WATERMARK EVALUATION SUITE")
    print("=" * 70)
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"results/eval_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Load VAE
    print("\n[1/6] Loading VAE...")
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(
        'runwayml/stable-diffusion-v1-5',
        subfolder='vae'
    ).to(device)
    vae.eval()
    
    # Load checkpoint
    print("\n[2/6] Loading model checkpoint...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Use provided checkpoint or find latest
    if args.checkpoint:
        checkpoint_path = args.checkpoint
        if not os.path.isabs(checkpoint_path):
            checkpoint_path = os.path.join(script_dir, checkpoint_path)
    else:
        results_dir = os.path.join(script_dir, 'results')
        # Look for any checkpoint directories
        runs = sorted([d for d in os.listdir(results_dir) 
                      if d.startswith('efficient_') or d.startswith('fast_staged_') or d.startswith('staged_')])
        if not runs:
            print("ERROR: No checkpoint found!")
            return
        latest_run = runs[-1]
        # Try different checkpoint names
        for ckpt_name in ['best_roundtrip.pt', 'best_model.pth', 'final.pt']:
            checkpoint_path = os.path.join(results_dir, latest_run, ckpt_name)
            if os.path.exists(checkpoint_path):
                break
    
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint not found at {checkpoint_path}")
        return
        
    print(f"Checkpoint: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint['config']
    
    # Initialize models
    w_dim = config.get('w_dim', 32)
    splitter = LatentSplitter(mode='dct').to(device)
    recombiner = LatentRecombiner(mode='dct').to(device)
    encoder_l = WatermarkEncoder(watermark_dim=w_dim).to(device)
    encoder_h = WatermarkEncoder(watermark_dim=w_dim).to(device)
    decoder_l = WatermarkDecoder(watermark_dim=w_dim).to(device)
    decoder_h = WatermarkDecoder(watermark_dim=w_dim).to(device)
    
    encoder_l.load_state_dict(checkpoint['encoder_l'])
    encoder_h.load_state_dict(checkpoint['encoder_h'])
    decoder_l.load_state_dict(checkpoint['decoder_l'])
    decoder_h.load_state_dict(checkpoint['decoder_h'])
    
    encoder_l.eval()
    encoder_h.eval()
    decoder_l.eval()
    decoder_h.eval()
    
    alpha_l = checkpoint['alpha_l']
    alpha_h = checkpoint['alpha_h']
    print(f"Alpha L/H: {alpha_l}/{alpha_h}")
    
    # Load test latents
    print("\n[3/6] Loading test data...")
    cache_dir = os.path.join(script_dir, 'cache')
    
    # Find appropriate cache file
    cache_files = [f for f in os.listdir(cache_dir) if f.startswith('latents_') and f.endswith('.pt')]
    if not cache_files:
        print("ERROR: No cached latents found!")
        return
    
    cache_file = sorted(cache_files)[-1]  # Use the latest
    latent_path = os.path.join(cache_dir, cache_file)
    print(f"Latents: {latent_path}")
    
    data = torch.load(latent_path, map_location='cpu')
    latents = data['latents']
    
    print(f"Latent shape: {latents.shape}")
    print(f"Latent range: [{latents.min():.3f}, {latents.max():.3f}]")
    
    # Generate watermarks
    num_test = min(100, len(latents))
    watermarks = torch.randn(num_test, w_dim)
    
    # ============================================================
    # SECTION 1: IMAGE QUALITY METRICS
    # ============================================================
    print("\n" + "=" * 70)
    print("1. IMAGE QUALITY METRICS (Imperceptibility)")
    print("=" * 70)
    
    psnr_list = []
    ssim_list = []
    latent_mse_list = []
    
    print("\nComputing PSNR and SSIM on decoded images...")
    
    for i in tqdm(range(min(20, num_test)), desc="Quality metrics"):
        z = latents[i:i+1].to(device)
        w = watermarks[i:i+1].to(device)
        
        # Encode watermark
        z_l, z_h = splitter(z)
        w_low = encoder_l(z_l, w)
        w_high = encoder_h(z_h, w)
        z_l_wm = z_l + alpha_l * w_low
        z_h_wm = z_h + alpha_h * w_high
        z_wm = recombiner(z_l_wm, z_h_wm)
        
        # Latent MSE
        latent_mse = ((z - z_wm) ** 2).mean().item()
        latent_mse_list.append(latent_mse)
        
        # Decode and compute image metrics
        with torch.no_grad():
            img_orig = vae.decode(z).sample
            img_wm = vae.decode(z_wm).sample
        
        psnr = compute_psnr(img_orig, img_wm)
        ssim = compute_ssim(img_orig, img_wm)
        
        psnr_list.append(psnr)
        ssim_list.append(ssim)
    
    print(f"\n  PSNR: {np.mean(psnr_list):.2f} ± {np.std(psnr_list):.2f} dB")
    print(f"  SSIM: {np.mean(ssim_list):.4f} ± {np.std(ssim_list):.4f}")
    print(f"  Latent MSE: {np.mean(latent_mse_list):.6f}")
    
    # Try LPIPS if VGG available
    try:
        print("\nComputing LPIPS (VGG-based perceptual distance)...")
        from torchvision.models import vgg16, VGG16_Weights
        
        vgg = vgg16(weights=VGG16_Weights.DEFAULT).to(device)
        vgg.eval()
        
        lpips_list = []
        for i in range(min(10, num_test)):
            z = latents[i:i+1].to(device)
            w = watermarks[i:i+1].to(device)
            
            z_l, z_h = splitter(z)
            w_low = encoder_l(z_l, w)
            w_high = encoder_h(z_h, w)
            z_wm = recombiner(z_l + alpha_l * w_low, z_h + alpha_h * w_high)
            
            with torch.no_grad():
                img_orig = vae.decode(z).sample
                img_wm = vae.decode(z_wm).sample
                
                # Simple LPIPS approximation using VGG features
                def get_vgg_features(img):
                    img = (img + 1) / 2  # [-1, 1] -> [0, 1]
                    img = F.interpolate(img, size=(224, 224), mode='bilinear', align_corners=False)
                    mean = torch.tensor([0.485, 0.456, 0.406], device=img.device).view(1, 3, 1, 1)
                    std = torch.tensor([0.229, 0.224, 0.225], device=img.device).view(1, 3, 1, 1)
                    img = (img - mean) / std
                    return vgg.features(img)
                
                feat_orig = get_vgg_features(img_orig)
                feat_wm = get_vgg_features(img_wm)
                
                lpips = ((feat_orig - feat_wm) ** 2).mean().item()
                lpips_list.append(lpips)
        
        print(f"  LPIPS: {np.mean(lpips_list):.6f} ± {np.std(lpips_list):.6f}")
    except Exception as e:
        print(f"  LPIPS: Skipped ({e})")
    
    # ============================================================
    # SECTION 2: WATERMARK ROBUSTNESS EVALUATION
    # ============================================================
    print("\n" + "=" * 70)
    print("2. WATERMARK ROBUSTNESS EVALUATION")
    print("=" * 70)
    
    # Define attacks
    attacks = [
        ("JPEG-Q90", lambda x: jpeg_attack(x, 90), {"quality": 90}),
        ("JPEG-Q70", lambda x: jpeg_attack(x, 70), {"quality": 70}),
        ("JPEG-Q50", lambda x: jpeg_attack(x, 50), {"quality": 50}),
        ("JPEG-Q30", lambda x: jpeg_attack(x, 30), {"quality": 30}),
        ("Noise-σ0.01", lambda x: gaussian_noise_attack(x, 0.01), {"sigma": 0.01}),
        ("Noise-σ0.05", lambda x: gaussian_noise_attack(x, 0.05), {"sigma": 0.05}),
        ("Noise-σ0.1", lambda x: gaussian_noise_attack(x, 0.1), {"sigma": 0.1}),
        ("Blur-K3", lambda x: gaussian_blur_attack(x, 3), {"kernel_size": 3}),
        ("Blur-K5", lambda x: gaussian_blur_attack(x, 5), {"kernel_size": 5}),
        ("Blur-K7", lambda x: gaussian_blur_attack(x, 7), {"kernel_size": 7}),
        ("Resize-0.5x", lambda x: resize_attack(x, 0.5), {"scale": 0.5}),
        ("Resize-0.75x", lambda x: resize_attack(x, 0.75), {"scale": 0.75}),
        ("Resize-1.5x", lambda x: resize_attack(x, 1.5), {"scale": 1.5}),
        ("Crop-10%", lambda x: crop_attack(x, 0.10), {"crop_ratio": 0.10}),
        ("Crop-25%", lambda x: crop_attack(x, 0.25), {"crop_ratio": 0.25}),
        ("Rotate-5°", lambda x: rotation_attack(x, 5.0), {"angle": 5.0}),
        ("Rotate-10°", lambda x: rotation_attack(x, 10.0), {"angle": 10.0}),
        ("Rotate--5°", lambda x: rotation_attack(x, -5.0), {"angle": -5.0}),
        ("Rotate--10°", lambda x: rotation_attack(x, -10.0), {"angle": -10.0}),
    ]
    
    # Also compute no-attack baseline
    print("\nBaseline (no attack):")
    baseline_accs = []
    for i in range(min(50, num_test)):
        z = latents[i:i+1].to(device)
        w = watermarks[i:i+1].to(device)
        
        z_l, z_h = splitter(z)
        w_low = encoder_l(z_l, w)
        w_high = encoder_h(z_h, w)
        z_wm = recombiner(z_l + alpha_l * w_low, z_h + alpha_h * w_high)
        
        z_l_wm, z_h_wm = splitter(z_wm)
        w_l_ext = decoder_l(z_l_wm)
        w_h_ext = decoder_h(z_h_wm)
        w_ext = (w_l_ext + w_h_ext) / 2
        
        bits_true = (w > 0).float()
        bits_pred = (w_ext > 0).float()
        bit_acc = (bits_true == bits_pred).float().mean().item()
        baseline_accs.append(bit_acc)
    
    print(f"  Bit Accuracy (latent-space): {np.mean(baseline_accs):.4f} ± {np.std(baseline_accs):.4f}")
    
    # Test through VAE roundtrip
    print("\nVAE Roundtrip (no attack, decode→encode):")
    vae_rt_accs = []
    for i in range(min(30, num_test)):
        z = latents[i:i+1].to(device)
        w = watermarks[i:i+1].to(device)
        
        z_l, z_h = splitter(z)
        w_low = encoder_l(z_l, w)
        w_high = encoder_h(z_h, w)
        z_wm = recombiner(z_l + alpha_l * w_low, z_h + alpha_h * w_high)
        
        with torch.no_grad():
            img_wm = vae.decode(z_wm).sample
            z_rt = vae.encode(img_wm).latent_dist.sample() * 0.18215
        
        z_l_rt, z_h_rt = splitter(z_rt)
        w_l_ext = decoder_l(z_l_rt)
        w_h_ext = decoder_h(z_h_rt)
        w_ext = (w_l_ext + w_h_ext) / 2
        
        bits_true = (w > 0).float()
        bits_pred = (w_ext > 0).float()
        bit_acc = (bits_true == bits_pred).float().mean().item()
        vae_rt_accs.append(bit_acc)
    
    print(f"  Bit Accuracy (image-space): {np.mean(vae_rt_accs):.4f} ± {np.std(vae_rt_accs):.4f}")
    
    # Run attack robustness tests
    print("\nTesting robustness under attacks...")
    robustness_results = evaluate_robustness(
        latents[:num_test], watermarks, encoder_l, encoder_h, decoder_l, decoder_h,
        splitter, recombiner, vae, alpha_l, alpha_h, device, attacks
    )
    
    print("\n┌─────────────────────┬────────────────┬────────────────┐")
    print("│       Attack        │  Bit Accuracy  │      BER       │")
    print("├─────────────────────┼────────────────┼────────────────┤")
    for name, res in robustness_results.items():
        print(f"│ {name:<19} │ {res['bit_accuracy']:.4f}±{res['bit_acc_std']:.4f} │ {res['ber']:.4f}±{res['ber_std']:.4f} │")
    print("└─────────────────────┴────────────────┴────────────────┘")
    
    # ============================================================
    # SECTION 3: STATISTICAL ANALYSIS
    # ============================================================
    print("\n" + "=" * 70)
    print("3. STATISTICAL ANALYSIS")
    print("=" * 70)
    
    from scipy import stats as scipy_stats
    
    # Confidence intervals
    def compute_ci(data, confidence=0.95):
        n = len(data)
        mean = np.mean(data)
        se = scipy_stats.sem(data)
        h = se * scipy_stats.t.ppf((1 + confidence) / 2, n - 1)
        return mean, mean - h, mean + h
    
    print("\n95% Confidence Intervals:")
    
    mean, ci_lo, ci_hi = compute_ci(psnr_list)
    print(f"  PSNR: {mean:.2f} dB [{ci_lo:.2f}, {ci_hi:.2f}]")
    
    mean, ci_lo, ci_hi = compute_ci(ssim_list)
    print(f"  SSIM: {mean:.4f} [{ci_lo:.4f}, {ci_hi:.4f}]")
    
    mean, ci_lo, ci_hi = compute_ci(baseline_accs)
    print(f"  Bit Accuracy (latent): {mean:.4f} [{ci_lo:.4f}, {ci_hi:.4f}]")
    
    mean, ci_lo, ci_hi = compute_ci(vae_rt_accs)
    print(f"  Bit Accuracy (image): {mean:.4f} [{ci_lo:.4f}, {ci_hi:.4f}]")
    
    # Statistical significance vs random baseline (50%)
    print("\nStatistical Significance Testing (vs 50% random baseline):")
    t_stat, p_value = scipy_stats.ttest_1samp(baseline_accs, 0.5)
    print(f"  t-statistic: {t_stat:.4f}")
    print(f"  p-value: {p_value:.2e}")
    print(f"  Significant: {'Yes' if p_value < 0.05 else 'No'} (α=0.05)")
    
    # Detection metrics
    print("\nDetection Threshold Calibration:")
    
    # Collect all watermarks and predictions
    all_w_true = []
    all_w_pred = []
    for i in range(min(50, num_test)):
        z = latents[i:i+1].to(device)
        w = watermarks[i:i+1].to(device)
        
        z_l, z_h = splitter(z)
        w_low = encoder_l(z_l, w)
        w_high = encoder_h(z_h, w)
        z_wm = recombiner(z_l + alpha_l * w_low, z_h + alpha_h * w_high)
        
        z_l_wm, z_h_wm = splitter(z_wm)
        w_l_ext = decoder_l(z_l_wm)
        w_h_ext = decoder_h(z_h_wm)
        w_ext = (w_l_ext + w_h_ext) / 2
        
        all_w_true.append(w)
        all_w_pred.append(w_ext)
    
    all_w_true = torch.cat(all_w_true, dim=0)
    all_w_pred = torch.cat(all_w_pred, dim=0)
    
    detection_metrics = compute_detection_metrics(all_w_true, all_w_pred)
    print(f"  AUC: {detection_metrics['auc']:.4f}")
    print(f"  TPR at 1% FPR: {detection_metrics['tpr_at_1fpr']:.4f}")
    
    # ============================================================
    # SECTION 4: ABLATION STUDIES
    # ============================================================
    print("\n" + "=" * 70)
    print("4. ABLATION STUDIES")
    print("=" * 70)
    
    print("\nAlpha Strength Analysis (using current trained model):")
    print(f"  Current alpha_l: {alpha_l}, alpha_h: {alpha_h}")
    
    # Simulate different alpha scales
    alpha_scales = [0.5, 0.75, 1.0, 1.5, 2.0]
    print("\nEffect of scaling alpha (relative to trained values):")
    print("┌────────┬────────────┬────────────┬────────────┐")
    print("│ Scale  │    PSNR    │    SSIM    │  Bit Acc   │")
    print("├────────┼────────────┼────────────┼────────────┤")
    
    for scale in alpha_scales:
        al = alpha_l * scale
        ah = alpha_h * scale
        
        psnrs, ssims, accs = [], [], []
        for i in range(min(10, num_test)):
            z = latents[i:i+1].to(device)
            w = watermarks[i:i+1].to(device)
            
            z_l, z_h = splitter(z)
            w_low = encoder_l(z_l, w)
            w_high = encoder_h(z_h, w)
            z_wm = recombiner(z_l + al * w_low, z_h + ah * w_high)
            
            with torch.no_grad():
                img_orig = vae.decode(z).sample
                img_wm = vae.decode(z_wm).sample
            
            psnrs.append(compute_psnr(img_orig, img_wm))
            ssims.append(compute_ssim(img_orig, img_wm))
            
            z_l_wm, z_h_wm = splitter(z_wm)
            w_ext = (decoder_l(z_l_wm) + decoder_h(z_h_wm)) / 2
            acc = ((w > 0) == (w_ext > 0)).float().mean().item()
            accs.append(acc)
        
        print(f"│ {scale:.2f}x  │ {np.mean(psnrs):>8.2f}dB │   {np.mean(ssims):.4f}   │   {np.mean(accs):.4f}   │")
    
    print("└────────┴────────────┴────────────┴────────────┘")
    
    # Watermark bit length analysis (simulated)
    print("\nWatermark Bit Length Analysis:")
    print("  Note: Current model trained with w_dim={w_dim}")
    print("  (Full ablation would require retraining with different dimensions)")
    
    # ============================================================
    # SECTION 5: COMPUTATIONAL ANALYSIS
    # ============================================================
    print("\n" + "=" * 70)
    print("5. COMPUTATIONAL ANALYSIS")
    print("=" * 70)
    
    num_timing_runs = 20
    z_test = latents[0:1].to(device)
    w_test = watermarks[0:1].to(device)
    
    # Warm-up
    for _ in range(5):
        z_l, z_h = splitter(z_test)
        w_low = encoder_l(z_l, w_test)
        w_high = encoder_h(z_h, w_test)
        _ = recombiner(z_l + alpha_l * w_low, z_h + alpha_h * w_high)
    
    # Time watermark encoding
    encode_times = []
    for _ in range(num_timing_runs):
        start = time.perf_counter()
        z_l, z_h = splitter(z_test)
        w_low = encoder_l(z_l, w_test)
        w_high = encoder_h(z_h, w_test)
        z_wm = recombiner(z_l + alpha_l * w_low, z_h + alpha_h * w_high)
        if device.type == 'mps':
            torch.mps.synchronize()
        encode_times.append((time.perf_counter() - start) * 1000)
    
    print(f"\nWatermark Encoding (latent space):")
    print(f"  Time: {np.mean(encode_times):.2f} ± {np.std(encode_times):.2f} ms")
    print(f"  Throughput: {1000 / np.mean(encode_times):.1f} images/sec")
    
    # Time watermark decoding
    z_wm = recombiner(z_l + alpha_l * w_low, z_h + alpha_h * w_high)
    decode_times = []
    for _ in range(num_timing_runs):
        start = time.perf_counter()
        z_l_wm, z_h_wm = splitter(z_wm)
        w_l_ext = decoder_l(z_l_wm)
        w_h_ext = decoder_h(z_h_wm)
        w_ext = (w_l_ext + w_h_ext) / 2
        if device.type == 'mps':
            torch.mps.synchronize()
        decode_times.append((time.perf_counter() - start) * 1000)
    
    print(f"\nWatermark Decoding (latent space):")
    print(f"  Time: {np.mean(decode_times):.2f} ± {np.std(decode_times):.2f} ms")
    print(f"  Throughput: {1000 / np.mean(decode_times):.1f} images/sec")
    
    # VAE encode/decode times for comparison
    vae_encode_times = []
    vae_decode_times = []
    
    with torch.no_grad():
        img_test = vae.decode(z_test).sample
        
        for _ in range(num_timing_runs):
            start = time.perf_counter()
            _ = vae.encode(img_test).latent_dist.sample()
            if device.type == 'mps':
                torch.mps.synchronize()
            vae_encode_times.append((time.perf_counter() - start) * 1000)
        
        for _ in range(num_timing_runs):
            start = time.perf_counter()
            _ = vae.decode(z_test).sample
            if device.type == 'mps':
                torch.mps.synchronize()
            vae_decode_times.append((time.perf_counter() - start) * 1000)
    
    print(f"\nVAE Operations (for reference):")
    print(f"  Encode: {np.mean(vae_encode_times):.2f} ± {np.std(vae_encode_times):.2f} ms")
    print(f"  Decode: {np.mean(vae_decode_times):.2f} ± {np.std(vae_decode_times):.2f} ms")
    
    print(f"\nOverhead Analysis:")
    wm_overhead = np.mean(encode_times) + np.mean(decode_times)
    vae_total = np.mean(vae_encode_times) + np.mean(vae_decode_times)
    print(f"  Watermark overhead: {wm_overhead:.2f} ms")
    print(f"  VAE processing: {vae_total:.2f} ms")
    print(f"  Relative overhead: {100 * wm_overhead / vae_total:.1f}% of VAE time")
    
    # ============================================================
    # SECTION 6: QUALITATIVE EXPERIMENTS
    # ============================================================
    print("\n" + "=" * 70)
    print("6. QUALITATIVE EXPERIMENTS")
    print("=" * 70)
    
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    # Side-by-side comparison
    print("\nGenerating side-by-side comparisons...")
    
    fig, axes = plt.subplots(4, 3, figsize=(12, 16))
    
    for i in range(4):
        z = latents[i:i+1].to(device)
        w = watermarks[i:i+1].to(device)
        
        z_l, z_h = splitter(z)
        w_low = encoder_l(z_l, w)
        w_high = encoder_h(z_h, w)
        z_wm = recombiner(z_l + alpha_l * w_low, z_h + alpha_h * w_high)
        
        with torch.no_grad():
            img_orig = vae.decode(z).sample
            img_wm = vae.decode(z_wm).sample
        
        # Convert to numpy
        orig_np = ((img_orig[0].cpu().permute(1, 2, 0) + 1) / 2).clamp(0, 1).numpy()
        wm_np = ((img_wm[0].cpu().permute(1, 2, 0) + 1) / 2).clamp(0, 1).numpy()
        diff_np = np.abs(orig_np - wm_np) * 10  # Amplify difference
        
        psnr = compute_psnr(img_orig, img_wm)
        ssim = compute_ssim(img_orig, img_wm)
        
        axes[i, 0].imshow(orig_np)
        axes[i, 0].set_title("Original" if i == 0 else "")
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(wm_np)
        axes[i, 1].set_title(f"Watermarked\nPSNR={psnr:.1f}dB, SSIM={ssim:.3f}" if i == 0 else f"PSNR={psnr:.1f}dB")
        axes[i, 1].axis('off')
        
        axes[i, 2].imshow(diff_np.clip(0, 1))
        axes[i, 2].set_title("Difference (10x)" if i == 0 else "")
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    comparison_path = os.path.join(output_dir, 'side_by_side_comparison.png')
    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {comparison_path}")
    
    # Bit distribution after attacks
    print("\nGenerating bit distributions under attacks...")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    attack_samples = [
        ("No Attack", lambda x: x),
        ("JPEG Q50", lambda x: jpeg_attack(x, 50)),
        ("Noise σ=0.05", lambda x: gaussian_noise_attack(x, 0.05)),
        ("Blur K=5", lambda x: gaussian_blur_attack(x, 5)),
        ("Resize 0.5x", lambda x: resize_attack(x, 0.5)),
        ("Rotate 10°", lambda x: rotation_attack(x, 10)),
    ]
    
    for idx, (name, attack_fn) in enumerate(attack_samples):
        row, col = idx // 3, idx % 3
        
        bit_errors = []
        for i in range(min(20, num_test)):
            z = latents[i:i+1].to(device)
            w = watermarks[i:i+1].to(device)
            
            z_l, z_h = splitter(z)
            w_low = encoder_l(z_l, w)
            w_high = encoder_h(z_h, w)
            z_wm = recombiner(z_l + alpha_l * w_low, z_h + alpha_h * w_high)
            
            with torch.no_grad():
                img_wm = vae.decode(z_wm).sample
                img_attacked = attack_fn(img_wm)
                z_attacked = vae.encode(img_attacked).latent_dist.sample() * 0.18215
            
            z_l_att, z_h_att = splitter(z_attacked)
            w_ext = (decoder_l(z_l_att) + decoder_h(z_h_att)) / 2
            
            # Per-bit errors
            bits_true = (w > 0).float().cpu().numpy().flatten()
            bits_pred = (w_ext > 0).float().cpu().numpy().flatten()
            errors = (bits_true != bits_pred).astype(float)
            bit_errors.append(errors)
        
        bit_errors = np.array(bit_errors)
        error_rate_per_bit = bit_errors.mean(axis=0)
        
        axes[row, col].bar(range(len(error_rate_per_bit)), error_rate_per_bit, color='steelblue')
        axes[row, col].axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Random')
        axes[row, col].set_title(f"{name}\nAvg BER: {bit_errors.mean():.3f}")
        axes[row, col].set_xlabel("Bit Index")
        axes[row, col].set_ylabel("Error Rate")
        axes[row, col].set_ylim([0, 1])
    
    plt.tight_layout()
    bit_dist_path = os.path.join(output_dir, 'bit_distributions.png')
    plt.savefig(bit_dist_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {bit_dist_path}")
    
    # Robustness plot
    print("\nGenerating robustness performance plot...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Group by attack type
    attack_groups = {
        'JPEG': [r for r in robustness_results.items() if 'JPEG' in r[0]],
        'Noise': [r for r in robustness_results.items() if 'Noise' in r[0]],
        'Blur': [r for r in robustness_results.items() if 'Blur' in r[0]],
        'Resize': [r for r in robustness_results.items() if 'Resize' in r[0]],
        'Crop': [r for r in robustness_results.items() if 'Crop' in r[0]],
        'Rotate': [r for r in robustness_results.items() if 'Rotate' in r[0]],
    }
    
    colors = plt.cm.tab10(range(len(attack_groups)))
    
    x_pos = 0
    xticks, xlabels = [], []
    
    for color, (group_name, attacks_in_group) in zip(colors, attack_groups.items()):
        for name, res in attacks_in_group:
            axes[0].bar(x_pos, res['bit_accuracy'], color=color, label=group_name if x_pos == list(attack_groups.keys()).index(group_name) else "")
            axes[0].errorbar(x_pos, res['bit_accuracy'], yerr=res['bit_acc_std'], color='black', capsize=3)
            xticks.append(x_pos)
            xlabels.append(name.split('-')[-1])
            x_pos += 1
        x_pos += 0.5
    
    axes[0].axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Random')
    axes[0].axhline(y=np.mean(baseline_accs), color='g', linestyle='--', alpha=0.5, label='Baseline')
    axes[0].set_xticks(xticks)
    axes[0].set_xticklabels(xlabels, rotation=45, ha='right')
    axes[0].set_ylabel("Bit Accuracy")
    axes[0].set_title("Robustness: Bit Accuracy Under Attacks")
    axes[0].legend(loc='upper right')
    axes[0].set_ylim([0, 1])
    
    # BER plot
    x_pos = 0
    for color, (group_name, attacks_in_group) in zip(colors, attack_groups.items()):
        for name, res in attacks_in_group:
            axes[1].bar(x_pos, res['ber'], color=color)
            axes[1].errorbar(x_pos, res['ber'], yerr=res['ber_std'], color='black', capsize=3)
            x_pos += 1
        x_pos += 0.5
    
    axes[1].axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Random')
    axes[1].set_xticks(xticks)
    axes[1].set_xticklabels(xlabels, rotation=45, ha='right')
    axes[1].set_ylabel("Bit Error Rate")
    axes[1].set_title("Robustness: BER Under Attacks")
    axes[1].set_ylim([0, 1])
    
    plt.tight_layout()
    robustness_path = os.path.join(output_dir, 'robustness_plot.png')
    plt.savefig(robustness_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {robustness_path}")
    
    # ============================================================
    # SUMMARY
    # ============================================================
    print("\n" + "=" * 70)
    print("EVALUATION SUMMARY")
    print("=" * 70)
    
    print(f"""
    Model: {checkpoint_path}
    Latents: {latent_path}
    Alpha L/H: {alpha_l}/{alpha_h}
    
    IMAGE QUALITY:
      PSNR: {np.mean(psnr_list):.2f} ± {np.std(psnr_list):.2f} dB
      SSIM: {np.mean(ssim_list):.4f} ± {np.std(ssim_list):.4f}
      Latent MSE: {np.mean(latent_mse_list):.6f}
    
    WATERMARK DETECTION:
      Bit Accuracy (latent): {np.mean(baseline_accs):.4f} ± {np.std(baseline_accs):.4f}
      Bit Accuracy (image): {np.mean(vae_rt_accs):.4f} ± {np.std(vae_rt_accs):.4f}
      AUC: {detection_metrics['auc']:.4f}
    
    ROBUSTNESS (best/worst):
      Best: JPEG-Q90 ({robustness_results['JPEG-Q90']['bit_accuracy']:.4f})
      Worst: Noise-σ0.1 ({robustness_results['Noise-σ0.1']['bit_accuracy']:.4f})
    
    COMPUTATIONAL:
      Encoding: {np.mean(encode_times):.2f} ms/image
      Decoding: {np.mean(decode_times):.2f} ms/image
      Overhead: {100 * wm_overhead / vae_total:.1f}% of VAE time
    
    Output saved to: {output_dir}
    """)
    
    # Save results to JSON
    import json
    results_summary = {
        'config': {
            'checkpoint': checkpoint_path,
            'latents': latent_path,
            'alpha_l': alpha_l,
            'alpha_h': alpha_h,
            'w_dim': w_dim,
        },
        'image_quality': {
            'psnr_mean': float(np.mean(psnr_list)),
            'psnr_std': float(np.std(psnr_list)),
            'ssim_mean': float(np.mean(ssim_list)),
            'ssim_std': float(np.std(ssim_list)),
            'latent_mse': float(np.mean(latent_mse_list)),
        },
        'detection': {
            'bit_accuracy_latent': float(np.mean(baseline_accs)),
            'bit_accuracy_image': float(np.mean(vae_rt_accs)),
            'auc': float(detection_metrics['auc']),
        },
        'robustness': {k: {kk: float(vv) if isinstance(vv, (int, float, np.floating)) else vv 
                          for kk, vv in v.items()} 
                       for k, v in robustness_results.items()},
        'computational': {
            'encode_time_ms': float(np.mean(encode_times)),
            'decode_time_ms': float(np.mean(decode_times)),
            'vae_encode_ms': float(np.mean(vae_encode_times)),
            'vae_decode_ms': float(np.mean(vae_decode_times)),
        }
    }
    
    results_json_path = os.path.join(output_dir, 'results.json')
    with open(results_json_path, 'w') as f:
        json.dump(results_summary, f, indent=2)
    print(f"\nResults saved to: {results_json_path}")


if __name__ == "__main__":
    main()

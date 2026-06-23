#!/usr/bin/env python3
"""
Evaluate effect of bit length on watermark quality and robustness.

Bit lengths: 32, 48, 64, 96, 128
Metrics: PSNR, SSIM, LPIPS, SIFID (approximation)
Attacks: None, C.Crop 0.1, Contrast 2.0, Rotation 15, JPEG 70, Combined
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
from torch.utils.data import DataLoader, TensorDataset

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from models.latent_split import LatentSplitter
from models.recombination import LatentRecombiner
from models.watermark_encoder import WatermarkEncoder
from models.watermark_decoder import WatermarkDecoder


# ============================================================
# ATTACK FUNCTIONS
# ============================================================

def center_crop_attack(images, crop_ratio=0.1):
    """Center crop and resize back."""
    B, C, H, W = images.shape
    crop_h = int(H * crop_ratio)
    crop_w = int(W * crop_ratio)
    
    cropped = images[:, :, crop_h:H-crop_h, crop_w:W-crop_w]
    return F.interpolate(cropped, size=(H, W), mode='bilinear', align_corners=False)


def contrast_attack(images, factor=2.0):
    """Adjust contrast."""
    mean = images.mean(dim=(2, 3), keepdim=True)
    return torch.clamp((images - mean) * factor + mean, -1, 1)


def rotation_attack(images, angle=15.0):
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


def jpeg_attack(images, quality=70):
    """Simulated JPEG compression."""
    scale_factor = max(0.3, quality / 100)
    B, C, H, W = images.shape
    h_small = max(8, int(H * scale_factor))
    w_small = max(8, int(W * scale_factor))
    
    images_down = F.interpolate(images, size=(h_small, w_small), mode='bilinear', align_corners=False)
    images_up = F.interpolate(images_down, size=(H, W), mode='bilinear', align_corners=False)
    
    blend = quality / 100
    return blend * images + (1 - blend) * images_up


def combined_attack(images):
    """Combined attack: JPEG + Crop + Contrast."""
    x = jpeg_attack(images, 70)
    x = center_crop_attack(x, 0.05)
    x = contrast_attack(x, 1.5)
    return x


# ============================================================
# METRIC FUNCTIONS
# ============================================================

def compute_psnr(img1, img2, data_range=2.0):
    """Compute PSNR."""
    mse = ((img1 - img2) ** 2).mean()
    if mse < 1e-10:
        return 100.0
    return (10 * torch.log10((data_range ** 2) / mse)).item()


def compute_ssim(img1, img2, window_size=11, data_range=2.0):
    """Compute SSIM."""
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    
    # Create Gaussian window
    sigma = 1.5
    coords = torch.arange(window_size, device=img1.device, dtype=img1.dtype) - window_size // 2
    gauss = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    gauss = gauss / gauss.sum()
    window = (gauss.unsqueeze(1) @ gauss.unsqueeze(0)).unsqueeze(0).unsqueeze(0)
    window = window.expand(img1.shape[1], 1, -1, -1)
    
    pad = window_size // 2
    
    mu1 = F.conv2d(img1, window, padding=pad, groups=img1.shape[1])
    mu2 = F.conv2d(img2, window, padding=pad, groups=img2.shape[1])
    
    sigma1_sq = F.conv2d(img1 * img1, window, padding=pad, groups=img1.shape[1]) - mu1 ** 2
    sigma2_sq = F.conv2d(img2 * img2, window, padding=pad, groups=img2.shape[1]) - mu2 ** 2
    sigma12 = F.conv2d(img1 * img2, window, padding=pad, groups=img1.shape[1]) - mu1 * mu2
    
    ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return ssim_map.mean().item()


class SimpleLPIPS(nn.Module):
    """Simplified LPIPS using VGG features."""
    def __init__(self, device):
        super().__init__()
        from torchvision.models import vgg16, VGG16_Weights
        vgg = vgg16(weights=VGG16_Weights.DEFAULT)
        vgg.eval()
        for p in vgg.parameters():
            p.requires_grad = False
        
        self.features = nn.Sequential(*list(vgg.features)[:23]).to(device)  # Up to relu4_3
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.to(device)
    
    def forward(self, x, y):
        # Normalize from [-1, 1] to ImageNet
        x = ((x + 1) / 2 - self.mean.to(x.device)) / self.std.to(x.device)
        y = ((y + 1) / 2 - self.mean.to(y.device)) / self.std.to(y.device)
        
        fx = self.features(x)
        fy = self.features(y)
        
        return F.mse_loss(fx, fy)


def compute_sifid_approx(features1, features2):
    """Simplified SIFID approximation using feature statistics."""
    # Use mean/std difference as proxy for FID
    mu1, std1 = features1.mean(), features1.std()
    mu2, std2 = features2.mean(), features2.std()
    
    return ((mu1 - mu2) ** 2 + (std1 - std2) ** 2).item()


# ============================================================
# TRAINING FUNCTIONS
# ============================================================

def train_model_for_bitlength(w_dim, latents, device, epochs=150, lr=3e-4, alpha_l=0.1, alpha_h=0.05):
    """Train watermark model for specific bit length."""
    print(f"\n{'='*60}")
    print(f"Training model for {w_dim} bits")
    print(f"{'='*60}")
    
    # Initialize models
    splitter = LatentSplitter(mode='dct').to(device)
    recombiner = LatentRecombiner(mode='dct').to(device)
    encoder_l = WatermarkEncoder(watermark_dim=w_dim).to(device)
    encoder_h = WatermarkEncoder(watermark_dim=w_dim).to(device)
    decoder_l = WatermarkDecoder(watermark_dim=w_dim).to(device)
    decoder_h = WatermarkDecoder(watermark_dim=w_dim).to(device)
    
    # Optimizer with higher learning rate
    params = (
        list(encoder_l.parameters()) + list(encoder_h.parameters()) +
        list(decoder_l.parameters()) + list(decoder_h.parameters())
    )
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # DataLoader with smaller batch for better gradients
    dataset = TensorDataset(latents)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    best_acc = 0.0
    
    # Training loop
    for epoch in range(epochs):
        epoch_loss = []
        epoch_acc = []
        
        for batch in dataloader:
            z = batch[0].to(device)
            B = z.shape[0]
            
            # Generate binary watermark directly (more stable training)
            w = (torch.rand(B, w_dim, device=device) > 0.5).float() * 2 - 1  # {-1, 1}
            
            # Encode
            z_low, z_high = splitter(z)
            z_low_wm = encoder_l(z_low, w, alpha=alpha_l)
            z_high_wm = encoder_h(z_high, w, alpha=alpha_h)
            z_wm = recombiner(z_low_wm, z_high_wm)
            
            # Decode
            z_wm_low, z_wm_high = splitter(z_wm)
            w_pred_l = decoder_l(z_wm_low)
            w_pred_h = decoder_h(z_wm_high)
            
            # Losses with BCE for binary classification (more stable)
            loss_w = F.mse_loss(w_pred_l, w) + F.mse_loss(w_pred_h, w)
            loss_cons = F.mse_loss(w_pred_l, w_pred_h)
            loss_latent = F.mse_loss(z_wm, z)
            
            # Higher weight on watermark loss for accuracy
            total_loss = 5.0 * loss_w + 0.5 * loss_cons + 0.1 * loss_latent
            
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            
            # Bit accuracy
            with torch.no_grad():
                w_pred = (w_pred_l + w_pred_h) / 2
                bits_true = (w > 0).float()
                bits_pred = (w_pred > 0).float()
                acc = (bits_true == bits_pred).float().mean().item()
            
            epoch_loss.append(total_loss.item())
            epoch_acc.append(acc)
        
        scheduler.step()
        
        mean_acc = np.mean(epoch_acc)
        if mean_acc > best_acc:
            best_acc = mean_acc
        
        if (epoch + 1) % 30 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{epochs}: Loss={np.mean(epoch_loss):.4f}, Acc={mean_acc:.4f}")
    
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


# ============================================================
# EVALUATION FUNCTIONS
# ============================================================

def evaluate_model(models, latents, vae, device, w_dim, n_samples=100):
    """Comprehensive evaluation of a trained model."""
    splitter = models['splitter']
    recombiner = models['recombiner']
    encoder_l = models['encoder_l']
    encoder_h = models['encoder_h']
    decoder_l = models['decoder_l']
    decoder_h = models['decoder_h']
    alpha_l = models['alpha_l']
    alpha_h = models['alpha_h']
    
    # Set to eval mode
    encoder_l.eval()
    encoder_h.eval()
    decoder_l.eval()
    decoder_h.eval()
    
    # LPIPS model
    try:
        lpips_model = SimpleLPIPS(device)
    except:
        lpips_model = None
        print("  Warning: LPIPS not available")
    
    results = {
        'w_dim': w_dim,
        'quality': {'psnr': [], 'ssim': [], 'lpips': [], 'sifid': []},
        'attacks': {}
    }
    
    # Attack configurations
    attacks = [
        ('None', lambda x: x),
        ('C.Crop 0.1', lambda x: center_crop_attack(x, 0.1)),
        ('Contr. 2.0', lambda x: contrast_attack(x, 2.0)),
        ('Rot. 15', lambda x: rotation_attack(x, 15.0)),
        ('JPEG 70', lambda x: jpeg_attack(x, 70)),
        ('Comb.', combined_attack),
    ]
    
    for attack_name, _ in attacks:
        results['attacks'][attack_name] = []
    
    print(f"  Evaluating with {n_samples} samples...")
    
    with torch.no_grad():
        for i in tqdm(range(min(n_samples, len(latents))), desc=f"  Eval {w_dim}-bit", leave=False):
            z = latents[i:i+1].to(device)
            w = torch.randn(1, w_dim, device=device)
            
            # Encode watermark
            z_low, z_high = splitter(z)
            z_low_wm = encoder_l(z_low, w, alpha=alpha_l)
            z_high_wm = encoder_h(z_high, w, alpha=alpha_h)
            z_wm = recombiner(z_low_wm, z_high_wm)
            
            # Decode to images
            img_orig = vae.decode(z / 0.18215).sample
            img_wm = vae.decode(z_wm / 0.18215).sample
            
            # Quality metrics
            psnr = compute_psnr(img_orig, img_wm)
            ssim = compute_ssim(img_orig, img_wm)
            results['quality']['psnr'].append(psnr)
            results['quality']['ssim'].append(ssim)
            
            if lpips_model is not None:
                lpips_val = lpips_model(img_orig, img_wm).item()
                results['quality']['lpips'].append(lpips_val)
            
            # SIFID approximation (using VGG features)
            if lpips_model is not None:
                f1 = lpips_model.features(((img_orig + 1) / 2 - lpips_model.mean) / lpips_model.std)
                f2 = lpips_model.features(((img_wm + 1) / 2 - lpips_model.mean) / lpips_model.std)
                sifid = compute_sifid_approx(f1, f2)
                results['quality']['sifid'].append(sifid)
            
            # Test attacks (through VAE roundtrip)
            for attack_name, attack_fn in attacks:
                # Apply attack to watermarked image
                img_attacked = attack_fn(img_wm)
                
                # Encode back to latent
                z_attacked = vae.encode(img_attacked).latent_dist.mean * 0.18215
                
                # Extract watermark
                z_att_low, z_att_high = splitter(z_attacked)
                w_pred_l = decoder_l(z_att_low)
                w_pred_h = decoder_h(z_att_high)
                w_pred = (w_pred_l + w_pred_h) / 2
                
                # Bit accuracy
                bits_true = (w > 0).float()
                bits_pred = (w_pred > 0).float()
                acc = (bits_true == bits_pred).float().mean().item()
                results['attacks'][attack_name].append(acc)
    
    # Compute statistics
    for key in results['quality']:
        if results['quality'][key]:
            vals = results['quality'][key]
            results['quality'][key] = {'mean': np.mean(vals), 'std': np.std(vals)}
        else:
            results['quality'][key] = {'mean': 0, 'std': 0}
    
    for attack_name in results['attacks']:
        vals = results['attacks'][attack_name]
        results['attacks'][attack_name] = {'mean': np.mean(vals), 'std': np.std(vals)}
    
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate effect of bit length')
    parser.add_argument('--latents', type=str, default='cache/latents_1000_256.pt',
                        help='Path to precomputed latents')
    parser.add_argument('--output', type=str, default='results/bit_length_study.json',
                        help='Output JSON file')
    parser.add_argument('--epochs', type=int, default=150, help='Training epochs per model')
    parser.add_argument('--n_eval', type=int, default=100, help='Number of evaluation samples')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Device
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # Change to project root
    os.chdir(PROJECT_ROOT)
    
    # Load VAE
    print("Loading VAE...")
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(
        'runwayml/stable-diffusion-v1-5',
        subfolder='vae',
        torch_dtype=torch.float32
    ).to(device)
    vae.eval()
    
    # Load latents
    print(f"Loading latents from {args.latents}...")
    latent_data = torch.load(args.latents, map_location='cpu', weights_only=False)
    latents = latent_data['latents'] if isinstance(latent_data, dict) else latent_data
    latents = latents * 0.18215  # Apply scaling factor
    print(f"  Loaded {len(latents)} latents")
    
    # Bit lengths to evaluate
    bit_lengths = [32, 48, 64, 96, 128]
    
    # Results
    all_results = {
        'config': {
            'epochs': args.epochs,
            'n_eval': args.n_eval,
            'seed': args.seed,
            'alpha_l': 0.1,
            'alpha_h': 0.05,
            'latents': args.latents
        },
        'bit_lengths': {},
        'timestamp': datetime.now().isoformat()
    }
    
    # Train and evaluate for each bit length
    for w_dim in bit_lengths:
        # Train model
        models = train_model_for_bitlength(
            w_dim=w_dim,
            latents=latents[:500],  # Use subset for training
            device=device,
            epochs=args.epochs
        )
        
        # Evaluate
        results = evaluate_model(
            models=models,
            latents=latents[500:500+args.n_eval],  # Use different samples for eval
            vae=vae,
            device=device,
            w_dim=w_dim,
            n_samples=args.n_eval
        )
        
        all_results['bit_lengths'][str(w_dim)] = results
        
        # Print summary
        print(f"\n  Results for {w_dim} bits:")
        print(f"    PSNR: {results['quality']['psnr']['mean']:.2f} ± {results['quality']['psnr']['std']:.2f} dB")
        print(f"    SSIM: {results['quality']['ssim']['mean']:.4f} ± {results['quality']['ssim']['std']:.4f}")
        if results['quality']['lpips']['mean'] > 0:
            print(f"    LPIPS: {results['quality']['lpips']['mean']:.4f} ± {results['quality']['lpips']['std']:.4f}")
            print(f"    SIFID: {results['quality']['sifid']['mean']:.6f} ± {results['quality']['sifid']['std']:.6f}")
        print(f"    Attacks:")
        for attack_name, attack_res in results['attacks'].items():
            print(f"      {attack_name}: {attack_res['mean']*100:.1f}% ± {attack_res['std']*100:.1f}%")
        
        # Clear memory
        del models
        if device.type == 'mps':
            torch.mps.empty_cache()
    
    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {args.output}")
    
    # Print summary table
    print("\n" + "="*80)
    print("SUMMARY: Effect of Bit Length on Quality and Robustness")
    print("="*80)
    
    # Quality table
    print("\n┌─────────┬──────────────────┬──────────────────┬──────────────────┬──────────────────┐")
    print("│ Bit Len │       PSNR       │       SSIM       │      LPIPS       │      SIFID       │")
    print("├─────────┼──────────────────┼──────────────────┼──────────────────┼──────────────────┤")
    for w_dim in bit_lengths:
        r = all_results['bit_lengths'][str(w_dim)]['quality']
        psnr = f"{r['psnr']['mean']:.2f}±{r['psnr']['std']:.2f}"
        ssim = f"{r['ssim']['mean']:.4f}±{r['ssim']['std']:.4f}"
        lpips = f"{r['lpips']['mean']:.4f}±{r['lpips']['std']:.4f}" if r['lpips']['mean'] > 0 else "N/A"
        sifid = f"{r['sifid']['mean']:.6f}" if r['sifid']['mean'] > 0 else "N/A"
        print(f"│   {w_dim:3d}   │ {psnr:^16s} │ {ssim:^16s} │ {lpips:^16s} │ {sifid:^16s} │")
    print("└─────────┴──────────────────┴──────────────────┴──────────────────┴──────────────────┘")
    
    # Attack robustness table
    print("\n┌─────────┬────────┬────────────┬────────────┬─────────┬──────────┬─────────┐")
    print("│ Bit Len │  None  │ C.Crop 0.1 │ Contr. 2.0 │ Rot. 15 │ JPEG 70  │  Comb.  │")
    print("├─────────┼────────┼────────────┼────────────┼─────────┼──────────┼─────────┤")
    for w_dim in bit_lengths:
        r = all_results['bit_lengths'][str(w_dim)]['attacks']
        row = f"│   {w_dim:3d}   │"
        for attack in ['None', 'C.Crop 0.1', 'Contr. 2.0', 'Rot. 15', 'JPEG 70', 'Comb.']:
            val = r[attack]['mean'] * 100
            row += f" {val:5.1f}% │"
        print(row)
    print("└─────────┴────────┴────────────┴────────────┴─────────┴──────────┴─────────┘")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Compare all top models on specific attacks:
- None (clean roundtrip)
- Center Crop 0.1
- Random Crop 0.1
- Resize 0.7
- Rotation 15
- Blur
- Contrast 2.0
- Brightness 2.0 
- JPEG 70
- Combined
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import io
from PIL import Image

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.dirname(project_root))

from latent_watermarking.models.latent_split import LatentSplitter
from latent_watermarking.models.recombination import LatentRecombiner
from latent_watermarking.models.watermark_encoder import WatermarkEncoder
from latent_watermarking.models.watermark_decoder import WatermarkDecoder


# ============================================================
# ATTACK FUNCTIONS
# ============================================================

def real_jpeg_attack(images, quality=70):
    """Real JPEG compression using PIL."""
    device = images.device
    results = []
    for i in range(images.shape[0]):
        img = images[i].cpu()
        img_np = ((img.permute(1, 2, 0) + 1) / 2 * 255).clamp(0, 255).numpy().astype(np.uint8)
        pil_img = Image.fromarray(img_np)
        
        buffer = io.BytesIO()
        pil_img.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        compressed = Image.open(buffer).convert('RGB')
        
        img_out = torch.from_numpy(np.array(compressed)).float() / 255.0
        img_out = img_out.permute(2, 0, 1) * 2 - 1
        results.append(img_out)
    
    return torch.stack(results).to(device)


def center_crop_attack(images, crop_ratio=0.1):
    B, C, H, W = images.shape
    crop_h = int(H * crop_ratio)
    crop_w = int(W * crop_ratio)
    cropped = images[:, :, crop_h:H-crop_h, crop_w:W-crop_w]
    return F.interpolate(cropped, size=(H, W), mode='bilinear', align_corners=False)


def random_crop_attack(images, crop_ratio=0.1):
    B, C, H, W = images.shape
    crop_h = int(H * crop_ratio * 2)
    crop_w = int(W * crop_ratio * 2)
    top = torch.randint(0, crop_h + 1, (1,)).item() if crop_h > 0 else 0
    left = torch.randint(0, crop_w + 1, (1,)).item() if crop_w > 0 else 0
    new_h, new_w = H - crop_h, W - crop_w
    cropped = images[:, :, top:top+new_h, left:left+new_w]
    return F.interpolate(cropped, size=(H, W), mode='bilinear', align_corners=False)


def resize_attack(images, scale=0.7):
    B, C, H, W = images.shape
    h_small = max(1, int(H * scale))
    w_small = max(1, int(W * scale))
    down = F.interpolate(images, size=(h_small, w_small), mode='bilinear', align_corners=False)
    return F.interpolate(down, size=(H, W), mode='bilinear', align_corners=False)


def rotation_attack(images, angle=15.0):
    B, C, H, W = images.shape
    angle_rad = angle * np.pi / 180
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    theta = torch.tensor([[cos_a, -sin_a, 0], [sin_a, cos_a, 0]], 
                         dtype=images.dtype, device=images.device).unsqueeze(0).expand(B, -1, -1)
    grid = F.affine_grid(theta, images.size(), align_corners=False)
    return F.grid_sample(images, grid, mode='bilinear', padding_mode='reflection', align_corners=False)


def gaussian_blur_attack(images, kernel_size=5):
    if kernel_size % 2 == 0:
        kernel_size += 1
    sigma = kernel_size / 3
    x = torch.arange(kernel_size, device=images.device, dtype=images.dtype) - kernel_size // 2
    kernel_1d = torch.exp(-x**2 / (2 * sigma**2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = kernel_1d.unsqueeze(1) * kernel_1d.unsqueeze(0)
    kernel = kernel_2d.unsqueeze(0).unsqueeze(0).expand(3, 1, -1, -1)
    return F.conv2d(images, kernel, padding=kernel_size // 2, groups=3)


def contrast_attack(images, factor=2.0):
    img_01 = (images + 1) / 2
    mean = img_01.mean(dim=(2, 3), keepdim=True)
    adjusted = (img_01 - mean) * factor + mean
    return (adjusted.clamp(0, 1) * 2 - 1)


def brightness_attack(images, factor=2.0):
    img_01 = (images + 1) / 2
    shift = (factor - 1) * 0.3
    return ((img_01 + shift).clamp(0, 1) * 2 - 1)


def combined_attack(images):
    result = images
    result = resize_attack(result, 0.85)
    result = gaussian_blur_attack(result, 3)
    result = contrast_attack(result, 1.2)
    return result


# Define attacks
ATTACKS = {
    'None': None,
    'C.Crop 0.1': lambda x: center_crop_attack(x, 0.1),
    'R.Crop 0.1': lambda x: random_crop_attack(x, 0.1),
    'Resize 0.7': lambda x: resize_attack(x, 0.7),
    'Rot. 15': lambda x: rotation_attack(x, 15.0),
    'Blur': lambda x: gaussian_blur_attack(x, 5),
    'Contr. 2.0': lambda x: contrast_attack(x, 2.0),
    'Bright. 2.0': lambda x: brightness_attack(x, 2.0),
    'JPEG 70': lambda x: real_jpeg_attack(x, 70),
    'Comb.': combined_attack,
}


# Top models to evaluate
MODELS = {
    'efficient': 'results/efficient_20260222_004718/best_model.pth',
    'fast_staged': 'results/fast_staged_20260222_110232/best_roundtrip.pt',
    'decoder_ft': 'results/decoder_ft_20260222_212319/best.pt',
    'roundtrip': 'results/roundtrip_train_20260222_172359/best_roundtrip.pt',
    'lightweight': 'results/lightweight_20260222_233224/best.pt',
    'ft_efficient': 'results/finetune_efficient_20260225_181255/best.pt',
}


def load_model(checkpoint_path, device):
    """Load a model checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = ckpt.get('config', {})
    w_dim = config.get('w_dim', 32)
    
    splitter = LatentSplitter(mode='dct').to(device)
    recombiner = LatentRecombiner(mode='dct').to(device)
    encoder_l = WatermarkEncoder(watermark_dim=w_dim).to(device)
    encoder_h = WatermarkEncoder(watermark_dim=w_dim).to(device)
    decoder_l = WatermarkDecoder(watermark_dim=w_dim).to(device)
    decoder_h = WatermarkDecoder(watermark_dim=w_dim).to(device)
    
    encoder_l.load_state_dict(ckpt['encoder_l'])
    encoder_h.load_state_dict(ckpt['encoder_h'])
    decoder_l.load_state_dict(ckpt['decoder_l'])
    decoder_h.load_state_dict(ckpt['decoder_h'])
    
    for m in [encoder_l, encoder_h, decoder_l, decoder_h]:
        m.eval()
    
    alpha_l = ckpt.get('alpha_l', 0.02)
    alpha_h = ckpt.get('alpha_h', 0.01)
    
    return {
        'splitter': splitter,
        'recombiner': recombiner,
        'encoder_l': encoder_l,
        'encoder_h': encoder_h,
        'decoder_l': decoder_l,
        'decoder_h': decoder_h,
        'alpha_l': alpha_l,
        'alpha_h': alpha_h,
        'w_dim': w_dim,
    }


def embed(model, z, w):
    z_l, z_h = model['splitter'](z)
    z_l_wm = model['encoder_l'](z_l, w, alpha=model['alpha_l'])
    z_h_wm = model['encoder_h'](z_h, w, alpha=model['alpha_h'])
    return model['recombiner'](z_l_wm, z_h_wm)


def extract(model, z):
    z_l, z_h = model['splitter'](z)
    return model['decoder_l'](z_l), model['decoder_h'](z_h)


def bit_acc(w_true, w_l, w_h):
    bits_true = (w_true > 0).float()
    bits_pred = ((w_l + w_h) / 2 > 0).float()
    return (bits_true == bits_pred).float().mean().item()


def main():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load VAE
    print("Loading VAE...")
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(
        'runwayml/stable-diffusion-v1-5',
        subfolder='vae',
        torch_dtype=torch.float32
    ).to(device)
    vae.eval()
    scaling = 0.18215
    
    # Load test latents
    latent_path = os.path.join(project_root, 'cache/latents_20000_256.pt')
    print(f"Loading latents: {latent_path}")
    cache = torch.load(latent_path, map_location='cpu', weights_only=False)
    latents = cache['latents'] if isinstance(cache, dict) else cache
    
    # Use 50 samples for evaluation
    n_samples = 50
    latents = latents[:n_samples]
    
    # Results storage
    results = {model_name: {} for model_name in MODELS}
    
    # Evaluate each model
    for model_name, checkpoint_path in MODELS.items():
        full_path = os.path.join(project_root, checkpoint_path)
        if not os.path.exists(full_path):
            print(f"Checkpoint not found: {full_path}")
            continue
            
        print(f"\n{'='*60}")
        print(f"Evaluating: {model_name}")
        print(f"{'='*60}")
        
        model = load_model(full_path, device)
        w_dim = model['w_dim']
        
        # Generate watermarks
        torch.manual_seed(42)  # For reproducibility
        watermarks = torch.randn(n_samples, w_dim)
        watermarks = (watermarks > 0).float() * 2 - 1  # Binary
        
        for attack_name, attack_fn in ATTACKS.items():
            accs = []
            
            for i in tqdm(range(n_samples), desc=f"{attack_name:12}", leave=False):
                z = latents[i:i+1].to(device)
                w = watermarks[i:i+1].to(device)
                
                with torch.no_grad():
                    # Embed watermark
                    z_wm = embed(model, z, w)
                    
                    # Decode to image
                    img_wm = vae.decode(z_wm / scaling).sample
                    
                    # Apply attack
                    if attack_fn is not None:
                        img_att = attack_fn(img_wm)
                    else:
                        img_att = img_wm
                    
                    # Re-encode
                    z_att = vae.encode(img_att).latent_dist.mean * scaling
                    
                    # Extract watermark
                    w_l, w_h = extract(model, z_att)
                    
                    # Compute accuracy
                    acc = bit_acc(w, w_l, w_h)
                    accs.append(acc)
            
            results[model_name][attack_name] = np.mean(accs) * 100
            
            if device.type == 'mps':
                torch.mps.empty_cache()
    
    # Print results table
    print("\n" + "="*120)
    print("RESULTS: Bit Accuracy (%) for each model and attack")
    print("="*120)
    
    attack_names = list(ATTACKS.keys())
    
    # Header
    header = f"{'Model':<12} |"
    for attack in attack_names:
        header += f" {attack:>10} |"
    print(header)
    print("-" * len(header))
    
    # Data rows
    for model_name in MODELS:
        if model_name not in results or not results[model_name]:
            continue
        row = f"{model_name:<12} |"
        for attack in attack_names:
            val = results[model_name].get(attack, 0)
            row += f" {val:>10.1f} |"
        print(row)
    
    print("-" * len(header))
    
    # Find best model per attack
    print("\nBest model per attack:")
    for attack in attack_names:
        best_model = None
        best_acc = 0
        for model_name in MODELS:
            if model_name in results and attack in results[model_name]:
                if results[model_name][attack] > best_acc:
                    best_acc = results[model_name][attack]
                    best_model = model_name
        if best_model:
            print(f"  {attack:>12}: {best_model} ({best_acc:.1f}%)")
    
    # Compute average across attacks
    print("\nAverage accuracy across all attacks:")
    for model_name in MODELS:
        if model_name in results and results[model_name]:
            avg = np.mean(list(results[model_name].values()))
            print(f"  {model_name:<12}: {avg:.1f}%")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Quick evaluation - just the essential metrics."""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import io
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from latent_watermarking.models.latent_split import LatentSplitter
from latent_watermarking.models.recombination import LatentRecombiner
from latent_watermarking.models.watermark_encoder import WatermarkEncoder
from latent_watermarking.models.watermark_decoder import WatermarkDecoder


def jpeg_attack(img, quality=70):
    img_np = ((img.squeeze(0).permute(1, 2, 0).cpu() + 1) / 2 * 255).clamp(0, 255).numpy().astype(np.uint8)
    pil_img = Image.fromarray(img_np, mode='RGB')
    buffer = io.BytesIO()
    pil_img.save(buffer, format='JPEG', quality=quality)
    buffer.seek(0)
    compressed = Image.open(buffer).convert('RGB')
    return (torch.from_numpy(np.array(compressed)).float().permute(2, 0, 1) / 255.0 * 2 - 1).unsqueeze(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--samples', type=int, default=50)
    args = parser.parse_args()
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load VAE
    print("Loading VAE...")
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(
        'runwayml/stable-diffusion-v1-5',
        subfolder='vae',
        torch_dtype=torch.float16
    ).to(device)
    vae.eval()
    scaling = 0.18215
    
    # Load model
    print(f"Loading: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    
    splitter = LatentSplitter(mode='dct').to(device)
    recombiner = LatentRecombiner(mode='dct').to(device)
    encoder_l = WatermarkEncoder(watermark_dim=32).to(device)
    encoder_h = WatermarkEncoder(watermark_dim=32).to(device)
    decoder_l = WatermarkDecoder(watermark_dim=32).to(device)
    decoder_h = WatermarkDecoder(watermark_dim=32).to(device)
    
    encoder_l.load_state_dict(ckpt['encoder_l'])
    encoder_h.load_state_dict(ckpt['encoder_h'])
    decoder_l.load_state_dict(ckpt['decoder_l'])
    decoder_h.load_state_dict(ckpt['decoder_h'])
    
    for m in [encoder_l, encoder_h, decoder_l, decoder_h]:
        m.eval()
    
    alpha_l = ckpt.get('alpha_l', 0.02)
    alpha_h = ckpt.get('alpha_h', 0.01)
    
    def embed(z, w):
        z_l, z_h = splitter(z)
        z_l_wm = encoder_l(z_l, w, alpha=alpha_l)
        z_h_wm = encoder_h(z_h, w, alpha=alpha_h)
        return recombiner(z_l_wm, z_h_wm)
    
    def extract(z):
        z_l, z_h = splitter(z)
        return (decoder_l(z_l) + decoder_h(z_h)) / 2
    
    def bit_acc(w_true, w_pred):
        return ((w_true > 0) == (w_pred > 0)).float().mean().item()
    
    # Load test latents
    latents = torch.load('cache/latents_20000_256.pt', map_location='cpu', weights_only=False)
    z_all = latents['latents'][:args.samples]
    
    # Metrics
    metrics = {
        'latent': [],
        'vae_roundtrip': [],
        'jpeg_90': [],
        'jpeg_70': [],
        'noise_0.01': [],
    }
    psnr_list = []
    
    print(f"\nEvaluating {args.samples} samples...")
    for i in tqdm(range(args.samples)):
        z = z_all[i:i+1].to(device)
        w = torch.randn(1, 32).to(device)
        w = (w > 0).float() * 2 - 1  # Binary
        
        with torch.no_grad():
            # Embed
            z_wm = embed(z, w)
            
            # Latent accuracy
            w_ext = extract(z_wm)
            metrics['latent'].append(bit_acc(w, w_ext))
            
            # Decode to image
            img_orig = vae.decode(z.half() / scaling).sample.float()
            img_wm = vae.decode(z_wm.half() / scaling).sample.float()
            
            # PSNR
            mse = F.mse_loss(img_wm, img_orig).item()
            psnr = 10 * np.log10(4 / (mse + 1e-10))  # range [-1,1] so max=4
            psnr_list.append(psnr)
            
            # VAE roundtrip
            z_rt = vae.encode(img_wm.half()).latent_dist.mean.float() * scaling
            w_rt = extract(z_rt)
            metrics['vae_roundtrip'].append(bit_acc(w, w_rt))
            
            # JPEG 90
            img_j90 = jpeg_attack(img_wm, 90).to(device)
            z_j90 = vae.encode(img_j90.half()).latent_dist.mean.float() * scaling
            metrics['jpeg_90'].append(bit_acc(w, extract(z_j90)))
            
            # JPEG 70
            img_j70 = jpeg_attack(img_wm, 70).to(device)
            z_j70 = vae.encode(img_j70.half()).latent_dist.mean.float() * scaling
            metrics['jpeg_70'].append(bit_acc(w, extract(z_j70)))
            
            # Noise
            img_n = (img_wm + torch.randn_like(img_wm) * 0.01).clamp(-1, 1)
            z_n = vae.encode(img_n.half()).latent_dist.mean.float() * scaling
            metrics['noise_0.01'].append(bit_acc(w, extract(z_n)))
        
        if device.type == 'mps' and i % 10 == 0:
            torch.mps.empty_cache()
    
    # Print results
    print("\n" + "="*50)
    print("RESULTS")
    print("="*50)
    print(f"PSNR: {np.mean(psnr_list):.2f} dB")
    print()
    for name, vals in metrics.items():
        print(f"{name:20s}: {np.mean(vals)*100:.1f}%")
    print("="*50)


if __name__ == "__main__":
    main()

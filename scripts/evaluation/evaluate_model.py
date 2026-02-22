#!/usr/bin/env python3
"""Evaluate trained watermarking model."""

import os
import sys
import torch
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from latent_watermarking.models.latent_split import LatentSplitter
from latent_watermarking.models.recombination import LatentRecombiner
from latent_watermarking.models.watermark_encoder import WatermarkEncoder
from latent_watermarking.models.watermark_decoder import WatermarkDecoder
from latent_watermarking.attacks.latent_noise import LatentNoiseAttack


def main():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Find latest checkpoint
    results_dir = os.path.join(os.path.dirname(__file__), 'results')
    runs = sorted([d for d in os.listdir(results_dir) if d.startswith('efficient_')])
    if not runs:
        print("No training runs found!")
        return
    
    latest_run = runs[-1]
    checkpoint_path = os.path.join(results_dir, latest_run, 'best_model.pth')
    print(f"Loading: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint['config']
    
    # Initialize models
    w_dim = config.get('w_dim', 32)
    splitter = LatentSplitter(mode=config.get('latent_split', 'dct')).to(device)
    recombiner = LatentRecombiner(mode=config.get('latent_split', 'dct')).to(device)
    encoder_l = WatermarkEncoder(watermark_dim=w_dim).to(device)
    encoder_h = WatermarkEncoder(watermark_dim=w_dim).to(device)
    decoder_l = WatermarkDecoder(watermark_dim=w_dim).to(device)
    decoder_h = WatermarkDecoder(watermark_dim=w_dim).to(device)
    attack = LatentNoiseAttack().to(device)
    
    # Load weights
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
    
    # Load cached latents
    cache_dir = os.path.join(os.path.dirname(__file__), 'cache')
    cache_files = [f for f in os.listdir(cache_dir) if f.endswith('.pt')]
    if not cache_files:
        print("No cached latents found!")
        return
    
    cache_path = os.path.join(cache_dir, 'latents_10000_128.pt')
    if not os.path.exists(cache_path):
        cache_path = os.path.join(cache_dir, sorted(cache_files)[-1])
    print(f"Loading latents: {cache_path}")
    latents = torch.load(cache_path, map_location='cpu')['latents']
    print(f"Evaluating on {len(latents)} samples")
    
    # Metrics
    bit_acc_clean = []
    bit_acc_attacked = []
    latent_mse = []
    latent_psnr = []
    
    num_eval = min(1000, len(latents))
    
    with torch.no_grad():
        for i in tqdm(range(0, num_eval, 16), desc="Evaluating"):
            batch = latents[i:i+16].to(device)
            B = batch.shape[0]
            
            # Random watermark
            w = torch.randn(B, w_dim, device=device)
            
            # Encode watermark
            z_low, z_high = splitter(batch)
            z_low_wm = encoder_l(z_low, w, alpha=alpha_l)
            z_high_wm = encoder_h(z_high, w, alpha=alpha_h)
            z_wm = recombiner(z_low_wm, z_high_wm)
            
            # Latent quality
            mse = ((batch - z_wm) ** 2).mean().item()
            psnr = 10 * np.log10(1.0 / (mse + 1e-10))
            latent_mse.append(mse)
            latent_psnr.append(psnr)
            
            # Decode clean
            z_wm_low, z_wm_high = splitter(z_wm)
            w_pred_l = decoder_l(z_wm_low)
            w_pred_h = decoder_h(z_wm_high)
            w_pred = (w_pred_l + w_pred_h) / 2
            
            # Bit accuracy clean
            bits_true = (w > 0).float()
            bits_pred = (w_pred > 0).float()
            acc = (bits_true == bits_pred).float().mean().item()
            bit_acc_clean.append(acc)
            
            # After attack
            attack.eval()
            z_attacked = attack(z_wm)
            z_att_low, z_att_high = splitter(z_attacked)
            w_pred_att = (decoder_l(z_att_low) + decoder_h(z_att_high)) / 2
            
            bits_pred_att = (w_pred_att > 0).float()
            acc_att = (bits_true == bits_pred_att).float().mean().item()
            bit_acc_attacked.append(acc_att)
            
            if device.type == 'mps':
                torch.mps.empty_cache()
    
    print("\n" + "="*60)
    print("EVALUATION METRICS")
    print("="*60)
    
    print(f"\nWatermark Recovery:")
    print(f"  Bit Accuracy (clean):    {np.mean(bit_acc_clean)*100:.2f}% +/- {np.std(bit_acc_clean)*100:.2f}%")
    print(f"  Bit Accuracy (attacked): {np.mean(bit_acc_attacked)*100:.2f}% +/- {np.std(bit_acc_attacked)*100:.2f}%")
    
    print(f"\nLatent Quality:")
    print(f"  MSE:  {np.mean(latent_mse):.6f} +/- {np.std(latent_mse):.6f}")
    print(f"  PSNR: {np.mean(latent_psnr):.2f} dB +/- {np.std(latent_psnr):.2f} dB")
    
    print(f"\nModel Config:")
    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"  Alpha L/H: {alpha_l:.3f}/{alpha_h:.3f}")
    print(f"  Watermark dim: {w_dim} bits")
    print(f"  Training loss: {checkpoint['metrics']['loss']:.4f}")
    print(f"  Training bit acc: {checkpoint['metrics']['bit_acc']*100:.2f}%")
    print("="*60)


if __name__ == '__main__':
    main()

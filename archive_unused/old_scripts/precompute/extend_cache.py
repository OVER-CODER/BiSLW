#!/usr/bin/env python3
"""
Extend existing roundtrip cache with more samples.
Reuses previously computed samples to save time.
"""

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


class VAEWrapper:
    def __init__(self, model_id='runwayml/stable-diffusion-v1-5', device='cpu'):
        self.device = device
        self.model_id = model_id
        self._vae = None
        self.scaling_factor = 0.18215
        
    def load(self):
        if self._vae is None:
            print("Loading VAE...")
            from diffusers import AutoencoderKL
            self._vae = AutoencoderKL.from_pretrained(
                self.model_id, subfolder='vae', torch_dtype=torch.float32
            ).to(self.device)
            self._vae.eval()
            for p in self._vae.parameters():
                p.requires_grad = False
        return self._vae
    
    def unload(self):
        if self._vae is not None:
            del self._vae
            self._vae = None
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            print("VAE unloaded")
    
    @torch.no_grad()
    def roundtrip(self, z):
        vae = self.load()
        z_scaled = z / self.scaling_factor
        img = vae.decode(z_scaled).sample
        z_rt = vae.encode(img).latent_dist.mean * self.scaling_factor
        return z_rt


def main():
    import argparse
    import yaml
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--existing-cache', type=str, required=True, help='Path to existing cache')
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint')
    parser.add_argument('--output-cache', type=str, required=True, help='Output cache path')
    parser.add_argument('--target-samples', type=int, default=20000)
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    args = parser.parse_args()
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Load existing cache
    print(f"Loading existing cache: {args.existing_cache}")
    existing = torch.load(args.existing_cache, map_location='cpu', weights_only=False)
    existing_z_orig = existing['z_orig']
    existing_watermarks = existing['watermarks']
    existing_z_roundtrip = existing['z_roundtrip']
    existing_count = len(existing_z_orig)
    print(f"Existing samples: {existing_count}")
    
    needed = args.target_samples - existing_count
    if needed <= 0:
        print(f"Already have {existing_count} samples, target is {args.target_samples}. Nothing to do.")
        return
    
    print(f"Need to compute {needed} more samples")
    
    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    
    w_dim = config.get('w_dim', 32)
    splitter = LatentSplitter(mode=config.get('latent_split', 'dct')).to(device)
    recombiner = LatentRecombiner(mode=config.get('latent_split', 'dct')).to(device)
    encoder_l = WatermarkEncoder(watermark_dim=w_dim).to(device)
    encoder_h = WatermarkEncoder(watermark_dim=w_dim).to(device)
    
    encoder_l.load_state_dict(checkpoint['encoder_l'])
    encoder_h.load_state_dict(checkpoint['encoder_h'])
    encoder_l.eval()
    encoder_h.eval()
    
    alpha_l = checkpoint.get('alpha_l', 0.02)
    alpha_h = checkpoint.get('alpha_h', 0.01)
    
    # Load all latents
    num_images = config.get('num_images', 20000)
    image_size = config.get('image_size', 256)
    latent_path = f"cache/latents_{num_images}_{image_size}.pt"
    
    print(f"Loading latents: {latent_path}")
    data = torch.load(latent_path, map_location='cpu', weights_only=False)
    latents = data['latents']
    
    # Select new indices (avoid overlap with existing)
    # Existing used first 5000 indices, so use 5000-20000
    new_indices = torch.arange(existing_count, existing_count + needed)
    if new_indices[-1] >= len(latents):
        # Wrap around or use random
        new_indices = torch.randperm(len(latents))[:needed]
    
    # Compute new roundtrip samples
    print(f"Computing {needed} new roundtrip samples...")
    
    vae = VAEWrapper(device=device)
    
    new_z_orig = []
    new_watermarks = []
    new_z_roundtrip = []
    
    batch_size = 4
    
    for i in tqdm(range(0, needed, batch_size), desc="Computing roundtrip"):
        idx = new_indices[i:i+batch_size]
        z_batch = latents[idx].to(device)
        B = z_batch.shape[0]
        
        # Generate watermarks
        w_batch = torch.randn(B, w_dim, device=device)
        
        # Embed watermark
        with torch.no_grad():
            z_l, z_h = splitter(z_batch)
            z_l_wm = encoder_l(z_l, w_batch, alpha=alpha_l)
            z_h_wm = encoder_h(z_h, w_batch, alpha=alpha_h)
            z_wm = recombiner(z_l_wm, z_h_wm)
            
            # VAE roundtrip
            z_rt = vae.roundtrip(z_wm)
        
        new_z_orig.append(z_batch.cpu())
        new_watermarks.append(w_batch.cpu())
        new_z_roundtrip.append(z_rt.cpu())
        
        if device.type == 'mps':
            torch.mps.empty_cache()
    
    vae.unload()
    
    # Concatenate new samples
    new_z_orig = torch.cat(new_z_orig, dim=0)
    new_watermarks = torch.cat(new_watermarks, dim=0)
    new_z_roundtrip = torch.cat(new_z_roundtrip, dim=0)
    
    # Merge with existing
    merged_z_orig = torch.cat([existing_z_orig, new_z_orig], dim=0)
    merged_watermarks = torch.cat([existing_watermarks, new_watermarks], dim=0)
    merged_z_roundtrip = torch.cat([existing_z_roundtrip, new_z_roundtrip], dim=0)
    
    print(f"Merged: {len(merged_z_orig)} total samples")
    
    # Save
    os.makedirs(os.path.dirname(args.output_cache), exist_ok=True)
    torch.save({
        'z_orig': merged_z_orig,
        'watermarks': merged_watermarks,
        'z_roundtrip': merged_z_roundtrip
    }, args.output_cache)
    
    print(f"Saved to: {args.output_cache}")
    print("Done!")


if __name__ == "__main__":
    main()

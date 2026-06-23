#!/usr/bin/env python3
"""
Train on precomputed roundtrip cache.
Fast training without VAE in the loop.
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from datetime import datetime
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from latent_watermarking.models.latent_split import LatentSplitter
from latent_watermarking.models.recombination import LatentRecombiner
from latent_watermarking.models.watermark_encoder import WatermarkEncoder
from latent_watermarking.models.watermark_decoder import WatermarkDecoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache', type=str, required=True, help='Roundtrip cache path')
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint to resume from')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--eval-interval', type=int, default=10)
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    args = parser.parse_args()
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")
    
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Load cache
    print(f"Loading roundtrip cache: {args.cache}")
    cache = torch.load(args.cache, map_location='cpu', weights_only=False)
    z_orig = cache['z_orig']
    watermarks = cache['watermarks']
    z_roundtrip = cache['z_roundtrip']
    print(f"Loaded {len(z_orig)} samples")
    
    # Load model
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    
    w_dim = config.get('w_dim', 32)
    splitter = LatentSplitter(mode=config.get('latent_split', 'dct')).to(device)
    recombiner = LatentRecombiner(mode=config.get('latent_split', 'dct')).to(device)
    encoder_l = WatermarkEncoder(watermark_dim=w_dim).to(device)
    encoder_h = WatermarkEncoder(watermark_dim=w_dim).to(device)
    decoder_l = WatermarkDecoder(watermark_dim=w_dim).to(device)
    decoder_h = WatermarkDecoder(watermark_dim=w_dim).to(device)
    
    encoder_l.load_state_dict(checkpoint['encoder_l'])
    encoder_h.load_state_dict(checkpoint['encoder_h'])
    decoder_l.load_state_dict(checkpoint['decoder_l'])
    decoder_h.load_state_dict(checkpoint['decoder_h'])
    
    alpha_l = checkpoint.get('alpha_l', 0.02)
    alpha_h = checkpoint.get('alpha_h', 0.01)
    print(f"Alpha: {alpha_l}/{alpha_h}")
    
    # Optimizer
    all_params = (
        list(encoder_l.parameters()) + list(encoder_h.parameters()) +
        list(decoder_l.parameters()) + list(decoder_h.parameters())
    )
    optimizer = torch.optim.AdamW(all_params, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Output dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"results/roundtrip_train_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Dataset
    dataset = TensorDataset(z_orig, watermarks, z_roundtrip)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    
    print(f"\nTraining for {args.epochs} epochs on {len(z_orig)} samples")
    print(f"Output: {output_dir}\n")
    
    best_rt_acc = 0.0
    
    for epoch in range(args.epochs):
        encoder_l.train()
        encoder_h.train()
        decoder_l.train()
        decoder_h.train()
        
        epoch_loss = []
        epoch_rt_acc = []
        epoch_clean_acc = []
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        for z_batch, w_batch, z_rt_batch in pbar:
            z_batch = z_batch.to(device)
            w_batch = w_batch.to(device)
            z_rt_batch = z_rt_batch.to(device)
            
            # Extract from roundtrip latent (primary objective)
            z_l_rt, z_h_rt = splitter(z_rt_batch)
            w_pred_rt_l = decoder_l(z_l_rt)
            w_pred_rt_h = decoder_h(z_h_rt)
            
            # Also train on fresh embedding
            z_l, z_h = splitter(z_batch)
            z_l_wm = encoder_l(z_l, w_batch, alpha=alpha_l)
            z_h_wm = encoder_h(z_h, w_batch, alpha=alpha_h)
            z_wm = recombiner(z_l_wm, z_h_wm)
            
            z_l_wm2, z_h_wm2 = splitter(z_wm)
            w_pred_l = decoder_l(z_l_wm2)
            w_pred_h = decoder_h(z_h_wm2)
            
            # Losses
            loss_rt = F.mse_loss(w_pred_rt_l, w_batch) + F.mse_loss(w_pred_rt_h, w_batch)
            loss_clean = F.mse_loss(w_pred_l, w_batch) + F.mse_loss(w_pred_h, w_batch)
            loss_cons = F.mse_loss(w_pred_rt_l, w_pred_rt_h)
            loss_latent = F.mse_loss(z_wm, z_batch)
            
            total_loss = 2.0 * loss_rt + 0.5 * loss_clean + 0.3 * loss_cons + 1.0 * loss_latent
            
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, 1.0)
            optimizer.step()
            
            # Metrics
            with torch.no_grad():
                bits_true = (w_batch > 0).float()
                bits_pred_rt = ((w_pred_rt_l + w_pred_rt_h) / 2 > 0).float()
                bits_pred_clean = ((w_pred_l + w_pred_h) / 2 > 0).float()
                rt_acc = (bits_true == bits_pred_rt).float().mean().item()
                clean_acc = (bits_true == bits_pred_clean).float().mean().item()
            
            epoch_loss.append(total_loss.item())
            epoch_rt_acc.append(rt_acc)
            epoch_clean_acc.append(clean_acc)
            
            pbar.set_postfix(loss=f"{total_loss.item():.3f}", rt_acc=f"{rt_acc:.3f}")
            
            if device.type == 'mps':
                torch.mps.empty_cache()
        
        scheduler.step()
        
        avg_loss = np.mean(epoch_loss)
        avg_rt_acc = np.mean(epoch_rt_acc)
        avg_clean_acc = np.mean(epoch_clean_acc)
        
        print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, RT Acc={avg_rt_acc:.4f}, Clean Acc={avg_clean_acc:.4f}")
        
        if avg_rt_acc > best_rt_acc:
            best_rt_acc = avg_rt_acc
            torch.save({
                'epoch': epoch + 1,
                'encoder_l': encoder_l.state_dict(),
                'encoder_h': encoder_h.state_dict(),
                'decoder_l': decoder_l.state_dict(),
                'decoder_h': decoder_h.state_dict(),
                'optimizer': optimizer.state_dict(),
                'alpha_l': alpha_l,
                'alpha_h': alpha_h,
                'metrics': {'rt_acc': avg_rt_acc, 'clean_acc': avg_clean_acc},
                'config': config
            }, f"{output_dir}/best_roundtrip.pt")
            print(f"  New best: {best_rt_acc:.4f}")
        
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch + 1,
                'encoder_l': encoder_l.state_dict(),
                'encoder_h': encoder_h.state_dict(),
                'decoder_l': decoder_l.state_dict(),
                'decoder_h': decoder_h.state_dict(),
                'alpha_l': alpha_l,
                'alpha_h': alpha_h,
                'config': config
            }, f"{output_dir}/epoch{epoch+1}.pt")
    
    # Save final
    torch.save({
        'epoch': args.epochs,
        'encoder_l': encoder_l.state_dict(),
        'encoder_h': encoder_h.state_dict(),
        'decoder_l': decoder_l.state_dict(),
        'decoder_h': decoder_h.state_dict(),
        'alpha_l': alpha_l,
        'alpha_h': alpha_h,
        'config': config
    }, f"{output_dir}/final.pt")
    
    print(f"\nTraining complete!")
    print(f"Best RT accuracy: {best_rt_acc:.4f}")
    print(f"Results: {output_dir}")


if __name__ == "__main__":
    main()

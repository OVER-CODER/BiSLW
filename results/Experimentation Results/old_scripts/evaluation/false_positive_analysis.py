#!/usr/bin/env python3
"""
False Positive Analysis for Latent Watermarking
- ROC curve (TPR vs FPR) for watermark detection
- Attribution accuracy vs number of users
- Detection threshold analysis
- Score distribution analysis
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from sklearn.metrics import roc_curve, auc

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.latent_split import LatentSplitter
from models.recombination import LatentRecombiner
from models.watermark_encoder import WatermarkEncoder
from models.watermark_decoder import WatermarkDecoder


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_bit_accuracy(extracted, target):
    """Compute bit accuracy between extracted and target watermarks."""
    extracted_bits = (extracted > 0).float()
    target_bits = (target > 0).float()
    return (extracted_bits == target_bits).float().mean(dim=-1)


def train_models(device, w_dim=32, epochs=150, alpha_l=0.1, alpha_h=0.05, n_train=2000):
    """Train watermark encoder/decoder pair following the BiSLW architecture."""
    # Initialize all models
    splitter = LatentSplitter(mode='dct').to(device)
    recombiner = LatentRecombiner(mode='dct').to(device)
    encoder_l = WatermarkEncoder(watermark_dim=w_dim).to(device)
    encoder_h = WatermarkEncoder(watermark_dim=w_dim).to(device)
    decoder_l = WatermarkDecoder(watermark_dim=w_dim).to(device)
    decoder_h = WatermarkDecoder(watermark_dim=w_dim).to(device)
    
    # Generate training latents
    latents = torch.randn(n_train, 4, 64, 64)
    
    params = (
        list(encoder_l.parameters()) + list(encoder_h.parameters()) +
        list(decoder_l.parameters()) + list(decoder_h.parameters())
    )
    optimizer = torch.optim.AdamW(params, lr=2e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    batch_size = 32
    n_batches = len(latents) // batch_size
    
    # Training
    print_interval = max(1, epochs // 10)  # Print ~10 times during training
    for epoch in range(epochs):
        encoder_l.train()
        encoder_h.train()
        decoder_l.train()
        decoder_h.train()
        
        indices = torch.randperm(len(latents))
        
        for b in range(n_batches):
            idx = indices[b * batch_size:(b + 1) * batch_size]
            z = latents[idx].to(device)
            w = torch.randn(batch_size, w_dim, device=device)
            
            # Forward
            z_low, z_high = splitter(z)
            z_low_wm = encoder_l(z_low, w, alpha=alpha_l)
            z_high_wm = encoder_h(z_high, w, alpha=alpha_h)
            z_wm = recombiner(z_low_wm, z_high_wm)
            
            z_wm_low, z_wm_high = splitter(z_wm)
            w_pred_l = decoder_l(z_wm_low)
            w_pred_h = decoder_h(z_wm_high)
            
            # Losses
            loss_w = F.mse_loss(w_pred_l, w) + F.mse_loss(w_pred_h, w)
            loss_cons = F.mse_loss(w_pred_l, w_pred_h)
            loss_latent = F.mse_loss(z_wm, z)
            
            loss = loss_w + 0.3 * loss_cons + 5.0 * loss_latent
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
        
        scheduler.step()
        
        if (epoch + 1) % print_interval == 0 or epoch == 0:
            print(f"  Epoch {epoch+1}/{epochs}")
    
    # Set to evaluation mode
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
    z_low, z_high = models['splitter'](z)
    z_low_wm = models['encoder_l'](z_low, w, alpha=models['alpha_l'])
    z_high_wm = models['encoder_h'](z_high, w, alpha=models['alpha_h'])
    z_wm = models['recombiner'](z_low_wm, z_high_wm)
    return z_wm


def extract_watermark(models, z_wm):
    """Extract watermark from latent."""
    z_wm_low, z_wm_high = models['splitter'](z_wm)
    w_pred_l = models['decoder_l'](z_wm_low)
    w_pred_h = models['decoder_h'](z_wm_high)
    # Average the predictions from both bands
    return (w_pred_l + w_pred_h) / 2


def run_detection_analysis(models, device, w_dim=32, n_samples=500):
    """
    Analyze detection performance (TPR/FPR).
    Compare watermarked vs non-watermarked images.
    """
    print("  Running detection analysis...")
    
    with torch.no_grad():
        # Generate watermarked samples
        z = torch.randn(n_samples, 4, 64, 64, device=device)
        w_true = torch.randn(n_samples, w_dim, device=device)
        
        z_watermarked = embed_watermark(models, z, w_true)
        w_extracted_wm = extract_watermark(models, z_watermarked)
        
        # Generate non-watermarked samples (clean latents)
        z_clean = torch.randn(n_samples, 4, 64, 64, device=device)
        w_extracted_clean = extract_watermark(models, z_clean)
        
        # Compute detection scores using bit accuracy
        acc_wm = compute_bit_accuracy(w_extracted_wm, w_true).cpu().numpy()
        
        # For clean images, compare with random watermarks (should be ~0.5)
        w_random = torch.randn(n_samples, w_dim, device=device)
        acc_clean = compute_bit_accuracy(w_extracted_clean, w_random).cpu().numpy()
        
        # Correlation-based scores
        w_extracted_wm_norm = F.normalize(w_extracted_wm, dim=-1)
        w_true_norm = F.normalize(w_true, dim=-1)
        corr_wm = (w_extracted_wm_norm * w_true_norm).sum(dim=-1).cpu().numpy()
        
        w_extracted_clean_norm = F.normalize(w_extracted_clean, dim=-1)
        w_random_norm = F.normalize(w_random, dim=-1)
        corr_clean = (w_extracted_clean_norm * w_random_norm).sum(dim=-1).cpu().numpy()
    
    # Create labels: 1 for watermarked, 0 for clean
    y_true = np.concatenate([np.ones(n_samples), np.zeros(n_samples)])
    
    # ROC analysis using bit accuracy
    scores_acc = np.concatenate([acc_wm, acc_clean])
    fpr_acc, tpr_acc, thresholds_acc = roc_curve(y_true, scores_acc)
    auc_acc = auc(fpr_acc, tpr_acc)
    
    # ROC analysis using correlation
    scores_corr = np.concatenate([corr_wm, corr_clean])
    fpr_corr, tpr_corr, thresholds_corr = roc_curve(y_true, scores_corr)
    auc_corr = auc(fpr_corr, tpr_corr)
    
    return {
        'bit_accuracy': {
            'fpr': fpr_acc.tolist(),
            'tpr': tpr_acc.tolist(),
            'thresholds': thresholds_acc.tolist(),
            'auc': auc_acc,
            'scores_wm': acc_wm.tolist(),
            'scores_clean': acc_clean.tolist()
        },
        'correlation': {
            'fpr': fpr_corr.tolist(),
            'tpr': tpr_corr.tolist(),
            'thresholds': thresholds_corr.tolist(),
            'auc': auc_corr,
            'scores_wm': corr_wm.tolist(),
            'scores_clean': corr_clean.tolist()
        }
    }


def run_attribution_analysis(models, device, w_dim=32, 
                             user_counts=[2, 5, 10, 20, 50, 100, 200, 500, 1000],
                             n_samples_per_user=20):
    """
    Analyze attribution accuracy with different numbers of users.
    Each user has a unique watermark.
    Uses batched processing for efficiency.
    """
    print("  Running attribution analysis...")
    results = {}
    
    for n_users in user_counts:
        print(f"    Testing {n_users} users...")
        
        with torch.no_grad():
            # Generate unique watermarks for each user
            user_watermarks = torch.randn(n_users, w_dim, device=device)
            
            n_test_users = min(n_users, 50)  # Test up to 50 users
            total_samples = n_test_users * n_samples_per_user
            
            # Generate all test samples at once (batched)
            z_all = torch.randn(total_samples, 4, 64, 64, device=device)
            
            # Create user IDs and corresponding watermarks for all samples
            user_ids = torch.arange(n_test_users, device=device).repeat_interleave(n_samples_per_user)
            w_all = user_watermarks[user_ids]  # [total_samples, w_dim]
            
            # Process in batches to avoid OOM
            batch_size = 64
            all_predictions = []
            
            for i in range(0, total_samples, batch_size):
                end_idx = min(i + batch_size, total_samples)
                z_batch = z_all[i:end_idx]
                w_batch = w_all[i:end_idx]
                
                # Embed and extract watermarks
                z_wm = embed_watermark(models, z_batch, w_batch)
                w_extracted = extract_watermark(models, z_wm)
                
                # Attribution: find closest user watermark using cosine similarity
                # w_extracted: [batch, w_dim], user_watermarks: [n_users, w_dim]
                w_extracted_norm = F.normalize(w_extracted, dim=-1)
                user_wm_norm = F.normalize(user_watermarks, dim=-1)
                
                # Compute similarities: [batch, n_users]
                similarities = torch.mm(w_extracted_norm, user_wm_norm.T)
                predictions = similarities.argmax(dim=1)
                all_predictions.append(predictions)
            
            # Combine predictions and compute accuracy
            all_predictions = torch.cat(all_predictions)
            correct = (all_predictions == user_ids).sum().item()
            
            accuracy = correct / total_samples
            results[str(n_users)] = {
                'accuracy': accuracy,
                'n_tested': total_samples
            }
            print(f"      Accuracy: {accuracy*100:.1f}%")
    
    return results


def run_threshold_analysis(models, device, w_dim=32, n_samples=300):
    """Analyze detection at different thresholds."""
    print("  Running threshold analysis...")
    
    thresholds = np.linspace(0.5, 1.0, 11)
    
    with torch.no_grad():
        # Watermarked samples
        z = torch.randn(n_samples, 4, 64, 64, device=device)
        w_true = torch.randn(n_samples, w_dim, device=device)
        
        z_watermarked = embed_watermark(models, z, w_true)
        w_extracted_wm = extract_watermark(models, z_watermarked)
        acc_wm = compute_bit_accuracy(w_extracted_wm, w_true).cpu().numpy()
        
        # Clean samples
        z_clean = torch.randn(n_samples, 4, 64, 64, device=device)
        w_extracted_clean = extract_watermark(models, z_clean)
        w_random = torch.randn(n_samples, w_dim, device=device)
        acc_clean = compute_bit_accuracy(w_extracted_clean, w_random).cpu().numpy()
    
    results = {}
    for thresh in thresholds:
        tpr = (acc_wm >= thresh).mean()
        fpr = (acc_clean >= thresh).mean()
        precision = tpr / (tpr + fpr + 1e-8)
        f1 = 2 * tpr * (1-fpr) / (tpr + (1-fpr) + 1e-8) if (tpr + (1-fpr)) > 0 else 0
        results[f"{thresh:.2f}"] = {
            'tpr': float(tpr),
            'fpr': float(fpr),
            'precision': float(precision),
            'f1': float(f1)
        }
    
    return results


def plot_results(detection_results, attribution_results, threshold_results, output_dir):
    """Generate plots for the analysis."""
    
    # Set style
    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'legend.fontsize': 10,
        'figure.figsize': (6, 5)
    })
    
    # Plot 1: ROC Curve (TPR vs FPR)
    fig, ax = plt.subplots(figsize=(6, 5))
    
    # Bit accuracy ROC
    fpr_acc = detection_results['bit_accuracy']['fpr']
    tpr_acc = detection_results['bit_accuracy']['tpr']
    auc_acc = detection_results['bit_accuracy']['auc']
    
    ax.plot(fpr_acc, tpr_acc, 'b-', linewidth=2, 
            label=f'Bit Accuracy (AUC = {auc_acc:.3f})')
    
    # Correlation ROC
    fpr_corr = detection_results['correlation']['fpr']
    tpr_corr = detection_results['correlation']['tpr']
    auc_corr = detection_results['correlation']['auc']
    
    ax.plot(fpr_corr, tpr_corr, 'r--', linewidth=2,
            label=f'Correlation (AUC = {auc_corr:.3f})')
    
    # Diagonal (random)
    ax.plot([0, 1], [0, 1], 'k:', linewidth=1, label='Random')
    
    ax.set_xlabel('False Positive Rate (FPR)')
    ax.set_ylabel('True Positive Rate (TPR)')
    ax.set_title('Watermark Detection ROC Curve')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'roc_curve.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'roc_curve.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved ROC curve")
    
    # Plot 2: Attribution Accuracy vs Number of Users
    fig, ax = plt.subplots(figsize=(6, 5))
    
    n_users = sorted([int(k) for k in attribution_results.keys()])
    accuracies = [attribution_results[str(n)]['accuracy'] * 100 for n in n_users]
    
    ax.semilogx(n_users, accuracies, 'bo-', linewidth=2, markersize=8, label='BiSLW')
    
    # Add random baseline
    random_baseline = [100.0 / n for n in n_users]
    ax.semilogx(n_users, random_baseline, 'r--', linewidth=1.5, 
                label='Random Baseline (1/N)')
    
    ax.set_xlabel('Number of Users')
    ax.set_ylabel('Attribution Accuracy (%)')
    ax.set_title('User Attribution Accuracy')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 105])
    
    # Add annotations for key points
    for n, acc in zip(n_users, accuracies):
        if n in [2, 10, 100, 1000]:
            ax.annotate(f'{acc:.1f}%', (n, acc), textcoords="offset points", 
                       xytext=(0, 10), ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'attribution_accuracy.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'attribution_accuracy.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved attribution accuracy plot")
    
    # Plot 3: Score Distribution (Watermarked vs Clean)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    # Bit accuracy distribution
    ax = axes[0]
    scores_wm = detection_results['bit_accuracy']['scores_wm']
    scores_clean = detection_results['bit_accuracy']['scores_clean']
    
    ax.hist(scores_clean, bins=30, alpha=0.7, label='Non-watermarked', color='red', density=True)
    ax.hist(scores_wm, bins=30, alpha=0.7, label='Watermarked', color='blue', density=True)
    ax.axvline(x=0.5, color='k', linestyle=':', label='Random (0.5)')
    ax.set_xlabel('Bit Accuracy')
    ax.set_ylabel('Density')
    ax.set_title('Detection Score Distribution (Bit Accuracy)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Correlation distribution
    ax = axes[1]
    scores_wm_corr = detection_results['correlation']['scores_wm']
    scores_clean_corr = detection_results['correlation']['scores_clean']
    
    ax.hist(scores_clean_corr, bins=30, alpha=0.7, label='Non-watermarked', color='red', density=True)
    ax.hist(scores_wm_corr, bins=30, alpha=0.7, label='Watermarked', color='blue', density=True)
    ax.axvline(x=0, color='k', linestyle=':', label='Random (0)')
    ax.set_xlabel('Correlation Score')
    ax.set_ylabel('Density')
    ax.set_title('Detection Score Distribution (Correlation)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'score_distribution.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'score_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved score distribution plot")
    
    # Plot 4: TPR/FPR at Different Thresholds
    fig, ax = plt.subplots(figsize=(6, 5))
    
    thresholds = sorted([float(k) for k in threshold_results.keys()])
    tprs = [threshold_results[f"{t:.2f}"]['tpr'] * 100 for t in thresholds]
    fprs = [threshold_results[f"{t:.2f}"]['fpr'] * 100 for t in thresholds]
    
    ax.plot(thresholds, tprs, 'b-o', linewidth=2, markersize=6, label='TPR')
    ax.plot(thresholds, fprs, 'r-s', linewidth=2, markersize=6, label='FPR')
    
    ax.set_xlabel('Detection Threshold (Bit Accuracy)')
    ax.set_ylabel('Rate (%)')
    ax.set_title('TPR/FPR vs Detection Threshold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0.5, 1.0])
    ax.set_ylim([-5, 105])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'threshold_analysis.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'threshold_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved threshold analysis plot")


def main():
    parser = argparse.ArgumentParser(description='False Positive Analysis')
    parser.add_argument('--epochs', type=int, default=150, help='Training epochs')
    parser.add_argument('--w_dim', type=int, default=32, help='Watermark dimension')
    parser.add_argument('--n_samples', type=int, default=500, help='Number of samples for detection')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output_dir', type=str, default='results/false_positive_analysis')
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    # Device selection - prefer MPS on Mac, then CUDA, then CPU
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Training parameters
    alpha_l = 0.1
    alpha_h = 0.05
    
    print("Training watermark model...")
    models = train_models(
        device, w_dim=args.w_dim, epochs=args.epochs,
        alpha_l=alpha_l, alpha_h=alpha_h
    )
    
    print("\nRunning detection analysis (TPR/FPR)...")
    detection_results = run_detection_analysis(
        models, device,
        w_dim=args.w_dim, n_samples=args.n_samples
    )
    print(f"  Bit Accuracy AUC: {detection_results['bit_accuracy']['auc']:.4f}")
    print(f"  Correlation AUC: {detection_results['correlation']['auc']:.4f}")
    
    print("\nRunning attribution analysis...")
    attribution_results = run_attribution_analysis(
        models, device,
        w_dim=args.w_dim, n_samples_per_user=20,
        user_counts=[2, 5, 10, 20, 50, 100, 200, 500, 1000]
    )
    
    print("\nRunning threshold analysis...")
    threshold_results = run_threshold_analysis(
        models, device,
        w_dim=args.w_dim, n_samples=300
    )
    
    print("\nGenerating plots...")
    plot_results(detection_results, attribution_results, threshold_results, output_dir)
    
    # Save results to JSON
    results = {
        'config': {
            'w_dim': args.w_dim,
            'epochs': args.epochs,
            'n_samples': args.n_samples,
            'alpha_l': alpha_l,
            'alpha_h': alpha_h,
            'seed': args.seed
        },
        'detection': {
            'bit_accuracy_auc': detection_results['bit_accuracy']['auc'],
            'correlation_auc': detection_results['correlation']['auc'],
            'mean_wm_score': float(np.mean(detection_results['bit_accuracy']['scores_wm'])),
            'mean_clean_score': float(np.mean(detection_results['bit_accuracy']['scores_clean'])),
            'std_wm_score': float(np.std(detection_results['bit_accuracy']['scores_wm'])),
            'std_clean_score': float(np.std(detection_results['bit_accuracy']['scores_clean']))
        },
        'attribution': attribution_results,
        'threshold_analysis': threshold_results,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(output_dir / 'false_positive_analysis.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_dir}/")
    print("\nSummary:")
    print(f"  Detection AUC (Bit Acc): {detection_results['bit_accuracy']['auc']:.4f}")
    print(f"  Detection AUC (Corr): {detection_results['correlation']['auc']:.4f}")
    print(f"  Attribution Acc (10 users): {attribution_results['10']['accuracy']*100:.1f}%")
    print(f"  Attribution Acc (100 users): {attribution_results['100']['accuracy']*100:.1f}%")


if __name__ == '__main__':
    main()

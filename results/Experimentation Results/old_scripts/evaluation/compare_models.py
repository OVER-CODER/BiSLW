#!/usr/bin/env python3
"""Compare all evaluated models."""

import json
import glob

results = []
for path in glob.glob('results/eval_*/results.json'):
    try:
        with open(path) as f:
            data = json.load(f)
        
        ckpt = data.get('config', {}).get('checkpoint', 'unknown')
        ckpt_short = ckpt.split('/')[-2] if '/' in ckpt else ckpt
        
        psnr = data.get('image_quality', {}).get('psnr_mean', 0)
        ssim = data.get('image_quality', {}).get('ssim_mean', 0)
        latent_acc = data.get('detection', {}).get('bit_accuracy_latent', 0)
        image_acc = data.get('detection', {}).get('bit_accuracy_image', 0)
        
        attacks = data.get('robustness', {})
        avg_attack = sum(a.get('bit_accuracy', 0) for a in attacks.values()) / len(attacks) if attacks else 0
        
        results.append({
            'eval': path.split('/')[-2],
            'model': ckpt_short,
            'psnr': psnr,
            'ssim': ssim,
            'latent': latent_acc,
            'roundtrip': image_acc,
            'attacks': avg_attack
        })
    except Exception as e:
        pass

results.sort(key=lambda x: x['latent'], reverse=True)

print('=' * 110)
print(f"{'Model':<45} {'PSNR':>10} {'SSIM':>10} {'Latent':>10} {'Roundtrip':>10} {'Attacks':>10}")
print('=' * 110)
for r in results:
    print(f"{r['model']:<45} {r['psnr']:>10.2f} {r['ssim']:>10.4f} {r['latent']*100:>9.1f}% {r['roundtrip']*100:>9.1f}% {r['attacks']*100:>9.1f}%")
print('=' * 110)
print("\nTargets: PSNR >= 40 dB, SSIM >= 0.91, Latent Acc > 80%")

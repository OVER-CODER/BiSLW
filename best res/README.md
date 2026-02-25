# Best Models Summary

**Last Updated:** 25 February 2026

## Quick Reference: Best Model by Use Case

| Use Case | Recommended Model | Key Metric |
|----------|-------------------|------------|
| **Best Attack Robustness** | `finetune_efficient_20260225_181255` | 77.7% avg attack acc |
| **Best Detection Accuracy** | `lightweight_20260222_233224` | 94.2% roundtrip acc |
| **Best Image Quality** | `efficient_20260222_004718` | 19.82 dB PSNR, 0.698 SSIM |
| **Best Balance** | `finetune_efficient_20260225_181255` | Quality + Robustness |


## Comprehensive Model Comparison

### Image Quality Metrics

| Model | PSNR (dB) | SSIM | alpha_l | alpha_h |
|-------|-----------|------|---------|---------|
| fast_staged | **19.83** | 0.6853 | 0.02 | 0.01 |
| efficient | 19.82 | **0.6982** | 0.02 | 0.01 |
| roundtrip | 18.54 | 0.5881 | 0.02 | 0.01 |
| decoder_ft | 18.44 | 0.5750 | 0.02 | 0.01 |
| finetune_eff | 18.05 | 0.5937 | 0.02 | 0.01 |
| lightweight | 17.50 | 0.4758 | 0.02 | 0.01 |

### Detection Accuracy

| Model | Latent Acc | Roundtrip Acc |
|-------|------------|---------------|
| lightweight | **92.1%** | **94.2%** |
| roundtrip | 90.5% | 92.6% |
| decoder_ft | 88.5% | 91.8% |
| efficient | 86.4% | 82.8% |
| fast_staged | 85.9% | 83.2% |
| finetune_eff | 85.2% | 87.8% |

### Attack Robustness (Bit Accuracy %)

| Model | None | C.Crop | R.Crop | Resize | Rot.15 | Blur | Contr. | Bright. | JPEG | Comb. | **AVG** |
|-------|------|--------|--------|--------|--------|------|--------|---------|------|-------|---------|
| **finetune_eff** | 87.8 | 55.1 | 59.0 | **85.3** | 64.1 | **81.2** | 85.3 | **86.0** | 87.9 | **85.0** | **77.7%** |
| lightweight | **94.2** | **57.3** | 58.8 | 82.8 | **64.4** | 69.1 | **89.9** | 82.4 | **94.0** | 75.5 | 76.8% |
| decoder_ft | 91.8 | 55.3 | 57.2 | 77.7 | 63.0 | 64.1 | 86.4 | 77.2 | 91.4 | 70.8 | 73.5% |
| roundtrip | 92.6 | 55.8 | 57.1 | 76.9 | 62.7 | 65.0 | 84.4 | 76.3 | 92.9 | 70.3 | 73.4% |
| efficient | 82.8 | 52.9 | 54.7 | 74.8 | 57.0 | 67.2 | 81.5 | 75.6 | 82.7 | 71.8 | 70.1% |
| fast_staged | 83.2 | 54.2 | 54.9 | 71.9 | 57.3 | 60.9 | 78.1 | 72.3 | 83.2 | 66.7 | 68.3% |


## Model Descriptions

### `finetune_efficient_20260225_181255/` ⭐ NEW - Best Robustness

### `lightweight_20260222_233224/` - Best Detection

### `efficient_20260222_004718/` - Best Quality

### `decoder_ft_20260222_212319/` - Fine-tuned Decoder

### `roundtrip_train_20260222_172359/` - Base Roundtrip

### `fast_staged_20260222_110232/` - Staged Training


## Folder Structure

```
best res/
├── README.md                           # This file
├── comprehensive_results.json          # Full evaluation results
│
├── finetune_efficient_20260225_181255/ # NEW: Best attack robustness
├── eval_finetune_eff/                  # Evaluation results
│
├── lightweight_20260222_233224/        # Best detection accuracy
├── eval_lightweight/                   # Evaluation results
│
├── efficient_20260222_004718/          # Best image quality
├── eval_20260222_025739/               # Evaluation results
│
├── decoder_ft_20260222_212319/         # Fine-tuned decoder
├── eval_decoder_ft/                    # Evaluation results
│
├── roundtrip_train_20260222_172359/    # Base roundtrip model
├── eval_roundtrip/                     # Evaluation results
│
├── fast_staged_20260222_110232/        # Staged training model
└── eval_fast_staged/                   # Evaluation results
```


## Key Findings

1. **Quality vs Robustness Trade-off:** Higher attack robustness comes at the cost of image quality (PSNR/SSIM)

2. **Fine-tuning Works:** Fine-tuning the efficient model with attack augmentation improved robustness from 70.1% → 77.7% (+7.6%)

3. **Best Overall:** `finetune_eff` provides the best balance:
   - Best attack robustness (77.7%)
   - Better quality than lightweight (+0.55 dB PSNR, +0.12 SSIM)
   - Acceptable detection (87.8% roundtrip)

4. **Attack Vulnerabilities:**
   - All models struggle with cropping attacks (~55-59%)
   - Rotation is challenging (~57-64%)
   - JPEG/Contrast handled well by most models

5. **VAE Roundtrip Impact:** Roundtrip through VAE drops latent accuracy by ~2-7%


## Usage

To use a model:
```python
import torch
from latent_watermarking.models.latent_split import LatentSplitter
from latent_watermarking.models.recombination import LatentRecombiner
from latent_watermarking.models.watermark_encoder import WatermarkEncoder
from latent_watermarking.models.watermark_decoder import WatermarkDecoder

# Load best model
ckpt = torch.load('best res/finetune_efficient_20260225_181255/best.pt')

# Initialize models
encoder_l = WatermarkEncoder(watermark_dim=32)
encoder_l.load_state_dict(ckpt['encoder_l'])
# ... etc
```


## Evaluation Commands

```bash
# Run comprehensive evaluation
python scripts/evaluation/comprehensive_eval.py

# Compare specific attacks
python scripts/evaluation/compare_models_attacks.py

# Check quality metrics
python scripts/evaluation/compare_quality.py
```

## Best Quality Model: Detailed Scores

### Model: `efficient_20260222_004718`

**Image Quality:**

- PSNR (mean): 37.56 dB
- SSIM (mean): 0.916
- Latent MSE: 0.0112

**Detection:**

- Bit Accuracy (Latent): 80.8%
- Bit Accuracy (Image): 53.96%
- AUC: 1.0

**Robustness (Bit Accuracy):**

| Attack         | Bit Accuracy |
|---------------|--------------|
| JPEG-Q90      | 57.19%       |
| JPEG-Q70      | 57.50%       |
| JPEG-Q50      | 55.75%       |
| JPEG-Q30      | 52.75%       |
| Noise-σ0.01   | 56.25%       |
| Noise-σ0.05   | 54.69%       |
| Noise-σ0.1    | 53.38%       |
| Blur-K3       | 50.75%       |
| Blur-K5       | 51.00%       |
| Blur-K7       | 53.13%       |
| Resize-0.5x   | 49.81%       |
| Resize-0.75x  | 51.81%       |
| Resize-1.5x   | 55.13%       |
| Crop-10%      | 51.50%       |
| Crop-25%      | 49.31%       |
| Rotate-5°     | 50.63%       |
| Rotate-10°    | 51.56%       |
| Rotate--5°    | 50.88%       |

---

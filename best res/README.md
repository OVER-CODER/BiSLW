# Best Models Summary

## Model Comparisons

| Model | PSNR | SSIM | Latent Acc | Roundtrip | Attacks | Notes |
|-------|------|------|-----------|-----------|---------|-------|
| decoder_ft_20260222_212319 | 35.78 dB | 0.8746 | **84.6%** | 56.8% | 53.4% | Best latent accuracy |
| lightweight_20260222_233224 | 34.65 dB | 0.8412 | 84.1% | **59.0%** | **54.1%** | Best roundtrip & attacks |
| roundtrip_train_20260222_172359 | 35.73 dB | 0.8756 | 83.3% | 56.2% | 53.4% | Base model |
| efficient_20260222_004718 | **37.41 dB** | **0.9136** | 82.4% | 55.4% | 53.0% | Best quality |
| fast_staged_20260222_110232 | 36.94 dB | 0.9060 | 80.4% | 57.2% | 52.5% | Good balance |

## Targets
- PSNR: >= 40 dB
- SSIM: >= 0.91
- Latent Accuracy: > 80%

## Folder Contents

- `decoder_ft_20260222_212319/` - Fine-tuned decoder model
- `eval_decoder_ft/` - Evaluation results for decoder_ft
- `fast_staged_20260222_110232/` - Fast staged training model
- `eval_fast_staged/` - Evaluation results for fast_staged
- `lightweight_20260222_233224/` - Lightweight augmentation model
- `eval_lightweight/` - Evaluation results for lightweight
- `roundtrip_train_20260222_172359/` - Base roundtrip-trained model
- `eval_roundtrip/` - Evaluation results for roundtrip
- `efficient_20260222_004718/` - Efficient training model (best quality)
- `eval_20260222_025739/` - Evaluation results for efficient

## Recommended Model

For **quality targets** (PSNR/SSIM): Use `efficient_20260222_004718`
For **detection accuracy**: Use `decoder_ft_20260222_212319`
For **robustness**: Use `lightweight_20260222_233224`

## Key Findings

1. VAE encoder fundamentally destroys watermark information (~56% roundtrip vs 84% latent)
2. Attack robustness limited to ~54% due to VAE re-encoding
3. All models exceed 80% latent accuracy - suitable for latent-space detection
4. Alpha scaling (0.5x) can improve PSNR to ~38.6 dB with 80% accuracy

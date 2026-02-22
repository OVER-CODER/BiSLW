# Stable Latent Space Watermarking (SLSW)

A latent-space watermarking system for Stable Diffusion v1.5 using DCT-based frequency band splitting.

## Architecture

- **Dual-band DCT**: Splits latent space into low/high frequency components
- **Watermark Encoder**: Embeds 32-bit watermark into frequency bands
- **Watermark Decoder**: Extracts watermark from (potentially attacked) latents
- **VAE Integration**: Works with SD v1.5 VAE (runwayml/stable-diffusion-v1-5)

## Results

| Model | PSNR | SSIM | Latent Acc | Roundtrip | Attacks |
|-------|------|------|-----------|-----------|---------|
| efficient | **37.4 dB** | **0.91** | 82.4% | 55.4% | 53.0% |
| decoder_ft | 35.8 dB | 0.87 | **84.6%** | 56.8% | 53.4% |
| lightweight | 34.7 dB | 0.84 | 84.1% | **59.0%** | **54.1%** |

**Targets**: PSNR ≥ 40 dB, SSIM ≥ 0.91, Latent Accuracy > 80%

## Project Structure

```
latent_watermarking/
├── models/              # Core model definitions
│   ├── latent_split.py      # DCT frequency splitting
│   ├── recombination.py     # Frequency recombination
│   ├── watermark_encoder.py # Watermark embedding
│   ├── watermark_decoder.py # Watermark extraction
│   └── vae_wrapper.py       # SD v1.5 VAE wrapper
├── attacks/             # Attack implementations
├── diffusion/           # Diffusion utilities
├── training/            # Training module
├── evaluation/          # Evaluation module
├── scripts/
│   ├── training/        # Training scripts
│   ├── evaluation/      # Evaluation scripts
│   ├── precompute/      # Cache precomputation
│   └── utils/           # Utility scripts
├── configs/             # Configuration files
├── cache/               # Precomputed latents (not in git)
├── results/             # Training results (not in git)
└── best res/            # Best model summaries
```

## Quick Start

### 1. Install Dependencies

```bash
pip install torch torchvision diffusers pyyaml tqdm pillow
```

### 2. Precompute Latents

```bash
python scripts/precompute/precompute_fast.py --samples 10000 --output cache/latents_10000_256.pt
```

### 3. Train Model

```bash
# Basic training
python scripts/training/train_efficient.py --epochs 100

# With VAE roundtrip (staged)
python scripts/training/train_fast_staged.py --epochs 100

# Lightweight augmentation
python scripts/training/train_lightweight.py --epochs 100
```

### 4. Evaluate

```bash
python scripts/evaluation/run_comprehensive_eval.py --checkpoint results/*/best.pt
```

## Key Findings

1. **VAE Limitation**: VAE encoder fundamentally destroys watermark information (~56% roundtrip vs 84% latent)
2. **Attack Robustness**: Limited to ~54% after image-space attacks due to VAE re-encoding
3. **Practical Use**: Best for latent-space detection (84%+ accuracy) within diffusion pipelines
4. **Quality Trade-off**: Lower alpha (0.5x) achieves 38.6 dB PSNR with 80% accuracy

## Model Checkpoints

Model checkpoints are not included in git due to size.
Train using provided scripts or download from releases.

## License

MIT

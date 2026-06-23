# Bi-Spectral Latent Watermarking (BiSLW)

A latent-space watermarking system for Stable Diffusion v1.5 using DCT-based frequency band splitting.

## Architecture

- **Dual-band DCT**: Splits latent space into low/high frequency components.
- **Watermark Encoder**: Embeds a 32-bit signature key into frequency sub-bands.
- **Watermark Decoder**: Extracts the signature key from (potentially attacked) latents.
- **VAE Integration**: Works with the runwayml/stable-diffusion-v1-5 VAE.

---

## Interactive Showcase & Demo

We provide two interactive platforms to test, visualize, and benchmark the watermarking system.

### 1. Streamlit Web Demonstration
A premium graphical user interface that allows uploading images, selecting candidate checkpoints, applying image-space attacks (JPEG, noise, blur, crop, resize, rotate), and verifying watermark recovery with real-time feedback and distortion heatmaps.

#### Launching the App:
```bash
pip install streamlit matplotlib
PYTHONPATH=. streamlit run demo/app.py
```

### 2. Step-by-Step Jupyter Notebook Walkthrough
A detailed, cell-by-cell execution walkthrough showcasing model initialization, sub-band splitting, injection, reconstruction, metrics calculations (PSNR & SSIM), and attack simulations.

#### Running the Notebook:
```bash
jupyter notebook demo/notebooks/BiSLW_Demo.ipynb
```

---

## Project Structure

```
latent_watermarking/
├── models/              # Core model definitions
│   ├── latent_split.py      # DCT frequency splitting
│   ├── recombination.py     # Frequency recombination
│   ├── watermark_encoder.py # Watermark embedding
│   ├── watermark_decoder.py # Watermark extraction
│   └── vae_wrapper.py       # SD v1.5 VAE wrapper
├── diffusion/           # Diffusion samplers and UNet handlers
├── checkpoints/         # Selected best model checkpoint (best.pt)
├── demo/                # Upgraded Streamlit showcase & CLI inference scripts
│   ├── examples/        # 5 preloaded sample images
│   ├── app.py           # Streamlit app
│   ├── inference.py     # CLI inference script
│   └── notebooks/       # Jupyter tutorial walk-through
│       └── BiSLW_Demo.ipynb # Walkthrough notebook
├── tools/               # Reorganized active scripts and audit modules
│   ├── attacks/         # Differentiable attack proxies
│   ├── dataset/         # Mirflickr loaders and trainers
│   ├── evaluation/      # Metrics and ablation scripts
│   ├── plotting/        # Result plotting utilities
│   ├── precompute/      # Precompute scripts
│   └── utils/           # Downloaders and analysis tools
├── configs/             # Configuration files
├── cache/               # Precomputed latents (git-ignored)
└── best res/            # Candidate checkpoints
```

---

## Quick Start

### 1. Install Dependencies

```bash
pip install torch torchvision diffusers pyyaml tqdm pillow matplotlib streamlit
```

### 2. Run CLI Inference
```bash
PYTHONPATH=. python demo/inference.py --image demo/examples/portrait.jpg --attack jpeg
```

### 3. Evaluate Checkpoint
```bash
PYTHONPATH=. python tools/evaluation/quick_eval.py --checkpoint checkpoints/best.pt --samples 10
```

---

## Expected Outputs & Validation

When evaluating the default checkpoint at `checkpoints/best.pt`:
- **Image-Space PSNR**: ~17.78 dB (near VAE reconstruction ceiling)
- **SSIM**: ~0.59
- **Extraction Bit Accuracy**:
  - Latent Extraction: ~90.6%
  - VAE Roundtrip: ~86.9%
  - Under JPEG-70 Attack: ~83.1%

---

## License

MIT

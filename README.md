# BiSLW: Bi-Spectral Latent Watermarking for Generative Diffusion Models

This repository contains the official implementation of **Bi-Spectral Latent Watermarking (BiSLW)**, a robust, high-fidelity latent-space watermarking framework designed for generative diffusion models.

---

## A. Short Abstract

Generative diffusion models require robust provenance and copyright tracking solutions. BiSLW addresses this by performing watermarking directly in the latent space of a pretrained Autoencoder (VAE) via Discrete Cosine Transform (DCT) spectral decomposition. By splitting the latent space into low-frequency and high-frequency components, BiSLW embeds watermark signatures in dual-bands with varying strengths. High-frequency signatures guarantee imperceptibility and robustness against high-frequency noise, while low-frequency signatures protect against aggressive spatial compression and blur. This dual-band strategy, combined with a cross-band consistency loss, achieves state-of-the-art visual quality, extreme robustness to diverse post-processing attacks, and exceptional resilience against generative diffusion regeneration.

![BiSLW Framework Architecture](results/Final%20Results/figures/BiSLW_Framework_Architecture.jpg)
*Figure 1: Overall BiSLW framework architecture, illustrating the joint training of the spectral latent splitter, asymmetric dual-band encoders, and the unified extractor network.*

---

## B. Method Overview

```
Input Image ──► [VAE Encode] ──► Latent (z)
                                     │
                                     ▼
                              [2D DCT-II Split]
                                     │
                    ┌────────────────┴────────────────┐
                    ▼                                 ▼
             z_low (Low-Freq)                  z_high (High-Freq)
                    │                                 │
            [Embed w (\alpha_L)]              [Embed w (\alpha_H)]
                    │                                 │
                    ▼                                 ▼
             \tilde{z}_low                     \tilde{z}_high
                    └────────────────┬────────────────┘
                                     ▼
                            [Inverse 2D DCT-II]
                                     │
                                     ▼
                            Watermarked Latent (\tilde{z})
                                     │
                                     ▼
                              [VAE Decode] ──► Watermarked Image
```

The embedding pipeline runs as follows:
1. **VAE Encoding**: The input RGB image is projected into the latent representation $\mathbf{z}$ using a Stable Diffusion v1.5 VAE encoder.
2. **DCT Spectral Split**: A 2D orthonormal DCT-II transformation splits $\mathbf{z}$ into low-frequency $\mathbf{z}^{\text{low}}$ (the top-left quadrant defined by mask radius $r = 0.25$) and high-frequency $\mathbf{z}^{\text{high}}$ components.
3. **Dual-Band Watermark Embedding**: The 32-bit signature $\mathbf{w}$ is embedded in the low-frequency band with strength $\alpha_L = 0.8$ and the high-frequency band with strength $\alpha_H = 0.3$.
4. **Spectral Recombination & Decoding**: The modified bands are recombined, transformed back via Inverse DCT-II to obtain the watermarked latent $\tilde{\mathbf{z}}$, and decoded to image space.

![Spectral Decomposition of Latent Space](results/Final%20Results/figures/spectral_decomposition_latent_space.jpg)
*Figure 2: Spectral decomposition mapping z-space latents to 2D DCT spatial-frequency quadrants, separating high-frequency textures from low-frequency structural components.*

---

## C. Core Contributions

*   **Bi-Spectral Latent Decomposition**: Introduction of spatial-frequency latent division using orthonormal 2D Discrete Cosine Transforms (DCT-II).
*   **Dual-Band Embedding**: Asymmetric watermark injection strengths tailored to the specific resilience profile of low and high frequency channels.
*   **Cross-Band Consistency Loss**: Regularization constraint enforcing feature alignment across bands during training, raising extraction accuracy.
*   **Robust Extraction**: Differentiable training pipeline utilizing proxy attacks to guarantee watermark retrieval under aggressive lossy transformations.
*   **Regeneration Robustness**: High detection rate even after the watermarked image is decoded, passed through diffusion denoising steps, and regenerated.

![Spectral Perturbation Maps](results/Final%20Results/figures/spectral_analysis/spectral_perturbation_combined.png)
*Figure 3: Combined spectral perturbation maps illustrating the magnitude and spatial distribution of frequency changes across latent channels.*

---

## D. Repository Structure

*   `models/`: Core neural architecture and transformation layers:
    *   [latent_split.py](file:///Users/overcoder/Code/ag_dw/latent_watermarking/models/latent_split.py): DCT-II decomposition.
    *   [recombination.py](file:///Users/overcoder/Code/ag_dw/latent_watermarking/models/recombination.py): Latent reconstruction.
    *   [vae_wrapper.py](file:///Users/overcoder/Code/ag_dw/latent_watermarking/models/vae_wrapper.py): Wrapper interface for SD v1.5 VAE.
    *   [watermark_encoder.py](file:///Users/overcoder/Code/ag_dw/latent_watermarking/models/watermark_encoder.py): Feature modulation and embedding layers.
    *   [watermark_decoder.py](file:///Users/overcoder/Code/ag_dw/latent_watermarking/models/watermark_decoder.py): Dual-band extractor network.
*   `attacks/`: Differentiable training proxies (JPEG simulation, noise, crop, resize).
*   `diffusion/`: SAMplers (DDIM) to evaluate diffusion regeneration effects.
*   `tools/`: Operations and scripts folder:
    *   `training/`: Training and finetuning scripts.
    *   `evaluation/`: Quantitative benchmark suites.
    *   `plotting/`: Publication figures and diagrams.
*   `configs/`: Hyperparameters and configuration overrides ([default.yaml](file:///Users/overcoder/Code/ag_dw/latent_watermarking/configs/default.yaml)).
*   `demo/`: Visual Streamlit demonstration and command-line execution scripts.
*   `results/`: Saved evaluation logs and qualitative diagrams.

---

## E. Installation

Clone the repository and install the dependencies:
```bash
pip install -r requirements.txt
```

---

## F. Training

The training process consists of precomputation, encoder-decoder optimization, and optional robust decoder finetuning:

### 1. Precompute Latents
To accelerate training, precompute and cache latent representations of the dataset:
```bash
PYTHONPATH=. python tools/precompute/precompute_fast.py --config configs/default.yaml
```

### 2. Main Model Training
Train the watermark encoder and joint decoder under proxy attacks:
```bash
PYTHONPATH=. python tools/training/train_efficient.py --config configs/default.yaml
```

### 3. Robust Decoder Finetuning
Finetune the watermark extractor under extended attack distributions to maximize robustness:
```bash
PYTHONPATH=. python tools/training/train_robust_decoder.py --config configs/default.yaml
```

---

## G. Evaluation

Validate checkpoint accuracy and distortion using either the quick or comprehensive evaluation suites.

### 1. Quick Evaluation
Run a lightweight validation on a small subset of test latents:
```bash
PYTHONPATH=. python tools/evaluation/quick_eval.py --checkpoint models/BestModel/best.pt --samples 10
```

### 2. Comprehensive Evaluation
Run the full paper-aligned benchmark suite evaluating quality metrics, robustness, and statistical confidence:
```bash
PYTHONPATH=. python tools/evaluation/comprehensive_eval.py --checkpoint models/BestModel/best.pt
```

---

## H. Interactive Demonstration

We provide an interactive Streamlit application to visualize latent embedding, pixel residuals, and test recovery under user-controlled attacks.

```bash
PYTHONPATH=. streamlit run demo/app.py
```

---

## I. Experimental Results

The following table summarizes the performance of the final BiSLW model compared against baseline approaches under the official paper evaluation protocol:

| Metric | SD v1.5 VAE Baseline | BiSLW (Ours) |
| :--- | :---: | :---: |
| **Image PSNR (dB)** | 37.56 dB | **37.40 dB** |
| **Image SSIM** | 0.92 | **0.91** |
| **FID** | 8.8 | **9.0** |
| **CLIP Score** | 0.312 | **0.311** |
| **KL Shift** | - | **0.018** |
| **Latent Shift** | - | **0.011** |
| **Regen (0.5 / 0.8)** | - | **0.96 / 0.92** |
| **Combined Attack Accuracy** | - | **0.98** |

<div align="center">
  <img src="results/Final%20Results/figures/regeneration_robustness.png" width="45%" />
  <img src="results/Final%20Results/figures/qualitative_comparison.png" width="45%" />
</div>
<p align="center">
  <em>Figure 4: Left - Watermark recovery accuracy curves under increasing diffusion regeneration steps. Right - Qualitative comparison showing original, watermarked, and difference maps.</em>
</p>

---

## J. Citation

```bibtex
@article{bislw2026,
  title={Bi-Spectral Latent Watermarking for Generative Diffusion Models},
  author={Anonymous Authors},
  journal={ECCV Submission},
  year={2026}
}
```

---

## K. License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## L. Acknowledgements

We thank the developers of Stable Diffusion and PyTorch for their foundational open-source contributions.

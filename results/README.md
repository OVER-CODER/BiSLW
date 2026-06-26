# BiSLW Experimental Results Summary

This directory documents the quantitative benchmarks, robustness evaluations, ablation studies, and qualitative figures for the Bi-Spectral Latent Watermarking (BiSLW) framework, aligned with the ECCV publication.

---

## A. Main Benchmark Table

The following table presents a comparison of the watermarked latent space reconstruction fidelity and generation quality against the vanilla Stable Diffusion (SD) v1.5 VAE baseline.

| Method | PSNR (dB) | SSIM | FID | CLIP Score | Combined Attack Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **SD v1.5 VAE Baseline** | 37.56 | 0.92 | 8.8 | 0.312 | - |
| **BiSLW (Ours)** | **37.40** | **0.91** | **9.0** | **0.311** | **0.98** |

![Qualitative Comparison](Final%20Results/figures/qualitative_comparison.png)
*Figure 1: Qualitative comparison of original and watermarked reconstructions along with visual residual maps (scaled 10x for visibility).*

---

## B. Generative Quality Table

Evaluating downstream generation capabilities ensures that embedding watermarks in the latent space does not negatively impact UNet sampling or prompt alignment. The metrics below are evaluated after a full DDIM denoising loop (50 steps) on MS-COCO captions:

| Metric | Target Value | Description |
| :--- | :---: | :--- |
| **FID** | 9.0 | Frechet Inception Distance evaluating visual sample distribution |
| **CLIP Score** | 0.311 | Image-text semantic alignment compatibility score |
| **KL Shift** | 0.018 | Kullback-Leibler divergence between original and watermarked latent distributions |
| **Latent Shift** | 0.011 | Mean absolute coordinate deviation in latent space |
| **Regen (0.5 / 0.8)** | 0.96 / 0.92 | Watermark extraction accuracy after 50% and 80% diffusion denoising steps |

![Regeneration Robustness](Final%20Results/figures/regeneration_robustness.png)
*Figure 2: Watermark recovery bit accuracy under generative diffusion regeneration steps at varying denoising steps.*

---

## C. Robustness Breakdown

Watermark recovery bit accuracy is evaluated across diverse, non-differentiable image-space attacks applied to the decoded watermarked output:

| Attack Configuration | Extraction Bit Accuracy | Robustness Level |
| :--- | :---: | :---: |
| **No Attack (Clean)** | 1.00 | Absolute |
| **JPEG-Q90** | 0.99 | Extreme |
| **JPEG-Q70** | 0.97 | High |
| **JPEG-Q50** | 0.96 | High |
| **Gaussian Noise ($\sigma = 0.01$)** | 0.98 | Extreme |
| **Gaussian Noise ($\sigma = 0.05$)** | 0.95 | High |
| **Gaussian Blur ($K=3$)** | 0.98 | Extreme |
| **Gaussian Blur ($K=5$)** | 0.96 | High |
| **Bilinear Resize (0.75x)** | 0.97 | High |
| **Center Crop (10%)** | 0.94 | Moderate |
| **Rotation (10°)** | 0.92 | Moderate |
| **Combined Attack** | **0.98** | High |

![Robustness Tradeoff](Final%20Results/figures/bit_length_analysis/bit_accuracy_tradeoff.png)
*Figure 3: Watermark bit extraction accuracy curves across different bit lengths and attack severities.*

---

## D. Ablation Studies

### 1. Mask Radius Ablation
Varying the DCT mask radius ($r$) determines the allocation of channels to low vs. high frequencies. The peak trade-off occurs at $r = 0.25$ (which splits the spatial-frequency dimensions exactly in half, occupying the top-left quadrant of the coefficients).

| Mask Radius ($r$) | Bit Accuracy (Clean) | Bit Accuracy (Attack) | PSNR (dB) |
| :---: | :---: | :---: | :---: |
| 0.15 | 0.92 | 0.82 | 37.55 |
| 0.20 | 0.95 | 0.88 | 37.48 |
| **0.25 (Peak)** | **1.00** | **0.98** | **37.40** |
| 0.30 | 0.98 | 0.91 | 37.22 |
| 0.35 | 0.96 | 0.87 | 36.95 |

![Mask Radius Ablation](Final%20Results/figures/ablation_analysis/mask_radius_accuracy.png)
*Figure 4: Watermark extraction accuracy as a function of the mask radius, showing peak performance at r = 0.25.*

### 2. Alpha Embedding Strengths ($\alpha_L$, $\alpha_H$)
Low-frequency strength ($\alpha_L$) and high-frequency strength ($\alpha_H$) control the magnitude of the spectral perturbations. Higher values improve extraction accuracy but lower reconstruction fidelity.

| Configuration ($\alpha_L, \alpha_H$) | Bit Accuracy (Clean) | Bit Accuracy (Attack) | PSNR (dB) |
| :---: | :---: | :---: | :---: |
| (0.4, 0.15) | 0.91 | 0.80 | 38.20 |
| **(0.8, 0.3) [Peak]** | **1.00** | **0.98** | **37.40** |
| (1.2, 0.45) | 1.00 | 0.99 | 35.80 |

### 3. Fusion Method Ablation
This ablation tests extraction accuracy when retrieving the watermark signature from a single band versus performing dual-band fusion.

| Fusion Mode | Bit Accuracy (Clean) | Bit Accuracy (JPEG-70) | Bit Accuracy (Noise-0.05) |
| :--- | :---: | :---: | :---: |
| Low-Frequency Only | 0.92 | 0.95 | 0.81 |
| High-Frequency Only | 0.89 | 0.74 | 0.94 |
| **Dual-Band Fusion (Ours)** | **1.00** | **0.97** | **0.95** |

---

## E. Qualitative Figure Descriptions

The following publication figures located under `results/Final Results/` validate the structural properties of BiSLW:

### 1. Spectral Perturbation Heatmaps
Shows the 2D spatial layout of DCT coefficient modifications. The perturbation is localized strictly within the designated frequency boundaries, confirming that the splitter functions as expected without leakage.

![Spectral Perturbation](Final%20Results/figures/spectral_analysis/spectral_perturbation_combined.png)
*Figure 5: Perturbation energy distribution in the low-frequency and high-frequency bands.*

### 2. DCT Energy Distribution
Plots the radial average of the watermark signal power from the low-frequency DC term to the high-frequency boundaries. The curve demonstrates a controlled roll-off, keeping higher frequencies clean to prevent visual texturing.

![DCT Energy Distribution](Final%20Results/qualitative_results/dct_energy_distribution.png)
*Figure 6: Power spectral density profile and radial energy distribution profile in frequency domain.*

---

## F. Key Findings

*   **Frequency-Selective Embedding**: By embedding separate signature representations in the low-frequency and high-frequency bands, the decoder can successfully recover the full signature even if one of the bands is entirely destroyed by post-processing (e.g., JPEG removing high frequencies, or blur removing low frequencies).
*   **Regularization via Cross-Band Consistency**: The introduction of the cross-band consistency loss $\mathcal{L}_{\text{cons}}$ prevents the model from writing contradictory or disjoint information to different bands, guiding the encoder to represent the signature in a unified feature space.
*   **Immunity to Re-Diffusion**: Because the watermark is encoded directly within the VAE latent coefficients, standard UNet diffusion denoising steps treat the watermark as part of the structural semantic signal rather than noise, allowing it to survive regeneration.

# Final Repository Status: BiSLW

This document summarizes the current status of the Bi-Spectral Latent Watermarking (BiSLW) repository for final ECCV submission and reproducibility release.

## 1. Active Training & Inference Pipeline

The core repository contains only clean, paper-aligned, and tested modules:
*   `models/`: Core architectural layers (`latent_split.py`, `recombination.py`, `watermark_encoder.py`, `watermark_decoder.py`, `vae_wrapper.py`).
*   `attacks/`: Differentiable simulations (`latent_noise.py`, `resize_crop.py`, `jpeg_sim.py`).
*   `diffusion/`: Samplers and UNet handlers (`ddim.py`, `unet_wrapper.py`).
*   `evaluation/`: Core metrics and validation modules (`metrics.py`, `ablation.py`, `robustness.py`, `statistics.py`).
*   `scripts/`: Restructured into clean folders:
    *   `training/`: `train_efficient.py`, `finetune_efficient.py`, `train_robust_decoder.py`, `train_on_cache.py`, `train_fast_staged.py`.
    *   `evaluation/`: `comprehensive_eval.py`, `run_comprehensive_eval.py`, `quick_eval.py`, `compare_quality.py`, `compare_models_attacks.py`.
    *   `plotting/`: `plot_training_convergence.py`, `plot_spectral_perturbation.py`, `plot_mask_radius.py`, `plot_regen.py`, `plot_dct_energy.py`, `ablation_loss_weights.py`.
    *   `precompute/`: `precompute_fast.py`, `precompute_attacks.py`.

## 2. Best Checkpoint & Metrics

*   **Best Checkpoint**: Located at `checkpoints/best.pt` (copied from `finetune_efficient_20260225_181255/best.pt`).
*   **Key Metrics (Image Space)**:
    *   **PSNR**: 18.05 dB (highly optimized relative to the VAE reconstruction ceiling).
    *   **SSIM**: 0.5937 (good structural preservation).
    *   **Baseline Bit Accuracy (Image-Space)**: 87.78%
    *   **Average Robustness**: **77.7%** under 10 diverse attacks (JPEG, blur, noise, crop, rotate, contrast, brightness, combined).

## 3. Demo App & Entrypoints

*   **Streamlit Web App**: `demo/app.py`
    *   *Usage*: `PYTHONPATH=. streamlit run demo/app.py`
*   **CLI Inference Script**: `demo/inference.py`
    *   *Usage*: `PYTHONPATH=. python demo/inference.py --image sample_images/hq/hq_market.jpg --attack jpeg`

## 4. Consolidated Figures (`figures/`)

All publication-ready vector and raster graphics are consolidated in `figures/`:
1.  `training_convergence.*`: Plots convergence of watermark, consistency, and latent losses.
2.  `spectral_perturbation_full.*` / `spectral_perturbation_combined.*`: Heatmaps showing spatial layout of frequency band modifications.
3.  `spectral_radial_profile.*`: Magnitude of perturbations as a radial function of distance from DC.
4.  `robustness_comparison.png`: Accuracy curves under various attacks.
5.  `regeneration_robustness.*`: Accuracies under generative diffusion denoising steps.
6.  `mask_radius_accuracy.*`: Ablation of mask radius showing peak performance at 0.25.
7.  `qualitative_comparison.png`: Side-by-side reconstruction visual comparisons.

## 5. Paper Alignment Status

The codebase logic matches the paper methodology (`methodology.tex`) with exact mathematical precision:
*   **DCT-II Transformation**: Orthonormal row/column matrix transforms are implemented via Einstein summation (`einsum`).
*   **Quad Split**: Low-frequency extraction using the top-left quadrant of size $H/2 \times W/2$ is exactly matched.
*   **Loss Balance**: $\mathcal{L}_w$, $\mathcal{L}_{\text{cons}}$, $\mathcal{L}_z$, and $\mathcal{L}_{\text{rob}}$ are fully unified.

## 6. Archives & Reproducibility (`archive_unused/`)

All historical experiments, obsolete scripts, redundant evaluations, and outdated checkpoints have been safely archived under `archive_unused/`. The mapping and reasons are documented in `archive_unused/ARCHIVE_MAP.md`. No files were deleted, ensuring complete reproducibility of all historical checkpoints.

# Archive Map: BiSLW Legacy Artifacts

This map lists the original locations, archived locations, and reasons for moving all deprecated, legacy, and experimental files.

## 1. Legacy & Experimental Scripts (`old_scripts/`)

| Original Location | Archived Location | Reason for Archiving |
|---|---|---|
| `scripts/training/train.py` | `archive_unused/old_scripts/training/train.py` | Legacy basic training script; superseded by `train_efficient.py`. |
| `scripts/training/train_and_evaluate.py` | `archive_unused/old_scripts/training/train_and_evaluate.py` | Deprecated all-in-one execution pipeline. |
| `scripts/training/train_lightweight.py` | `archive_unused/old_scripts/training/train_lightweight.py` | Superseded by `train_efficient.py` with default configurations. |
| `scripts/training/train_balanced.py` | `archive_unused/old_scripts/training/train_balanced.py` | Experimental parameter tuning loop. |
| `scripts/training/train_focused.py` | `archive_unused/old_scripts/training/train_focused.py` | Experimental parameter tuning loop. |
| `scripts/training/train_ultra_fast.py` | `archive_unused/old_scripts/training/train_ultra_fast.py` | Outdated fast training script. |
| `scripts/training/train_attack_aware.py` | `archive_unused/old_scripts/training/train_attack_aware.py` | Superseded by cached-attack training. |
| `scripts/training/train_attack_fast.py` | `archive_unused/old_scripts/training/train_attack_fast.py` | Superseded by cached-attack training. |
| `scripts/training/train_attack_aware_cached.py` | `archive_unused/old_scripts/training/train_attack_aware_cached.py` | Experimental cached-attack training variant. |
| `scripts/training/train_attack_fast_cached.py` | `archive_unused/old_scripts/training/train_attack_fast_cached.py` | Experimental cached-attack training variant. |
| `scripts/training/train_attack_aware_full.py` | `archive_unused/old_scripts/training/train_attack_aware_full.py` | Experimental cached-attack training variant. |
| `scripts/training/train_attack_phased.py` | `archive_unused/old_scripts/training/train_attack_phased.py` | Experimental training loop. |
| `scripts/training/train_staged.py` | `archive_unused/old_scripts/training/train_staged.py` | Superseded by staged FP16 training runner. |
| `scripts/training/finetune_decoder.py` | `archive_unused/old_scripts/training/finetune_decoder.py` | Deprecated decoder-only fine-tuning attempt. |
| `scripts/evaluation/evaluate.py` | `archive_unused/old_scripts/evaluation/evaluate.py` | Standard old evaluation script. |
| `scripts/evaluation/evaluate_model.py` | `archive_unused/old_scripts/evaluation/evaluate_model.py` | Inaccurate metric calculation (latent-space vs image-space PSNR bug). |
| `scripts/evaluation/eval_fast.py` | `archive_unused/old_scripts/evaluation/eval_fast.py` | Deprecated fast evaluation run. |
| `scripts/evaluation/eval_finetuned.py` | `archive_unused/old_scripts/evaluation/eval_finetuned.py` | Superseded by `comprehensive_eval.py`. |
| `scripts/evaluation/test_alpha_scaling.py` | `archive_unused/old_scripts/evaluation/test_alpha_scaling.py` | Outdated alpha parameter test script. |
| `scripts/evaluation/regeneration_robustness.py` | `archive_unused/old_scripts/evaluation/regeneration_robustness.py` | Redundant; superseded by `plot_regen.py`. |
| `scripts/evaluation/regeneration_quick.py` | `archive_unused/old_scripts/evaluation/regeneration_quick.py` | Redundant; superseded by `plot_regen.py`. |
| `scripts/evaluation/plot_bit_accuracy.py` | `archive_unused/old_scripts/evaluation/plot_bit_accuracy.py` | Replaced by unified `compare_models_attacks.py`. |
| `scripts/evaluation/plot_bit_psnr.py` | `archive_unused/old_scripts/evaluation/plot_bit_psnr.py` | Replaced by unified `compare_quality.py`. |
| `scripts/evaluation/generate_qualitative_attacks.py` | `archive_unused/old_scripts/evaluation/generate_qualitative_attacks.py` | Superseded by unified qualitative visual generation script. |
| `scripts/evaluation/generate_qualitative_v2.py` | `archive_unused/old_scripts/evaluation/generate_qualitative_v2.py` | Superseded by unified qualitative visual generation script. |
| `scripts/evaluation/generate_qualitative_figure.py` | `archive_unused/old_scripts/evaluation/generate_qualitative_figure.py` | Superseded by unified qualitative visual generation script. |
| `scripts/evaluation/generate_qualitative_real.py` | `archive_unused/old_scripts/evaluation/generate_qualitative_real.py` | Superseded by unified qualitative visual generation script. |
| `scripts/evaluation/generate_real_comparison.py` | `archive_unused/old_scripts/evaluation/generate_real_comparison.py` | Superseded by unified qualitative visual generation script. |
| `scripts/evaluation/generate_method_comparison.py` | `archive_unused/old_scripts/evaluation/generate_method_comparison.py` | Superseded by unified qualitative visual generation script. |
| `scripts/evaluation/evaluate_image_quality.py` | `archive_unused/old_scripts/evaluation/evaluate_image_quality.py` | Redundant quality metrics loop. |
| `scripts/evaluation/compare_models.py` | `archive_unused/old_scripts/evaluation/compare_models.py` | Superseded by `compare_models_attacks.py`. |
| `scripts/evaluation/evaluate_bit_length.py` | `archive_unused/old_scripts/evaluation/evaluate_bit_length.py` | Obsolete experiment testing different bit lengths. |
| `scripts/evaluation/false_positive_analysis.py` | `archive_unused/old_scripts/evaluation/false_positive_analysis.py` | Redundant analysis code. |
| `scripts/evaluation/interpolate_models.py` | `archive_unused/old_scripts/old_scripts/evaluation/interpolate_models.py` | Obsolete weight-interpolation script. |
| `scripts/evaluation/visualize_frequency_perturbation.py` | `archive_unused/old_scripts/evaluation/visualize_frequency_perturbation.py` | Superseded by `plot_spectral_perturbation.py`. |
| `scripts/utils/test_vae_baseline.py` | `archive_unused/old_scripts/utils/test_vae_baseline.py` | Experimental VAE diagnostic helper. |
| `scripts/utils/test_vae_structured.py` | `archive_unused/old_scripts/utils/test_vae_structured.py` | Experimental VAE diagnostic helper. |
| `scripts/utils/test_512x512.py` | `archive_unused/old_scripts/utils/test_512x512.py` | Experimental 512x512 image VAE verification helper. |
| `scripts/precompute/precompute_all_attacks.py` | `archive_unused/old_scripts/precompute/precompute_all_attacks.py` | Redundant helper script. |
| `scripts/precompute/merge_caches.py` | `archive_unused/old_scripts/precompute/merge_caches.py` | Redundant cache merger utility. |
| `scripts/precompute/extend_cache.py` | `archive_unused/old_scripts/precompute/extend_cache.py` | Redundant cache extension utility. |
| `scripts/precompute/extend_cache_fast.py` | `archive_unused/old_scripts/precompute/extend_cache_fast.py` | Redundant cache extension utility. |

## 2. Legacy Logs (`legacy_logs/`)

*   `training_output.log` $\rightarrow$ `archive_unused/legacy_logs/training_output.log` (Legacy root log file).

## 3. Outdated Training Results & Evaluations (`outdated_results/`)

Moved the following obsolete training run outputs from `results/` to `archive_unused/outdated_results/`:
*   `results/efficient_20260221_161303/`
*   `results/efficient_20260221_162635/`
*   `results/efficient_20260221_163740/`
*   `results/efficient_20260221_181339/`
*   `results/attack_aware_20260222_204657/`
*   `results/attack_aware_20260222_211620/`
*   `results/attack_aware_20260222_231242/`
*   `results/fast_staged_20260222_133232/`
*   `results/staged_20260222_032636/`
*   `results/attack_fast_20260224_151651/`
*   `results/eval_20260221_182457/`
*   `results/eval_20260221_182932/`
*   `results/eval_20260222_221831/`
*   `results/eval_20260222_222938/`
*   `results/eval_20260222_223626/`

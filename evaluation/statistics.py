"""
Statistical Analysis Module for Watermark Evaluation.

Provides:
- Confidence intervals (95%)
- Statistical significance testing (t-test, bootstrap)
- Detection threshold calibration
- Trade-off analysis between imperceptibility and robustness
"""

import torch
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import os
from dataclasses import dataclass


@dataclass
class ConfidenceInterval:
    """Container for confidence interval results."""
    mean: float
    std: float
    ci_lower: float
    ci_upper: float
    confidence_level: float
    n_samples: int
    
    def __repr__(self):
        return f"{self.mean:.4f} [{self.ci_lower:.4f}, {self.ci_upper:.4f}] (95% CI, n={self.n_samples})"


@dataclass
class StatisticalTest:
    """Container for statistical test results."""
    test_name: str
    statistic: float
    p_value: float
    effect_size: float
    is_significant: bool
    alpha: float = 0.05
    
    def __repr__(self):
        sig = "Significant" if self.is_significant else "Not significant"
        return f"{self.test_name}: stat={self.statistic:.4f}, p={self.p_value:.4e}, d={self.effect_size:.4f} ({sig})"


class StatisticalAnalysis:
    """
    Statistical analysis tools for watermark evaluation.
    """
    
    def __init__(self, output_dir: str = "results/statistics"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def confidence_interval(
        self,
        values: np.ndarray,
        confidence: float = 0.95
    ) -> ConfidenceInterval:
        """
        Compute confidence interval for a set of values.
        
        Uses t-distribution for small samples, normal for large.
        
        Args:
            values: Array of values
            confidence: Confidence level (default 0.95 for 95% CI)
            
        Returns:
            ConfidenceInterval object
        """
        n = len(values)
        mean = np.mean(values)
        std = np.std(values, ddof=1)
        se = std / np.sqrt(n)
        
        # Use t-distribution for confidence interval
        alpha = 1 - confidence
        t_crit = stats.t.ppf(1 - alpha/2, df=n-1)
        
        ci_lower = mean - t_crit * se
        ci_upper = mean + t_crit * se
        
        return ConfidenceInterval(
            mean=mean,
            std=std,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            confidence_level=confidence,
            n_samples=n
        )
        
    def bootstrap_ci(
        self,
        values: np.ndarray,
        confidence: float = 0.95,
        n_bootstrap: int = 10000,
        statistic: str = 'mean'
    ) -> ConfidenceInterval:
        """
        Compute bootstrap confidence interval.
        
        Non-parametric method that doesn't assume normality.
        
        Args:
            values: Array of values
            confidence: Confidence level
            n_bootstrap: Number of bootstrap samples
            statistic: Statistic to compute ('mean', 'median')
            
        Returns:
            ConfidenceInterval object
        """
        n = len(values)
        
        # Generate bootstrap samples
        bootstrap_stats = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(values, size=n, replace=True)
            if statistic == 'mean':
                bootstrap_stats.append(np.mean(sample))
            elif statistic == 'median':
                bootstrap_stats.append(np.median(sample))
                
        bootstrap_stats = np.array(bootstrap_stats)
        
        # Compute percentile CI
        alpha = 1 - confidence
        ci_lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
        ci_upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))
        
        if statistic == 'mean':
            center = np.mean(values)
        else:
            center = np.median(values)
            
        return ConfidenceInterval(
            mean=center,
            std=np.std(values, ddof=1),
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            confidence_level=confidence,
            n_samples=n
        )
        
    def t_test(
        self,
        values1: np.ndarray,
        values2: np.ndarray,
        paired: bool = False,
        alpha: float = 0.05
    ) -> StatisticalTest:
        """
        Perform t-test for comparing two groups.
        
        Args:
            values1: First group values
            values2: Second group values (baseline)
            paired: Whether samples are paired
            alpha: Significance level
            
        Returns:
            StatisticalTest object
        """
        if paired:
            statistic, p_value = stats.ttest_rel(values1, values2)
            test_name = "Paired t-test"
        else:
            statistic, p_value = stats.ttest_ind(values1, values2)
            test_name = "Independent t-test"
            
        # Cohen's d effect size
        pooled_std = np.sqrt((np.var(values1, ddof=1) + np.var(values2, ddof=1)) / 2)
        effect_size = (np.mean(values1) - np.mean(values2)) / (pooled_std + 1e-8)
        
        return StatisticalTest(
            test_name=test_name,
            statistic=statistic,
            p_value=p_value,
            effect_size=effect_size,
            is_significant=p_value < alpha,
            alpha=alpha
        )
        
    def wilcoxon_test(
        self,
        values1: np.ndarray,
        values2: np.ndarray,
        alpha: float = 0.05
    ) -> StatisticalTest:
        """
        Perform Wilcoxon signed-rank test (non-parametric paired test).
        
        Args:
            values1: First group values
            values2: Second group values
            alpha: Significance level
            
        Returns:
            StatisticalTest object
        """
        statistic, p_value = stats.wilcoxon(values1, values2)
        
        # Effect size: r = Z / sqrt(N)
        n = len(values1)
        z = stats.norm.ppf(1 - p_value/2)  # Approximate Z from p-value
        effect_size = z / np.sqrt(n)
        
        return StatisticalTest(
            test_name="Wilcoxon signed-rank test",
            statistic=statistic,
            p_value=p_value,
            effect_size=effect_size,
            is_significant=p_value < alpha,
            alpha=alpha
        )
        
    def mann_whitney_test(
        self,
        values1: np.ndarray,
        values2: np.ndarray,
        alpha: float = 0.05
    ) -> StatisticalTest:
        """
        Perform Mann-Whitney U test (non-parametric independent test).
        
        Args:
            values1: First group values
            values2: Second group values
            alpha: Significance level
            
        Returns:
            StatisticalTest object
        """
        statistic, p_value = stats.mannwhitneyu(values1, values2, alternative='two-sided')
        
        # Effect size: r = 1 - (2U)/(n1*n2)
        n1, n2 = len(values1), len(values2)
        effect_size = 1 - (2 * statistic) / (n1 * n2)
        
        return StatisticalTest(
            test_name="Mann-Whitney U test",
            statistic=statistic,
            p_value=p_value,
            effect_size=effect_size,
            is_significant=p_value < alpha,
            alpha=alpha
        )
        
    def calibrate_detection_threshold(
        self,
        watermark_scores: np.ndarray,
        non_watermark_scores: np.ndarray,
        target_fpr: float = 0.01
    ) -> Tuple[float, Dict]:
        """
        Calibrate detection threshold for a target false positive rate.
        
        Args:
            watermark_scores: Similarity scores for watermarked images
            non_watermark_scores: Similarity scores for non-watermarked images
            target_fpr: Target false positive rate (default 1%)
            
        Returns:
            threshold: Calibrated threshold
            metrics: Dictionary with TPR, FPR at threshold
        """
        # Sort scores and find threshold
        thresholds = np.linspace(
            min(non_watermark_scores.min(), watermark_scores.min()),
            max(non_watermark_scores.max(), watermark_scores.max()),
            1000
        )
        
        best_threshold = 0
        best_tpr = 0
        achieved_fpr = 1.0
        
        for thresh in thresholds:
            fpr = np.mean(non_watermark_scores > thresh)
            tpr = np.mean(watermark_scores > thresh)
            
            if fpr <= target_fpr and tpr > best_tpr:
                best_threshold = thresh
                best_tpr = tpr
                achieved_fpr = fpr
                
        # If we couldn't achieve target FPR, use the highest threshold
        if best_threshold == 0:
            best_threshold = np.percentile(non_watermark_scores, 100 * (1 - target_fpr))
            achieved_fpr = np.mean(non_watermark_scores > best_threshold)
            best_tpr = np.mean(watermark_scores > best_threshold)
            
        metrics = {
            "threshold": best_threshold,
            "tpr": best_tpr,
            "fpr": achieved_fpr,
            "target_fpr": target_fpr
        }
        
        return best_threshold, metrics
        
    def analyze_tradeoff(
        self,
        imperceptibility_scores: np.ndarray,  # e.g., PSNR or SSIM
        robustness_scores: np.ndarray,  # e.g., Bit accuracy after attack
        method_names: List[str] = None,
        metric_names: Tuple[str, str] = ("PSNR", "Bit Accuracy")
    ) -> Dict:
        """
        Analyze trade-off between imperceptibility and robustness.
        
        Args:
            imperceptibility_scores: Array of imperceptibility scores
            robustness_scores: Array of robustness scores
            method_names: Names of different methods/configurations
            metric_names: Names of the metrics
            
        Returns:
            Dictionary with analysis results
        """
        # Correlation analysis
        pearson_r, pearson_p = stats.pearsonr(imperceptibility_scores, robustness_scores)
        spearman_r, spearman_p = stats.spearmanr(imperceptibility_scores, robustness_scores)
        
        # Pareto frontier (non-dominated points)
        pareto_mask = np.ones(len(imperceptibility_scores), dtype=bool)
        for i in range(len(imperceptibility_scores)):
            for j in range(len(imperceptibility_scores)):
                if i != j:
                    # Point j dominates point i if j is better in both metrics
                    if (imperceptibility_scores[j] >= imperceptibility_scores[i] and
                        robustness_scores[j] >= robustness_scores[i] and
                        (imperceptibility_scores[j] > imperceptibility_scores[i] or
                         robustness_scores[j] > robustness_scores[i])):
                        pareto_mask[i] = False
                        break
                        
        pareto_indices = np.where(pareto_mask)[0]
        
        results = {
            "pearson_correlation": pearson_r,
            "pearson_p_value": pearson_p,
            "spearman_correlation": spearman_r,
            "spearman_p_value": spearman_p,
            "pareto_indices": pareto_indices,
            "pareto_imperceptibility": imperceptibility_scores[pareto_indices],
            "pareto_robustness": robustness_scores[pareto_indices],
            "metric_names": metric_names
        }
        
        return results
        
    def plot_tradeoff(
        self,
        tradeoff_results: Dict,
        imperceptibility_scores: np.ndarray,
        robustness_scores: np.ndarray,
        method_names: List[str] = None,
        save_path: str = None
    ):
        """
        Plot trade-off analysis results.
        
        Args:
            tradeoff_results: Results from analyze_tradeoff
            imperceptibility_scores: Array of imperceptibility scores
            robustness_scores: Array of robustness scores
            method_names: Names for each point
            save_path: Path to save plot
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot all points
        ax.scatter(imperceptibility_scores, robustness_scores, s=100, alpha=0.6, label='All configurations')
        
        # Highlight Pareto frontier
        pareto_imp = tradeoff_results['pareto_imperceptibility']
        pareto_rob = tradeoff_results['pareto_robustness']
        
        # Sort by imperceptibility for line plot
        sort_idx = np.argsort(pareto_imp)
        ax.plot(pareto_imp[sort_idx], pareto_rob[sort_idx], 'r-', linewidth=2, label='Pareto frontier')
        ax.scatter(pareto_imp, pareto_rob, s=150, c='red', marker='*', label='Pareto optimal')
        
        # Add labels if provided
        if method_names:
            for i, name in enumerate(method_names):
                ax.annotate(name, (imperceptibility_scores[i], robustness_scores[i]),
                           textcoords="offset points", xytext=(5, 5), fontsize=8)
                           
        metric_names = tradeoff_results['metric_names']
        ax.set_xlabel(metric_names[0], fontsize=12)
        ax.set_ylabel(metric_names[1], fontsize=12)
        ax.set_title(f'Trade-off Analysis: {metric_names[0]} vs {metric_names[1]}', fontsize=14)
        
        # Add correlation info
        corr_text = f"Pearson r = {tradeoff_results['pearson_correlation']:.3f} (p = {tradeoff_results['pearson_p_value']:.3e})"
        ax.text(0.05, 0.95, corr_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
               
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.savefig(os.path.join(self.output_dir, "tradeoff_analysis.png"), dpi=150, bbox_inches='tight')
            
        plt.close()
        
    def generate_statistics_report(
        self,
        metric_values: Dict[str, np.ndarray],
        baseline_values: Dict[str, np.ndarray] = None,
        save_path: str = None
    ) -> str:
        """
        Generate comprehensive statistics report.
        
        Args:
            metric_values: Dictionary of metric name -> values
            baseline_values: Optional baseline values for comparison
            save_path: Path to save report
            
        Returns:
            Report string
        """
        lines = []
        lines.append("=" * 80)
        lines.append("STATISTICAL ANALYSIS REPORT")
        lines.append("=" * 80)
        
        # Confidence intervals for each metric
        lines.append("\n1. DESCRIPTIVE STATISTICS (95% CI)")
        lines.append("-" * 40)
        
        for metric_name, values in metric_values.items():
            ci = self.confidence_interval(values)
            lines.append(f"\n{metric_name}:")
            lines.append(f"  Mean: {ci.mean:.4f} ± {ci.std:.4f}")
            lines.append(f"  95% CI: [{ci.ci_lower:.4f}, {ci.ci_upper:.4f}]")
            lines.append(f"  n = {ci.n_samples}")
            
        # Statistical tests vs baseline
        if baseline_values:
            lines.append("\n\n2. STATISTICAL SIGNIFICANCE VS BASELINE")
            lines.append("-" * 40)
            
            for metric_name in metric_values:
                if metric_name in baseline_values:
                    values = metric_values[metric_name]
                    baseline = baseline_values[metric_name]
                    
                    # T-test
                    t_result = self.t_test(values, baseline)
                    lines.append(f"\n{metric_name}:")
                    lines.append(f"  {t_result}")
                    
                    # Effect interpretation
                    if abs(t_result.effect_size) < 0.2:
                        effect_interp = "negligible"
                    elif abs(t_result.effect_size) < 0.5:
                        effect_interp = "small"
                    elif abs(t_result.effect_size) < 0.8:
                        effect_interp = "medium"
                    else:
                        effect_interp = "large"
                    lines.append(f"  Effect size interpretation: {effect_interp}")
                    
        lines.append("\n" + "=" * 80)
        
        report = "\n".join(lines)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
                
        return report

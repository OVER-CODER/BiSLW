"""Statistical Analysis Module for Watermark Evaluation.

Provides descriptive statistics, confidence intervals (t-distribution and bootstrap),
statistical hypothesis testing (t-test, Wilcoxon, Mann-Whitney), detection
threshold calibration, and Pareto frontier trade-off analysis.
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


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
        return f"{self.mean:.4f} [{self.ci_lower:.4f}, {self.ci_upper:.4f}] ({int(self.confidence_level * 100)}% CI, n={self.n_samples})"


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
    """Statistical analysis tools for watermark evaluation."""
    
    def __init__(self, output_dir: str = "results/statistics"):
        """Initializes the StatisticalAnalysis class.

        Args:
            output_dir (str): Directory for saving reports and plots.
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def confidence_interval(
        self,
        values: np.ndarray,
        confidence: float = 0.95
    ) -> ConfidenceInterval:
        """Computes the confidence interval using a Student's t-distribution.

        Args:
            values (np.ndarray): Array of sample values.
            confidence (float): Confidence level fraction (default 0.95).

        Returns:
            ConfidenceInterval: Calculated interval parameters.
        """
        n = len(values)
        mean = np.mean(values)
        std = np.std(values, ddof=1)
        se = std / np.sqrt(n)
        
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
        """Computes confidence interval using bootstrap resampling.

        Useful for non-normal distributions.

        Args:
            values (np.ndarray): Array of sample values.
            confidence (float): Confidence level fraction (default 0.95).
            n_bootstrap (int): Number of resampling iterations.
            statistic (str): Metric to compute ('mean' or 'median').

        Returns:
            ConfidenceInterval: Calculated bootstrap interval.
        """
        n = len(values)
        bootstrap_stats = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(values, size=n, replace=True)
            if statistic == 'mean':
                bootstrap_stats.append(np.mean(sample))
            elif statistic == 'median':
                bootstrap_stats.append(np.median(sample))
                
        bootstrap_stats = np.array(bootstrap_stats)
        
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
        """Performs a paired or independent Student's t-test.

        Args:
            values1 (np.ndarray): First group values.
            values2 (np.ndarray): Second group (or baseline) values.
            paired (bool): Whether the test samples are paired.
            alpha (float): Significance level (default 0.05).

        Returns:
            StatisticalTest: T-test outcomes and Cohen's d effect size.
        """
        if paired:
            statistic, p_value = stats.ttest_rel(values1, values2)
            test_name = "Paired t-test"
        else:
            statistic, p_value = stats.ttest_ind(values1, values2)
            test_name = "Independent t-test"
            
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
        """Performs a Wilcoxon signed-rank test.

        Non-parametric paired comparison.

        Args:
            values1 (np.ndarray): First group values.
            values2 (np.ndarray): Second group values.
            alpha (float): Significance level.

        Returns:
            StatisticalTest: Non-parametric test results.
        """
        statistic, p_value = stats.wilcoxon(values1, values2)
        
        n = len(values1)
        z = stats.norm.ppf(1 - p_value/2)
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
        """Performs a Mann-Whitney U test.

        Non-parametric independent group comparison.

        Args:
            values1 (np.ndarray): First group values.
            values2 (np.ndarray): Second group values.
            alpha (float): Significance level.

        Returns:
            StatisticalTest: Non-parametric test results.
        """
        statistic, p_value = stats.mannwhitneyu(values1, values2, alternative='two-sided')
        
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
        """Calibrates detection threshold for a target False Positive Rate (FPR).

        Args:
            watermark_scores (np.ndarray): Scores of watermarked images.
            non_watermark_scores (np.ndarray): Scores of non-watermarked images.
            target_fpr (float): Maximum acceptable false positive rate.

        Returns:
            Tuple[float, Dict]: Optimal threshold and its achieved metrics.
        """
        thresholds = np.linspace(
            min(non_watermark_scores.min(), watermark_scores.min()),
            max(non_watermark_scores.max(), watermark_scores.max()),
            1000
        )
        
        best_threshold = 0.0
        best_tpr = 0.0
        achieved_fpr = 1.0
        
        for thresh in thresholds:
            fpr = np.mean(non_watermark_scores > thresh)
            tpr = np.mean(watermark_scores > thresh)
            
            if fpr <= target_fpr and tpr > best_tpr:
                best_threshold = thresh
                best_tpr = tpr
                achieved_fpr = fpr
                
        if best_threshold == 0.0:
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
        imperceptibility_scores: np.ndarray,
        robustness_scores: np.ndarray,
        method_names: Optional[List[str]] = None,
        metric_names: Tuple[str, str] = ("PSNR", "Bit Accuracy")
    ) -> Dict:
        """Analyzes trade-offs between imperceptibility and robustness.

        Computes Pearson/Spearman correlations and identifies non-dominated Pareto frontier points.

        Args:
            imperceptibility_scores (np.ndarray): Quality values.
            robustness_scores (np.ndarray): Robustness accuracy values.
            method_names (Optional[List[str]]): Optional configuration name labels.
            metric_names (Tuple[str, str]): Label titles for quality and robustness.

        Returns:
            Dict: Correlation outputs and Pareto optimal indices/coordinates.
        """
        pearson_r, pearson_p = stats.pearsonr(imperceptibility_scores, robustness_scores)
        spearman_r, spearman_p = stats.spearmanr(imperceptibility_scores, robustness_scores)
        
        pareto_mask = np.ones(len(imperceptibility_scores), dtype=bool)
        for i in range(len(imperceptibility_scores)):
            for j in range(len(imperceptibility_scores)):
                if i != j:
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
        method_names: Optional[List[str]] = None,
        save_path: Optional[str] = None
    ):
        """Plots imperceptibility vs robustness scatter and Pareto frontier curves.

        Args:
            tradeoff_results (Dict): Results from analyze_tradeoff.
            imperceptibility_scores (np.ndarray): Original quality metric array.
            robustness_scores (np.ndarray): Original robustness metric array.
            method_names (Optional[List[str]]): Optional text labels for points.
            save_path (Optional[str]): Output file path.
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        ax.scatter(imperceptibility_scores, robustness_scores, s=100, alpha=0.6, label='All configurations')
        
        pareto_imp = tradeoff_results['pareto_imperceptibility']
        pareto_rob = tradeoff_results['pareto_robustness']
        
        sort_idx = np.argsort(pareto_imp)
        ax.plot(pareto_imp[sort_idx], pareto_rob[sort_idx], 'r-', linewidth=2, label='Pareto frontier')
        ax.scatter(pareto_imp, pareto_rob, s=150, c='red', marker='*', label='Pareto optimal')
        
        if method_names:
            for i, name in enumerate(method_names):
                ax.annotate(name, (imperceptibility_scores[i], robustness_scores[i]),
                           textcoords="offset points", xytext=(5, 5), fontsize=8)
                           
        metric_names = tradeoff_results['metric_names']
        ax.set_xlabel(metric_names[0], fontsize=12)
        ax.set_ylabel(metric_names[1], fontsize=12)
        ax.set_title(f'Trade-off Analysis: {metric_names[0]} vs {metric_names[1]}', fontsize=14)
        
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
        baseline_values: Optional[Dict[str, np.ndarray]] = None,
        save_path: Optional[str] = None
    ) -> str:
        """Generates descriptive statistics and significance tests reports.

        Args:
            metric_values (Dict[str, np.ndarray]): Evaluated metrics.
            baseline_values (Optional[Dict[str, np.ndarray]]): Baseline metrics.
            save_path (Optional[str]): Output report path.

        Returns:
            str: Compiled report contents.
        """
        lines = []
        lines.append("=" * 80)
        lines.append("STATISTICAL ANALYSIS REPORT")
        lines.append("=" * 80)
        
        lines.append("\n1. DESCRIPTIVE STATISTICS (95% CI)")
        lines.append("-" * 40)
        
        for metric_name, values in metric_values.items():
            ci = self.confidence_interval(values)
            lines.append(f"\n{metric_name}:")
            lines.append(f"  Mean: {ci.mean:.4f} ± {ci.std:.4f}")
            lines.append(f"  95% CI: [{ci.ci_lower:.4f}, {ci.ci_upper:.4f}]")
            lines.append(f"  n = {ci.n_samples}")
            
        if baseline_values:
            lines.append("\n\n2. STATISTICAL SIGNIFICANCE VS BASELINE")
            lines.append("-" * 40)
            
            for metric_name in metric_values:
                if metric_name in baseline_values:
                    values = metric_values[metric_name]
                    baseline = baseline_values[metric_name]
                    
                    t_result = self.t_test(values, baseline)
                    lines.append(f"\n{metric_name}:")
                    lines.append(f"  {t_result}")
                    
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

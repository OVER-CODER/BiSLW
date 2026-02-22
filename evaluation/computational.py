"""
Computational Analysis Module for Watermark Evaluation.

Provides:
- Runtime per image
- Memory usage
- Overhead vs non-watermarked generation
"""

import torch
import time
import numpy as np
from typing import Dict, List, Callable, Optional
import matplotlib.pyplot as plt
import os
from dataclasses import dataclass
from contextlib import contextmanager


@dataclass
class ComputationalMetrics:
    """Container for computational metrics."""
    operation: str
    avg_time_ms: float
    std_time_ms: float
    min_time_ms: float
    max_time_ms: float
    peak_memory_mb: float
    avg_memory_mb: float
    throughput_imgs_per_sec: float
    
    def __repr__(self):
        return (f"{self.operation}: "
                f"Time={self.avg_time_ms:.2f}±{self.std_time_ms:.2f}ms, "
                f"Memory={self.peak_memory_mb:.1f}MB, "
                f"Throughput={self.throughput_imgs_per_sec:.2f} img/s")


class Timer:
    """Simple timer for measuring execution time."""
    
    def __init__(self, sync_cuda: bool = True):
        self.sync_cuda = sync_cuda
        self.times = []
        
    @contextmanager
    def measure(self):
        """Context manager for timing operations."""
        if self.sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
            
        start = time.perf_counter()
        yield
        
        if self.sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
            
        elapsed = (time.perf_counter() - start) * 1000  # ms
        self.times.append(elapsed)
        
    def reset(self):
        self.times = []
        
    @property
    def avg(self) -> float:
        return np.mean(self.times) if self.times else 0
        
    @property
    def std(self) -> float:
        return np.std(self.times) if len(self.times) > 1 else 0
        
    @property
    def min(self) -> float:
        return np.min(self.times) if self.times else 0
        
    @property
    def max(self) -> float:
        return np.max(self.times) if self.times else 0


class MemoryTracker:
    """Track GPU/CPU memory usage."""
    
    def __init__(self, device: torch.device):
        self.device = device
        self.measurements = []
        
    def _get_memory_mb(self) -> float:
        """Get current memory usage in MB."""
        if self.device.type == 'cuda':
            return torch.cuda.memory_allocated(self.device) / 1024**2
        elif self.device.type == 'mps':
            # MPS doesn't have direct memory tracking, estimate from tensors
            return 0
        else:
            import psutil
            return psutil.Process().memory_info().rss / 1024**2
            
    def _get_peak_memory_mb(self) -> float:
        """Get peak memory usage in MB."""
        if self.device.type == 'cuda':
            return torch.cuda.max_memory_allocated(self.device) / 1024**2
        else:
            return self._get_memory_mb()
            
    @contextmanager
    def track(self):
        """Context manager for tracking memory."""
        if self.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(self.device)
            
        start_mem = self._get_memory_mb()
        yield
        
        end_mem = self._get_memory_mb()
        peak_mem = self._get_peak_memory_mb()
        
        self.measurements.append({
            'start': start_mem,
            'end': end_mem,
            'peak': peak_mem,
            'delta': end_mem - start_mem
        })
        
    def reset(self):
        self.measurements = []
        
    @property
    def avg_memory(self) -> float:
        if not self.measurements:
            return 0
        return np.mean([m['delta'] for m in self.measurements])
        
    @property
    def peak_memory(self) -> float:
        if not self.measurements:
            return 0
        return np.max([m['peak'] for m in self.measurements])


class ComputationalAnalysis:
    """
    Computational analysis framework for watermark operations.
    """
    
    def __init__(
        self,
        device: torch.device = None,
        output_dir: str = "results/computational"
    ):
        """
        Args:
            device: Device for computation
            output_dir: Directory for saving results
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.timer = Timer(sync_cuda=(self.device.type == 'cuda'))
        self.memory_tracker = MemoryTracker(self.device)
        
    def warmup(self, func: Callable, *args, n_warmup: int = 5, **kwargs):
        """Run warmup iterations."""
        for _ in range(n_warmup):
            with torch.no_grad():
                _ = func(*args, **kwargs)
                
    def benchmark_operation(
        self,
        operation_name: str,
        func: Callable,
        *args,
        n_iterations: int = 100,
        n_warmup: int = 10,
        batch_size: int = 1,
        **kwargs
    ) -> ComputationalMetrics:
        """
        Benchmark a single operation.
        
        Args:
            operation_name: Name of the operation
            func: Function to benchmark
            args: Arguments to pass to function
            n_iterations: Number of benchmark iterations
            n_warmup: Number of warmup iterations
            batch_size: Batch size for throughput calculation
            kwargs: Keyword arguments to pass to function
            
        Returns:
            ComputationalMetrics object
        """
        self.timer.reset()
        self.memory_tracker.reset()
        
        # Warmup
        self.warmup(func, *args, n_warmup=n_warmup, **kwargs)
        
        # Benchmark
        for _ in range(n_iterations):
            with torch.no_grad():
                with self.timer.measure():
                    with self.memory_tracker.track():
                        _ = func(*args, **kwargs)
                        
        throughput = (batch_size * 1000) / self.timer.avg  # images per second
        
        return ComputationalMetrics(
            operation=operation_name,
            avg_time_ms=self.timer.avg,
            std_time_ms=self.timer.std,
            min_time_ms=self.timer.min,
            max_time_ms=self.timer.max,
            peak_memory_mb=self.memory_tracker.peak_memory,
            avg_memory_mb=self.memory_tracker.avg_memory,
            throughput_imgs_per_sec=throughput
        )
        
    @torch.no_grad()
    def analyze_watermark_pipeline(
        self,
        vae,
        splitter,
        recombiner,
        encoder_l,
        encoder_h,
        decoder_l,
        decoder_h,
        image_size: int = 512,
        batch_sizes: List[int] = None,
        n_iterations: int = 50
    ) -> Dict[str, ComputationalMetrics]:
        """
        Analyze computational costs of the full watermarking pipeline.
        
        Args:
            vae: VAE wrapper
            splitter: Latent splitter
            recombiner: Latent recombiner
            encoder_l, encoder_h: Watermark encoders
            decoder_l, decoder_h: Watermark decoders
            image_size: Image size for testing
            batch_sizes: List of batch sizes to test
            n_iterations: Number of iterations per test
            
        Returns:
            Dictionary mapping operation names to metrics
        """
        if batch_sizes is None:
            batch_sizes = [1, 2, 4, 8]
            
        results = {}
        
        for batch_size in batch_sizes:
            print(f"\nBenchmarking with batch_size={batch_size}...")
            
            # Create test inputs
            images = torch.randn(batch_size, 3, image_size, image_size, device=self.device)
            watermark = torch.randn(batch_size, 64, device=self.device)
            
            # Benchmark VAE Encoding
            results[f'vae_encode_b{batch_size}'] = self.benchmark_operation(
                f"VAE Encode (B={batch_size})",
                vae.encode,
                images,
                n_iterations=n_iterations,
                batch_size=batch_size
            )
            
            # Get latent for subsequent operations
            z = vae.encode(images)
            
            # Benchmark Latent Splitting
            results[f'split_b{batch_size}'] = self.benchmark_operation(
                f"Latent Split (B={batch_size})",
                splitter,
                z,
                n_iterations=n_iterations,
                batch_size=batch_size
            )
            
            z_low, z_high = splitter(z)
            
            # Benchmark Watermark Encoding
            def encode_watermark():
                z_low_wm = encoder_l(z_low, watermark, alpha=1.0)
                z_high_wm = encoder_h(z_high, watermark, alpha=1.0)
                return z_low_wm, z_high_wm
                
            results[f'wm_encode_b{batch_size}'] = self.benchmark_operation(
                f"Watermark Encode (B={batch_size})",
                encode_watermark,
                n_iterations=n_iterations,
                batch_size=batch_size
            )
            
            z_low_wm, z_high_wm = encode_watermark()
            
            # Benchmark Recombination
            results[f'recombine_b{batch_size}'] = self.benchmark_operation(
                f"Latent Recombine (B={batch_size})",
                recombiner,
                z_low_wm, z_high_wm,
                n_iterations=n_iterations,
                batch_size=batch_size
            )
            
            z_wm = recombiner(z_low_wm, z_high_wm)
            
            # Benchmark VAE Decoding
            results[f'vae_decode_b{batch_size}'] = self.benchmark_operation(
                f"VAE Decode (B={batch_size})",
                vae.decode,
                z_wm,
                n_iterations=n_iterations,
                batch_size=batch_size
            )
            
            # Benchmark Watermark Decoding
            z_wm_low, z_wm_high = splitter(z_wm)
            
            def decode_watermark():
                w_l = decoder_l(z_wm_low)
                w_h = decoder_h(z_wm_high)
                return (w_l + w_h) / 2
                
            results[f'wm_decode_b{batch_size}'] = self.benchmark_operation(
                f"Watermark Decode (B={batch_size})",
                decode_watermark,
                n_iterations=n_iterations,
                batch_size=batch_size
            )
            
            # Full pipeline: Image -> Watermarked Image
            def full_embed_pipeline():
                z = vae.encode(images)
                z_low, z_high = splitter(z)
                z_low_wm = encoder_l(z_low, watermark, alpha=1.0)
                z_high_wm = encoder_h(z_high, watermark, alpha=1.0)
                z_wm = recombiner(z_low_wm, z_high_wm)
                return vae.decode(z_wm)
                
            results[f'full_embed_b{batch_size}'] = self.benchmark_operation(
                f"Full Embed Pipeline (B={batch_size})",
                full_embed_pipeline,
                n_iterations=n_iterations,
                batch_size=batch_size
            )
            
            # Full pipeline: Watermarked Image -> Watermark
            images_wm = full_embed_pipeline()
            
            def full_extract_pipeline():
                z = vae.encode(images_wm)
                z_low, z_high = splitter(z)
                w_l = decoder_l(z_low)
                w_h = decoder_h(z_high)
                return (w_l + w_h) / 2
                
            results[f'full_extract_b{batch_size}'] = self.benchmark_operation(
                f"Full Extract Pipeline (B={batch_size})",
                full_extract_pipeline,
                n_iterations=n_iterations,
                batch_size=batch_size
            )
            
        return results
        
    def compute_overhead(
        self,
        results: Dict[str, ComputationalMetrics],
        batch_size: int = 1
    ) -> Dict[str, float]:
        """
        Compute overhead vs non-watermarked generation.
        
        Args:
            results: Results from analyze_watermark_pipeline
            batch_size: Batch size to analyze
            
        Returns:
            Dictionary with overhead percentages
        """
        # Time for just encoding/decoding without watermarking
        vae_time = (
            results[f'vae_encode_b{batch_size}'].avg_time_ms +
            results[f'vae_decode_b{batch_size}'].avg_time_ms
        )
        
        # Time for full watermark embedding
        full_time = results[f'full_embed_b{batch_size}'].avg_time_ms
        
        # Overhead
        watermark_overhead_ms = full_time - vae_time
        watermark_overhead_pct = (watermark_overhead_ms / vae_time) * 100
        
        # Memory overhead (rough estimate)
        vae_memory = (
            results[f'vae_encode_b{batch_size}'].peak_memory_mb +
            results[f'vae_decode_b{batch_size}'].peak_memory_mb
        ) / 2
        full_memory = results[f'full_embed_b{batch_size}'].peak_memory_mb
        memory_overhead_pct = ((full_memory - vae_memory) / vae_memory) * 100 if vae_memory > 0 else 0
        
        return {
            'vae_only_time_ms': vae_time,
            'full_watermark_time_ms': full_time,
            'watermark_overhead_ms': watermark_overhead_ms,
            'watermark_overhead_pct': watermark_overhead_pct,
            'vae_memory_mb': vae_memory,
            'full_memory_mb': full_memory,
            'memory_overhead_pct': memory_overhead_pct
        }
        
    def plot_results(
        self,
        results: Dict[str, ComputationalMetrics],
        batch_sizes: List[int] = None,
        save_path: str = None
    ):
        """
        Plot computational analysis results.
        
        Args:
            results: Results from analyze_watermark_pipeline
            batch_sizes: Batch sizes to plot
            save_path: Path to save plot
        """
        if batch_sizes is None:
            batch_sizes = [1, 2, 4, 8]
            
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Extract data
        operations = ['vae_encode', 'split', 'wm_encode', 'recombine', 'vae_decode', 'wm_decode']
        
        # Time breakdown per batch size
        ax1 = axes[0, 0]
        x = np.arange(len(operations))
        width = 0.2
        
        for i, bs in enumerate(batch_sizes):
            times = [results.get(f'{op}_b{bs}', ComputationalMetrics('', 0, 0, 0, 0, 0, 0, 0)).avg_time_ms 
                    for op in operations]
            ax1.bar(x + i * width, times, width, label=f'B={bs}')
            
        ax1.set_ylabel('Time (ms)')
        ax1.set_title('Time Breakdown by Operation')
        ax1.set_xticks(x + width * (len(batch_sizes) - 1) / 2)
        ax1.set_xticklabels([op.replace('_', ' ').title() for op in operations], rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Throughput vs batch size
        ax2 = axes[0, 1]
        throughputs_embed = [results[f'full_embed_b{bs}'].throughput_imgs_per_sec for bs in batch_sizes]
        throughputs_extract = [results[f'full_extract_b{bs}'].throughput_imgs_per_sec for bs in batch_sizes]
        
        ax2.plot(batch_sizes, throughputs_embed, 'o-', label='Embed', markersize=10)
        ax2.plot(batch_sizes, throughputs_extract, 's-', label='Extract', markersize=10)
        ax2.set_xlabel('Batch Size')
        ax2.set_ylabel('Throughput (images/sec)')
        ax2.set_title('Throughput vs Batch Size')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Memory usage
        ax3 = axes[1, 0]
        memories = [results[f'full_embed_b{bs}'].peak_memory_mb for bs in batch_sizes]
        ax3.bar(batch_sizes, memories, color='coral', alpha=0.7)
        ax3.set_xlabel('Batch Size')
        ax3.set_ylabel('Peak Memory (MB)')
        ax3.set_title('Peak Memory Usage')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Overhead analysis
        ax4 = axes[1, 1]
        overheads = []
        for bs in batch_sizes:
            overhead = self.compute_overhead(results, bs)
            overheads.append(overhead['watermark_overhead_pct'])
            
        ax4.bar(batch_sizes, overheads, color='purple', alpha=0.7)
        ax4.set_xlabel('Batch Size')
        ax4.set_ylabel('Overhead (%)')
        ax4.set_title('Watermarking Overhead vs VAE-only')
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        else:
            plt.savefig(os.path.join(self.output_dir, "computational_analysis.png"), dpi=150, bbox_inches='tight')
            
        plt.close()
        
    def generate_report(
        self,
        results: Dict[str, ComputationalMetrics],
        batch_sizes: List[int] = None,
        save_path: str = None
    ) -> str:
        """
        Generate computational analysis report.
        
        Args:
            results: Results from analyze_watermark_pipeline
            batch_sizes: Batch sizes to include
            save_path: Path to save report
            
        Returns:
            Report string
        """
        if batch_sizes is None:
            batch_sizes = [1, 2, 4, 8]
            
        lines = []
        lines.append("=" * 80)
        lines.append("COMPUTATIONAL ANALYSIS REPORT")
        lines.append("=" * 80)
        
        # Summary table
        lines.append("\n1. TIMING SUMMARY (per operation)")
        lines.append("-" * 60)
        lines.append(f"{'Operation':<30} {'Time (ms)':>15} {'Throughput':>15}")
        lines.append("-" * 60)
        
        for key, metric in results.items():
            if '_b1' in key:  # Show batch_size=1 results
                lines.append(f"{metric.operation:<30} {metric.avg_time_ms:>12.2f}ms {metric.throughput_imgs_per_sec:>12.2f}/s")
                
        # Overhead analysis
        lines.append("\n\n2. OVERHEAD ANALYSIS")
        lines.append("-" * 60)
        
        for bs in batch_sizes:
            overhead = self.compute_overhead(results, bs)
            lines.append(f"\nBatch Size = {bs}:")
            lines.append(f"  VAE-only time: {overhead['vae_only_time_ms']:.2f} ms")
            lines.append(f"  Full watermark time: {overhead['full_watermark_time_ms']:.2f} ms")
            lines.append(f"  Watermark overhead: {overhead['watermark_overhead_ms']:.2f} ms ({overhead['watermark_overhead_pct']:.1f}%)")
            lines.append(f"  Peak memory: {overhead['full_memory_mb']:.1f} MB")
            
        # Scalability
        lines.append("\n\n3. SCALABILITY")
        lines.append("-" * 60)
        
        for bs in batch_sizes:
            metric = results[f'full_embed_b{bs}']
            time_per_image = metric.avg_time_ms / bs
            lines.append(f"Batch Size {bs}: {metric.avg_time_ms:.2f}ms total, {time_per_image:.2f}ms/image, {metric.throughput_imgs_per_sec:.2f} img/s")
            
        lines.append("\n" + "=" * 80)
        
        report = "\n".join(lines)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
        else:
            with open(os.path.join(self.output_dir, "computational_report.txt"), 'w') as f:
                f.write(report)
                
        return report

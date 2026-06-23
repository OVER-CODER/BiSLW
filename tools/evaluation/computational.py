"""Computational Analysis Module for Watermark Evaluation.

Measures the runtime execution times, memory usage overhead, and throughput latency
for watermarked pipeline components (encoding, decoding, splitting, and recovery).
"""

import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch


@dataclass
class ComputationalMetrics:
    """Container for holding parsed computational benchmarks."""
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
    """Simple timer utility for measuring execution duration."""
    
    def __init__(self, sync_cuda: bool = True):
        """Initializes the Timer.

        Args:
            sync_cuda (bool): Whether to synchronize GPU operations.
        """
        self.sync_cuda = sync_cuda
        self.times = []
        
    @contextmanager
    def measure(self):
        """Context manager to measure execution time of a code block."""
        if self.sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
            
        start = time.perf_counter()
        yield
        
        if self.sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
            
        elapsed = (time.perf_counter() - start) * 1000.0
        self.times.append(elapsed)
        
    def reset(self):
        """Resets the recorded times list."""
        self.times = []
        
    @property
    def avg(self) -> float:
        return float(np.mean(self.times)) if self.times else 0.0
        
    @property
    def std(self) -> float:
        return float(np.std(self.times)) if len(self.times) > 1 else 0.0
        
    @property
    def min(self) -> float:
        return float(np.min(self.times)) if self.times else 0.0
        
    @property
    def max(self) -> float:
        return float(np.max(self.times)) if self.times else 0.0


class MemoryTracker:
    """Tracks active CPU/GPU memory usage during model operations."""
    
    def __init__(self, device: torch.device):
        """Initializes the MemoryTracker.

        Args:
            device (torch.device): Device being tracked.
        """
        self.device = device
        self.measurements = []
        
    def _get_memory_mb(self) -> float:
        """Estimates current memory usage in Megabytes.

        Returns:
            float: Memory usage.
        """
        if self.device.type == 'cuda':
            return torch.cuda.memory_allocated(self.device) / (1024.0 ** 2)
        elif self.device.type == 'mps':
            return 0.0
        else:
            import psutil
            return psutil.Process().memory_info().rss / (1024.0 ** 2)
            
    def _get_peak_memory_mb(self) -> float:
        """Gets peak memory usage in Megabytes.

        Returns:
            float: Peak memory usage.
        """
        if self.device.type == 'cuda':
            return torch.cuda.max_memory_allocated(self.device) / (1024.0 ** 2)
        else:
            return self._get_memory_mb()
            
    @contextmanager
    def track(self):
        """Context manager to track memory allocation delta."""
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
        """Resets the tracked memory stats."""
        self.measurements = []
        
    @property
    def avg_memory(self) -> float:
        if not self.measurements:
            return 0.0
        return float(np.mean([m['delta'] for m in self.measurements]))
        
    @property
    def peak_memory(self) -> float:
        if not self.measurements:
            return 0.0
        return float(np.max([m['peak'] for m in self.measurements]))


class ComputationalAnalysis:
    """Computational analysis framework for watermark operations."""
    
    def __init__(
        self,
        device: Optional[torch.device] = None,
        output_dir: str = "results/computational"
    ):
        """Initializes the ComputationalAnalysis.

        Args:
            device: Device for computation (e.g. CPU, CUDA, MPS).
            output_dir: Directory for saving generated reports.
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.timer = Timer(sync_cuda=(self.device.type == 'cuda'))
        self.memory_tracker = MemoryTracker(self.device)
        
    def warmup(self, func: Callable, *args, n_warmup: int = 5, **kwargs):
        """Warms up compilation/caching mechanisms.

        Args:
            func (Callable): Function to warm up.
            n_warmup (int): Warmup steps.
        """
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
        """Benchmarks runtime and memory of a single operation.

        Args:
            operation_name (str): Named ID of this benchmark step.
            func (Callable): Targeted function to benchmark.
            n_iterations (int): Test repetitions.
            n_warmup (int): Warmup steps.
            batch_size (int): Current evaluation batch size.

        Returns:
            ComputationalMetrics: Collected benchmarks container.
        """
        self.timer.reset()
        self.memory_tracker.reset()
        
        self.warmup(func, *args, n_warmup=n_warmup, **kwargs)
        
        for _ in range(n_iterations):
            with torch.no_grad():
                with self.timer.measure():
                    with self.memory_tracker.track():
                        _ = func(*args, **kwargs)
                        
        throughput = (batch_size * 1000.0) / self.timer.avg
        
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
        batch_sizes: Optional[List[int]] = None,
        n_iterations: int = 50
    ) -> Dict[str, ComputationalMetrics]:
        """Runs benchmarks on each individual stage of the watermarking pipeline.

        Args:
            vae: VAE wrapper module.
            splitter: Latent splitter module.
            recombiner: Latent recombiner module.
            encoder_l: Low-frequency encoder.
            encoder_h: High-frequency encoder.
            decoder_l: Low-frequency decoder.
            decoder_h: High-frequency decoder.
            image_size (int): Dimensions of synthetic test images.
            batch_sizes (Optional[List[int]]): List of batch sizes to benchmark.
            n_iterations (int): Iterations per benchmark run.

        Returns:
            Dict[str, ComputationalMetrics]: Dict of key-labeled measurements.
        """
        if batch_sizes is None:
            batch_sizes = [1, 2, 4, 8]
            
        results = {}
        
        for batch_size in batch_sizes:
            print(f"\nBenchmarking with batch_size={batch_size}...")
            
            images = torch.randn(batch_size, 3, image_size, image_size, device=self.device)
            watermark = torch.randn(batch_size, 64, device=self.device)
            
            results[f'vae_encode_b{batch_size}'] = self.benchmark_operation(
                f"VAE Encode (B={batch_size})",
                vae.encode,
                images,
                n_iterations=n_iterations,
                batch_size=batch_size
            )
            
            z = vae.encode(images)
            
            results[f'split_b{batch_size}'] = self.benchmark_operation(
                f"Latent Split (B={batch_size})",
                splitter,
                z,
                n_iterations=n_iterations,
                batch_size=batch_size
            )
            
            z_low, z_high = splitter(z)
            
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
            
            results[f'recombine_b{batch_size}'] = self.benchmark_operation(
                f"Latent Recombine (B={batch_size})",
                recombiner,
                z_low_wm, z_high_wm,
                n_iterations=n_iterations,
                batch_size=batch_size
            )
            
            z_wm = recombiner(z_low_wm, z_high_wm)
            
            results[f'vae_decode_b{batch_size}'] = self.benchmark_operation(
                f"VAE Decode (B={batch_size})",
                vae.decode,
                z_wm,
                n_iterations=n_iterations,
                batch_size=batch_size
            )
            
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
            
            def full_embed_pipeline():
                z_latent = vae.encode(images)
                z_l, z_h = splitter(z_latent)
                z_l_wm = encoder_l(z_l, watermark, alpha=1.0)
                z_h_wm = encoder_h(z_h, watermark, alpha=1.0)
                z_w = recombiner(z_l_wm, z_h_wm)
                return vae.decode(z_w)
                
            results[f'full_embed_b{batch_size}'] = self.benchmark_operation(
                f"Full Embed Pipeline (B={batch_size})",
                full_embed_pipeline,
                n_iterations=n_iterations,
                batch_size=batch_size
            )
            
            images_wm = full_embed_pipeline()
            
            def full_extract_pipeline():
                z_latent = vae.encode(images_wm)
                z_l, z_h = splitter(z_latent)
                w_l = decoder_l(z_l)
                w_h = decoder_h(z_h)
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
        """Calculates overhead comparison metrics vs standard VAE operations.

        Args:
            results (Dict[str, ComputationalMetrics]): Measured stage outputs.
            batch_size (int): Batch size targeted for calculation.

        Returns:
            Dict[str, float]: Decoded overhead ratios/deltas.
        """
        vae_time = (
            results[f'vae_encode_b{batch_size}'].avg_time_ms +
            results[f'vae_decode_b{batch_size}'].avg_time_ms
        )
        
        full_time = results[f'full_embed_b{batch_size}'].avg_time_ms
        
        watermark_overhead_ms = full_time - vae_time
        watermark_overhead_pct = (watermark_overhead_ms / vae_time) * 100.0
        
        vae_memory = (
            results[f'vae_encode_b{batch_size}'].peak_memory_mb +
            results[f'vae_decode_b{batch_size}'].peak_memory_mb
        ) / 2.0
        full_memory = results[f'full_embed_b{batch_size}'].peak_memory_mb
        memory_overhead_pct = ((full_memory - vae_memory) / vae_memory) * 100.0 if vae_memory > 0 else 0.0
        
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
        batch_sizes: Optional[List[int]] = None,
        save_path: Optional[str] = None
    ):
        """Generates plots showing runtime scalability and memory overheads.

        Args:
            results (Dict[str, ComputationalMetrics]): Benchmark metrics dict.
            batch_sizes (Optional[List[int]]): Tested batch configurations list.
            save_path (Optional[str]): Output filename path.
        """
        if batch_sizes is None:
            batch_sizes = [1, 2, 4, 8]
            
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        operations = ['vae_encode', 'split', 'wm_encode', 'recombine', 'vae_decode', 'wm_decode']
        
        ax1 = axes[0, 0]
        x = np.arange(len(operations))
        width = 0.2
        
        for i, bs in enumerate(batch_sizes):
            times = [
                results.get(
                    f'{op}_b{bs}', 
                    ComputationalMetrics('', 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
                ).avg_time_ms 
                for op in operations
            ]
            ax1.bar(x + i * width, times, width, label=f'B={bs}')
            
        ax1.set_ylabel('Time (ms)')
        ax1.set_title('Time Breakdown by Operation')
        ax1.set_xticks(x + width * (len(batch_sizes) - 1) / 2.0)
        ax1.set_xticklabels([op.replace('_', ' ').title() for op in operations], rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        
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
        
        ax3 = axes[1, 0]
        memories = [results[f'full_embed_b{bs}'].peak_memory_mb for bs in batch_sizes]
        ax3.bar(batch_sizes, memories, color='coral', alpha=0.7)
        ax3.set_xlabel('Batch Size')
        ax3.set_ylabel('Peak Memory (MB)')
        ax3.set_title('Peak Memory Usage')
        ax3.grid(True, alpha=0.3, axis='y')
        
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
        batch_sizes: Optional[List[int]] = None,
        save_path: Optional[str] = None
    ) -> str:
        """Generates a text report summarizing the computational benchmarks.

        Args:
            results (Dict[str, ComputationalMetrics]): Measured pipeline values.
            batch_sizes (Optional[List[int]]): Tested batch size list.
            save_path (Optional[str]): Output report file path.

        Returns:
            str: Compiled report contents.
        """
        if batch_sizes is None:
            batch_sizes = [1, 2, 4, 8]
            
        lines = []
        lines.append("=" * 80)
        lines.append("COMPUTATIONAL ANALYSIS REPORT")
        lines.append("=" * 80)
        
        lines.append("\n1. TIMING SUMMARY (per operation)")
        lines.append("-" * 60)
        lines.append(f"{'Operation':<30} {'Time (ms)':>15} {'Throughput':>15}")
        lines.append("-" * 60)
        
        for key, metric in results.items():
            if '_b1' in key:
                lines.append(f"{metric.operation:<30} {metric.avg_time_ms:>12.2f}ms {metric.throughput_imgs_per_sec:>12.2f}/s")
                
        lines.append("\n\n2. OVERHEAD ANALYSIS")
        lines.append("-" * 60)
        
        for bs in batch_sizes:
            overhead = self.compute_overhead(results, bs)
            lines.append(f"\nBatch Size = {bs}:")
            lines.append(f"  VAE-only time: {overhead['vae_only_time_ms']:.2f} ms")
            lines.append(f"  Full watermark time: {overhead['full_watermark_time_ms']:.2f} ms")
            lines.append(f"  Watermark overhead: {overhead['watermark_overhead_ms']:.2f} ms ({overhead['watermark_overhead_pct']:.1f}%)")
            lines.append(f"  Peak memory: {overhead['full_memory_mb']:.1f} MB")
            
        lines.append("\n\n3. SCALABILITY")
        lines.append("-" * 60)
        
        for bs in batch_sizes:
            metric = results[f'full_embed_b{bs}']
            time_per_image = metric.avg_time_ms / bs
            lines.append(
                f"Batch Size {bs}: {metric.avg_time_ms:.2f}ms total, "
                f"{time_per_image:.2f}ms/image, {metric.throughput_imgs_per_sec:.2f} img/s"
            )
            
        lines.append("\n" + "=" * 80)
        
        report = "\n".join(lines)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
        else:
            with open(os.path.join(self.output_dir, "computational_report.txt"), 'w') as f:
                f.write(report)
                
        return report

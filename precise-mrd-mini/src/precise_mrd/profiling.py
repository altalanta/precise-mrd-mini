"""
Performance profiling module for MRD pipeline.

This module provides:
- Runtime performance measurement
- Memory usage tracking
- Bottleneck identification
- Performance reporting
"""

import time
import psutil
import logging
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
from contextlib import contextmanager
from functools import wraps


@dataclass
class ProfileResult:
    """Container for profiling results."""
    name: str
    duration: float
    memory_peak: float
    memory_delta: float
    cpu_percent: float
    metadata: Dict[str, Any]


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    total_runtime: float
    peak_memory_mb: float
    average_cpu_percent: float
    operations_per_second: float
    bottlenecks: List[str]
    detailed_timings: Dict[str, float]


class Profiler:
    """Performance profiler for MRD pipeline components."""
    
    def __init__(self, enable_detailed_profiling: bool = True):
        """Initialize profiler."""
        self.enable_detailed_profiling = enable_detailed_profiling
        self.results: List[ProfileResult] = []
        self.start_time: Optional[float] = None
        self.process = psutil.Process()
        self.logger = logging.getLogger(__name__)
    
    @contextmanager
    def profile(self, operation_name: str, metadata: Optional[Dict[str, Any]] = None):
        """Context manager for profiling operations."""
        if not self.enable_detailed_profiling:
            yield
            return
        
        # Record start state
        start_time = time.time()
        start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        start_cpu = self.process.cpu_percent()
        
        try:
            yield
        finally:
            # Record end state
            end_time = time.time()
            end_memory = self.process.memory_info().rss / 1024 / 1024  # MB
            end_cpu = self.process.cpu_percent()
            
            duration = end_time - start_time
            memory_delta = end_memory - start_memory
            memory_peak = max(start_memory, end_memory)
            avg_cpu = (start_cpu + end_cpu) / 2
            
            result = ProfileResult(
                name=operation_name,
                duration=duration,
                memory_peak=memory_peak,
                memory_delta=memory_delta,
                cpu_percent=avg_cpu,
                metadata=metadata or {}
            )
            
            self.results.append(result)
            
            if duration > 1.0:  # Log slow operations
                self.logger.info(f"Profiled {operation_name}: {duration:.2f}s, {memory_delta:+.1f}MB")
    
    def profile_function(self, func: Callable, *args, **kwargs) -> Any:
        """Profile a function call."""
        with self.profile(func.__name__):
            return func(*args, **kwargs)
    
    def start_session(self) -> None:
        """Start a profiling session."""
        self.start_time = time.time()
        self.results.clear()
        self.logger.info("Profiling session started")
    
    def end_session(self) -> PerformanceMetrics:
        """End profiling session and return metrics."""
        if self.start_time is None:
            raise RuntimeError("Profiling session not started")
        
        total_runtime = time.time() - self.start_time
        
        if not self.results:
            return PerformanceMetrics(
                total_runtime=total_runtime,
                peak_memory_mb=0.0,
                average_cpu_percent=0.0,
                operations_per_second=0.0,
                bottlenecks=[],
                detailed_timings={}
            )
        
        # Calculate metrics
        peak_memory = max(result.memory_peak for result in self.results)
        avg_cpu = np.mean([result.cpu_percent for result in self.results])
        operations_per_second = len(self.results) / total_runtime
        
        # Identify bottlenecks (operations taking > 10% of total time)
        bottlenecks = []
        detailed_timings = {}
        
        for result in self.results:
            detailed_timings[result.name] = result.duration
            if result.duration > total_runtime * 0.1:
                bottlenecks.append(f"{result.name}: {result.duration:.2f}s")
        
        metrics = PerformanceMetrics(
            total_runtime=total_runtime,
            peak_memory_mb=peak_memory,
            average_cpu_percent=avg_cpu,
            operations_per_second=operations_per_second,
            bottlenecks=bottlenecks,
            detailed_timings=detailed_timings
        )
        
        self.logger.info(f"Profiling session completed: {total_runtime:.2f}s, {peak_memory:.1f}MB peak")
        return metrics
    
    def get_summary_table(self) -> Dict[str, Any]:
        """Get summary table of profiling results."""
        if not self.results:
            return {"error": "No profiling data available"}
        
        # Group results by operation name
        operation_stats = {}
        
        for result in self.results:
            if result.name not in operation_stats:
                operation_stats[result.name] = {
                    "count": 0,
                    "total_duration": 0.0,
                    "total_memory_delta": 0.0,
                    "durations": [],
                    "memory_deltas": []
                }
            
            stats = operation_stats[result.name]
            stats["count"] += 1
            stats["total_duration"] += result.duration
            stats["total_memory_delta"] += result.memory_delta
            stats["durations"].append(result.duration)
            stats["memory_deltas"].append(result.memory_delta)
        
        # Calculate summary statistics
        summary = {}
        for name, stats in operation_stats.items():
            summary[name] = {
                "count": stats["count"],
                "total_duration": stats["total_duration"],
                "mean_duration": stats["total_duration"] / stats["count"],
                "median_duration": np.median(stats["durations"]),
                "std_duration": np.std(stats["durations"]),
                "min_duration": min(stats["durations"]),
                "max_duration": max(stats["durations"]),
                "total_memory_delta": stats["total_memory_delta"],
                "mean_memory_delta": stats["total_memory_delta"] / stats["count"]
            }
        
        return summary


class UMIProfiler:
    """Specialized profiler for UMI processing operations."""
    
    def __init__(self):
        """Initialize UMI profiler."""
        self.profiler = Profiler()
        self.umi_metrics = {}
    
    def profile_umi_consensus(
        self,
        umi_processor,
        reads: List[Any],
        n_reads: int
    ) -> Dict[str, float]:
        """Profile UMI consensus calling performance."""
        
        self.profiler.start_session()
        
        with self.profiler.profile("umi_grouping", {"n_reads": n_reads}):
            # Would call actual UMI grouping
            families = umi_processor.process_reads(reads)
        
        with self.profiler.profile("consensus_calling", {"n_families": len(families)}):
            # Would call consensus calling
            consensus_data = umi_processor.get_consensus_counts(families)
        
        metrics = self.profiler.end_session()
        
        # Calculate UMI-specific metrics
        reads_per_second = n_reads / metrics.total_runtime
        families_per_second = len(families) / metrics.total_runtime
        
        umi_metrics = {
            "reads_per_second": reads_per_second,
            "families_per_second": families_per_second,
            "total_runtime": metrics.total_runtime,
            "peak_memory_mb": metrics.peak_memory_mb,
            "consensus_efficiency": len(families) / n_reads if n_reads > 0 else 0.0
        }
        
        return umi_metrics
    
    def benchmark_family_sizes(
        self,
        umi_processor,
        family_sizes: List[int] = [1000, 5000, 10000, 50000, 100000]
    ) -> Dict[int, Dict[str, float]]:
        """Benchmark UMI processing across different dataset sizes."""
        
        benchmark_results = {}
        
        for n_reads in family_sizes:
            self.profiler.logger.info(f"Benchmarking {n_reads:,} reads...")
            
            # Generate synthetic reads
            from .io import SyntheticReadGenerator, TargetSite
            generator = SyntheticReadGenerator()
            site = TargetSite("chr1", 1000000, "A", "T", "ACG")
            
            reads = generator.generate_reads_for_site(
                site=site,
                n_umi_families=n_reads // 5,  # Average family size of 5
                allele_fraction=0.01
            )
            
            # Profile the processing
            metrics = self.profile_umi_consensus(umi_processor, reads, len(reads))
            benchmark_results[n_reads] = metrics
        
        return benchmark_results


def performance_decorator(operation_name: str = None):
    """Decorator for profiling function performance."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get profiler from args/kwargs or create a new one
            profiler = getattr(args[0], '_profiler', None) if args else None
            if profiler is None:
                profiler = Profiler()
            
            name = operation_name or func.__name__
            with profiler.profile(name):
                return func(*args, **kwargs)
        return wrapper
    return decorator


class PerformanceReporter:
    """Generate performance reports and recommendations."""
    
    def __init__(self):
        """Initialize performance reporter."""
        self.logger = logging.getLogger(__name__)
    
    def generate_performance_report(
        self,
        metrics: PerformanceMetrics,
        benchmark_data: Optional[Dict[int, Dict[str, float]]] = None
    ) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        
        report = {
            "summary": {
                "total_runtime_seconds": metrics.total_runtime,
                "peak_memory_mb": metrics.peak_memory_mb,
                "average_cpu_percent": metrics.average_cpu_percent,
                "operations_per_second": metrics.operations_per_second
            },
            "bottlenecks": metrics.bottlenecks,
            "detailed_timings": metrics.detailed_timings,
            "recommendations": self._generate_recommendations(metrics),
            "benchmark_comparison": {}
        }
        
        if benchmark_data:
            report["benchmark_comparison"] = self._compare_with_benchmarks(
                metrics, benchmark_data
            )
        
        return report
    
    def _generate_recommendations(self, metrics: PerformanceMetrics) -> List[str]:
        """Generate performance recommendations."""
        recommendations = []
        
        # Memory recommendations
        if metrics.peak_memory_mb > 8000:  # > 8GB
            recommendations.append(
                "High memory usage detected. Consider processing data in smaller chunks."
            )
        
        # Runtime recommendations
        if metrics.total_runtime > 3600:  # > 1 hour
            recommendations.append(
                "Long runtime detected. Consider enabling parallel processing or using optimized implementations."
            )
        
        # Bottleneck recommendations
        if len(metrics.bottlenecks) > 0:
            recommendations.append(
                f"Performance bottlenecks detected: {', '.join(metrics.bottlenecks)}. "
                "Consider optimizing these operations."
            )
        
        # CPU recommendations
        if metrics.average_cpu_percent < 50:
            recommendations.append(
                "Low CPU utilization. Consider enabling multi-threading or parallel processing."
            )
        
        return recommendations
    
    def _compare_with_benchmarks(
        self,
        metrics: PerformanceMetrics,
        benchmark_data: Dict[int, Dict[str, float]]
    ) -> Dict[str, Any]:
        """Compare performance with benchmark data."""
        
        comparison = {
            "performance_category": "unknown",
            "relative_speed": 1.0,
            "notes": []
        }
        
        # Simple comparison logic (would be more sophisticated in practice)
        if metrics.operations_per_second > 1000:
            comparison["performance_category"] = "excellent"
        elif metrics.operations_per_second > 100:
            comparison["performance_category"] = "good"
        elif metrics.operations_per_second > 10:
            comparison["performance_category"] = "acceptable"
        else:
            comparison["performance_category"] = "poor"
        
        return comparison
    
    def save_performance_report(
        self,
        report: Dict[str, Any],
        output_path: str = "results/performance_report.json"
    ) -> str:
        """Save performance report to file."""
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"Performance report saved to {output_file}")
        return str(output_file)


def run_performance_benchmark() -> Dict[str, Any]:
    """Run comprehensive performance benchmark."""
    
    profiler = UMIProfiler()
    reporter = PerformanceReporter()
    
    # Initialize UMI processor
    from .umi import UMIProcessor
    umi_processor = UMIProcessor()
    
    # Run benchmark
    benchmark_results = profiler.benchmark_family_sizes(umi_processor)
    
    # Generate report
    if benchmark_results:
        # Use largest benchmark for overall metrics
        largest_test = max(benchmark_results.keys())
        overall_metrics = PerformanceMetrics(
            total_runtime=benchmark_results[largest_test]["total_runtime"],
            peak_memory_mb=benchmark_results[largest_test]["peak_memory_mb"],
            average_cpu_percent=50.0,  # Mock value
            operations_per_second=benchmark_results[largest_test]["reads_per_second"],
            bottlenecks=[],
            detailed_timings={}
        )
        
        performance_report = reporter.generate_performance_report(
            overall_metrics, benchmark_results
        )
        
        # Add benchmark data to report
        performance_report["benchmark_results"] = benchmark_results
        
        # Save report
        report_path = reporter.save_performance_report(performance_report)
        
        return {
            "benchmark_results": benchmark_results,
            "performance_report": performance_report,
            "report_path": report_path
        }
    
    return {"error": "Benchmark failed to generate results"}


if __name__ == "__main__":
    # Run benchmark when called directly
    results = run_performance_benchmark()
    print(f"Benchmark completed. Results: {results.get('report_path', 'No report generated')}")
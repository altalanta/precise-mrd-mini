"""Performance monitoring and profiling for parallel operations and ML models."""

from __future__ import annotations

import functools
import time
import threading
from collections import defaultdict
from typing import Any, Dict, List, Optional, Callable
import psutil
import os

# Global performance monitoring state
_performance_monitor = None
_monitor_lock = threading.Lock()


class PerformanceMonitor:
    """Monitor performance metrics for parallel operations."""

    def __init__(self):
        """Initialize performance monitor."""
        self.timing_stats = defaultdict(lambda: {'calls': 0, 'total_time': 0.0, 'min_time': float('inf'), 'max_time': 0.0})
        self.memory_usage = []
        self.start_time = time.time()
        self._monitoring_active = True

    def start_monitoring(self):
        """Start performance monitoring."""
        self._monitoring_active = True
        self.start_time = time.time()

    def stop_monitoring(self):
        """Stop performance monitoring."""
        self._monitoring_active = False

    def record_timing(self, func_name: str, duration: float):
        """Record timing for a function."""
        if not self._monitoring_active:
            return

        stats = self.timing_stats[func_name]
        stats['calls'] += 1
        stats['total_time'] += duration
        stats['min_time'] = min(stats['min_time'], duration)
        stats['max_time'] = max(stats['max_time'], duration)

    def record_memory_usage(self):
        """Record current memory usage."""
        if not self._monitoring_active:
            return

        try:
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            self.memory_usage.append({
                'timestamp': time.time(),
                'rss_mb': memory_info.rss / 1024 / 1024,  # RSS in MB
                'vms_mb': memory_info.vms / 1024 / 1024,  # VMS in MB
            })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    def get_timing_stats(self) -> Dict[str, Dict[str, float]]:
        """Get timing statistics."""
        result = {}
        for func_name, stats in self.timing_stats.items():
            if stats['calls'] > 0:
                avg_time = stats['total_time'] / stats['calls']
                result[func_name] = {
                    'calls': stats['calls'],
                    'total_time': stats['total_time'],
                    'avg_time': avg_time,
                    'min_time': stats['min_time'],
                    'max_time': stats['max_time'],
                }
        return result

    def get_memory_stats(self) -> Dict[str, float]:
        """Get memory usage statistics."""
        if not self.memory_usage:
            return {}

        rss_values = [entry['rss_mb'] for entry in self.memory_usage]
        vms_values = [entry['vms_mb'] for entry in self.memory_usage]

        return {
            'peak_rss_mb': max(rss_values),
            'current_rss_mb': rss_values[-1] if rss_values else 0,
            'avg_rss_mb': sum(rss_values) / len(rss_values),
            'peak_vms_mb': max(vms_values),
            'current_vms_mb': vms_values[-1] if vms_values else 0,
            'avg_vms_mb': sum(vms_values) / len(vms_values),
        }

    def get_report(self) -> Dict[str, Any]:
        """Get complete performance report."""
        timing_stats = self.get_timing_stats()
        memory_stats = self.get_memory_stats()

        return {
            'timing_statistics': timing_stats,
            'memory_statistics': memory_stats,
            'total_functions_tracked': len(timing_stats),
            'monitoring_duration': time.time() - self.start_time,
            'peak_memory_mb': memory_stats.get('peak_rss_mb', 0),
        }


def get_performance_monitor() -> PerformanceMonitor:
    """Get the global performance monitor."""
    global _performance_monitor
    if _performance_monitor is None:
        with _monitor_lock:
            if _performance_monitor is None:
                _performance_monitor = PerformanceMonitor()
    return _performance_monitor


def reset_performance_monitor():
    """Reset the global performance monitor."""
    global _performance_monitor
    with _monitor_lock:
        _performance_monitor = PerformanceMonitor()


def timing_decorator(func: Callable) -> Callable:
    """Decorator to time function execution."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        monitor = get_performance_monitor()
        start_time = time.time()

        try:
            result = func(*args, **kwargs)
            return result
        finally:
            duration = time.time() - start_time
            monitor.record_timing(func.__name__, duration)
            monitor.record_memory_usage()

    return wrapper


def parallel_timing_decorator(func: Callable) -> Callable:
    """Decorator to time parallel function execution with additional metadata."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        monitor = get_performance_monitor()
        start_time = time.time()

        # Check if this is a parallel operation by looking for Dask or multiprocessing indicators
        is_parallel = any(
            kwarg in kwargs and (isinstance(kwargs[kwarg], (int, str)) and str(kwargs[kwarg]).isdigit())
            for kwarg in ['n_partitions', 'n_jobs', 'workers']
        ) or any(
            hasattr(arg, '__name__') and 'parallel' in arg.__name__.lower()
            for arg in args
        )

        try:
            result = func(*args, **kwargs)
            return result
        finally:
            duration = time.time() - start_time
            func_name = f"{func.__name__}_parallel" if is_parallel else func.__name__
            monitor.record_timing(func_name, duration)
            monitor.record_memory_usage()

    return wrapper


class MLPerformanceTracker:
    """Track ML model performance metrics."""

    def __init__(self):
        """Initialize ML performance tracker."""
        self.model_metrics = {}
        self.feature_importance = {}
        self.prediction_distributions = {}
        self.training_times = {}

    def record_model_metrics(self, model_name: str, metrics: Dict[str, float]):
        """Record ML model performance metrics.

        Args:
            model_name: Name of the ML model
            metrics: Dictionary of performance metrics
        """
        self.model_metrics[model_name] = metrics.copy()

    def record_feature_importance(self, model_name: str, importance: Dict[str, float]):
        """Record feature importance scores.

        Args:
            model_name: Name of the ML model
            importance: Dictionary mapping feature names to importance scores
        """
        self.feature_importance[model_name] = importance.copy()

    def record_prediction_distribution(self, model_name: str, predictions: np.ndarray):
        """Record prediction score distribution.

        Args:
            model_name: Name of the ML model
            predictions: Array of prediction scores
        """
        self.prediction_distributions[model_name] = {
            'mean': np.mean(predictions),
            'std': np.std(predictions),
            'min': np.min(predictions),
            'max': np.max(predictions),
            'median': np.median(predictions),
            'q25': np.percentile(predictions, 25),
            'q75': np.percentile(predictions, 75)
        }

    def record_training_time(self, model_name: str, training_time: float):
        """Record model training time.

        Args:
            model_name: Name of the ML model
            training_time: Training time in seconds
        """
        self.training_times[model_name] = training_time

    def get_ml_report(self) -> Dict[str, Any]:
        """Get comprehensive ML performance report.

        Returns:
            Dictionary with ML performance metrics
        """
        return {
            'model_metrics': self.model_metrics,
            'feature_importance': self.feature_importance,
            'prediction_distributions': self.prediction_distributions,
            'training_times': self.training_times,
            'n_models_tracked': len(self.model_metrics)
        }

    def compare_models(self) -> Dict[str, Any]:
        """Compare performance across different models.

        Returns:
            Dictionary with model comparison results
        """
        if not self.model_metrics:
            return {}

        comparison = {
            'best_model': None,
            'best_metric': None,
            'model_ranking': []
        }

        # Find best model by ROC AUC
        best_auc = 0.0
        best_model = None

        for model_name, metrics in self.model_metrics.items():
            auc = metrics.get('roc_auc', 0.0)
            if auc > best_auc:
                best_auc = auc
                best_model = model_name

        comparison['best_model'] = best_model
        comparison['best_metric'] = best_auc

        # Rank models by AUC
        model_ranking = sorted(
            [(name, metrics.get('roc_auc', 0.0)) for name, metrics in self.model_metrics.items()],
            key=lambda x: x[1],
            reverse=True
        )
        comparison['model_ranking'] = model_ranking

        return comparison


# Global ML performance tracker
_ml_tracker = None


def get_ml_performance_tracker() -> MLPerformanceTracker:
    """Get the global ML performance tracker."""
    global _ml_tracker
    if _ml_tracker is None:
        _ml_tracker = MLPerformanceTracker()
    return _ml_tracker


def reset_ml_performance_tracker():
    """Reset the global ML performance tracker."""
    global _ml_tracker
    _ml_tracker = MLPerformanceTracker()


def ml_performance_decorator(func: Callable) -> Callable:
    """Decorator to track ML model performance."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        tracker = get_ml_performance_tracker()
        start_time = time.time()

        try:
            result = func(*args, **kwargs)
            return result
        finally:
            training_time = time.time() - start_time
            model_name = getattr(func, '__name__', 'unknown_model')
            tracker.record_training_time(model_name, training_time)

    return wrapper


def get_performance_report() -> Dict[str, Any]:
    """Get current performance report."""
    monitor = get_performance_monitor()
    return monitor.get_report()

# Global ML performance tracker
_ml_tracker = None


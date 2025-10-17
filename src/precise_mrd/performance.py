"""Performance optimization and caching system for Precise MRD."""

from __future__ import annotations

import functools
import hashlib
import json
import pickle
import time
import weakref
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import psutil

from .config import PipelineConfig


class CacheStrategy(Enum):
    """Caching strategies for different data types and access patterns."""
    MEMORY = "memory"
    DISK = "disk"
    HYBRID = "hybrid"
    LRU = "lru"
    TTL = "ttl"


@dataclass
class CacheEntry:
    """Individual cache entry with metadata."""
    key: str
    value: Any
    strategy: CacheStrategy
    created_at: float
    accessed_at: float
    access_count: int = 0
    size_bytes: int = 0
    ttl_seconds: Optional[float] = None

    def is_expired(self) -> bool:
        """Check if TTL-based entry has expired."""
        if self.ttl_seconds is None:
            return False
        return (time.time() - self.created_at) > self.ttl_seconds

    def touch(self) -> None:
        """Update access metadata."""
        self.accessed_at = time.time()
        self.access_count += 1


class MemoryPool:
    """Memory pool for efficient array allocation."""

    def __init__(self, max_memory_mb: int = 1000):
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.current_memory_bytes = 0
        self.pools: Dict[str, List[np.ndarray]] = defaultdict(list)
        self._process = psutil.Process()

    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        memory_info = self._process.memory_info()
        return {
            'rss_mb': memory_info.rss / 1024 / 1024,
            'vms_mb': memory_info.vms / 1024 / 1024,
            'pool_mb': self.current_memory_bytes / 1024 / 1024,
            'available_mb': (self.max_memory_bytes - self.current_memory_bytes) / 1024 / 1024
        }

    def allocate_array(self, shape: Tuple[int, ...], dtype: np.dtype, strategy: str = 'reuse') -> np.ndarray:
        """Allocate array with memory pooling strategy."""
        key = f"{shape}_{dtype}"

        if strategy == 'reuse' and self.pools[key]:
            # Reuse existing array
            array = self.pools[key].pop()
            if array.shape == shape and array.dtype == dtype:
                self.current_memory_bytes += array.nbytes
                return array
            else:
                # Wrong shape/dtype, free and allocate new
                del array

        # Allocate new array
        array = np.empty(shape, dtype=dtype)
        self.current_memory_bytes += array.nbytes
        return array

    def return_array(self, array: np.ndarray) -> None:
        """Return array to pool for reuse."""
        if self.current_memory_bytes + array.nbytes > self.max_memory_bytes:
            # Pool is full, don't cache
            return

        key = f"{array.shape}_{array.dtype}"
        self.pools[key].append(array)
        self.current_memory_bytes += array.nbytes

    def cleanup(self, target_memory_mb: float = None) -> int:
        """Clean up memory pool to target size."""
        if target_memory_mb is None:
            target_memory_mb = self.max_memory_bytes * 0.7 / 1024 / 1024

        target_bytes = target_memory_mb * 1024 * 1024
        freed_bytes = 0

        # Remove oldest arrays first
        for key, arrays in self.pools.items():
            while arrays and self.current_memory_bytes > target_bytes:
                array = arrays.pop(0)
                self.current_memory_bytes -= array.nbytes
                freed_bytes += array.nbytes
                del array

        return freed_bytes


class IntelligentCache:
    """Multi-strategy intelligent caching system."""

    def __init__(self,
                 memory_limit_mb: int = 500,
                 disk_cache_dir: Optional[str] = None,
                 default_ttl: Optional[float] = None):
        self.memory_pool = MemoryPool(memory_limit_mb)
        self.disk_cache_dir = Path(disk_cache_dir) if disk_cache_dir else None
        self.default_ttl = default_ttl

        # In-memory cache with LRU strategy
        self.memory_cache: Dict[str, CacheEntry] = {}
        self.access_order: List[str] = []  # For LRU

        # Disk cache for persistence
        if self.disk_cache_dir:
            self.disk_cache_dir.mkdir(parents=True, exist_ok=True)

        # Statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'disk_hits': 0,
            'disk_misses': 0
        }

    def _compute_cache_key(self, func_name: str, args: Tuple, kwargs: Dict) -> str:
        """Compute deterministic cache key from function call."""
        # Create a deterministic string representation
        def serialize_obj(obj):
            """Safely serialize objects for cache keys."""
            if hasattr(obj, 'config_hash') and callable(obj.config_hash):
                # For config objects, use the hash method
                return f"config_hash_{obj.config_hash()}"
            elif hasattr(obj, '__dict__'):
                # For dataclass-like objects, use their dict representation
                return str(obj.__dict__)
            else:
                return str(obj)

        key_data = {
            'func': func_name,
            'args': [serialize_obj(arg) for arg in args],
            'kwargs': {k: serialize_obj(v) for k, v in kwargs.items() if k != 'rng'}  # Exclude RNG for determinism
        }
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.sha256(key_str.encode()).hexdigest()[:16]

    def _get_cache_strategy(self, value: Any, estimated_size: int) -> CacheStrategy:
        """Determine optimal caching strategy based on value characteristics."""
        if estimated_size > 100 * 1024 * 1024:  # > 100MB
            return CacheStrategy.DISK
        elif estimated_size > 10 * 1024 * 1024:  # > 10MB
            return CacheStrategy.HYBRID
        else:
            return CacheStrategy.MEMORY

    def _evict_lru(self, target_memory_mb: float = 100) -> None:
        """Evict least recently used entries to free memory."""
        target_bytes = target_memory_mb * 1024 * 1024

        # Sort by access time (oldest first)
        sorted_entries = sorted(
            self.memory_cache.items(),
            key=lambda x: x[1].accessed_at
        )

        for key, entry in sorted_entries:
            if self.memory_pool.current_memory_bytes <= target_bytes:
                break

            del self.memory_cache[key]
            if key in self.access_order:
                self.access_order.remove(key)
            self.stats['evictions'] += 1

            # Free memory pool resources
            if hasattr(entry.value, 'nbytes'):
                self.memory_pool.current_memory_bytes -= entry.size_bytes

    def get(self, key: str) -> Optional[Any]:
        """Retrieve value from cache."""
        # Check memory cache first
        if key in self.memory_cache:
            entry = self.memory_cache[key]
            if entry.is_expired():
                del self.memory_cache[key]
                if key in self.access_order:
                    self.access_order.remove(key)
                return None

            entry.touch()
            # Move to end of access order (most recently used)
            if key in self.access_order:
                self.access_order.remove(key)
            self.access_order.append(key)

            self.stats['hits'] += 1
            return entry.value

        # Check disk cache
        if self.disk_cache_dir:
            disk_path = self.disk_cache_dir / f"{key}.pkl"
            if disk_path.exists():
                try:
                    with open(disk_path, 'rb') as f:
                        value = pickle.load(f)
                    self.stats['disk_hits'] += 1
                    return value
                except Exception:
                    # Corrupted cache file
                    disk_path.unlink(missing_ok=True)

        self.stats['misses'] += 1
        return None

    def put(self, key: str, value: Any, strategy: Optional[CacheStrategy] = None, ttl: Optional[float] = None) -> None:
        """Store value in cache with appropriate strategy."""
        # Estimate size
        if hasattr(value, 'nbytes'):
            size_bytes = value.nbytes
        elif hasattr(value, '__sizeof__'):
            size_bytes = value.__sizeof__()
        else:
            size_bytes = 1024  # Default estimate

        if strategy is None:
            strategy = self._get_cache_strategy(value, size_bytes)

        entry = CacheEntry(
            key=key,
            value=value,
            strategy=strategy,
            created_at=time.time(),
            accessed_at=time.time(),
            size_bytes=size_bytes,
            ttl_seconds=ttl or self.default_ttl
        )

        if strategy == CacheStrategy.MEMORY:
            # Check if we need to evict
            if self.memory_pool.current_memory_bytes + size_bytes > self.memory_pool.max_memory_bytes:
                self._evict_lru()

            self.memory_cache[key] = entry
            self.access_order.append(key)

        elif strategy == CacheStrategy.DISK and self.disk_cache_dir:
            # Store on disk
            disk_path = self.disk_cache_dir / f"{key}.pkl"
            try:
                with open(disk_path, 'wb') as f:
                    pickle.dump(value, f)
            except Exception:
                # Disk write failed, skip caching
                pass

        elif strategy == CacheStrategy.HYBRID:
            # Store in both memory and disk
            self.put(key, value, CacheStrategy.MEMORY)
            self.put(key, value, CacheStrategy.DISK)

    def clear(self) -> None:
        """Clear all caches."""
        self.memory_cache.clear()
        self.access_order.clear()
        if self.disk_cache_dir:
            for file in self.disk_cache_dir.glob("*.pkl"):
                file.unlink(missing_ok=True)

    def get_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        memory_usage = self.memory_pool.get_memory_usage()
        return {
            **self.stats,
            'memory_entries': len(self.memory_cache),
            'memory_usage_mb': memory_usage['pool_mb'],
            'disk_entries': len(list(self.disk_cache_dir.glob("*.pkl"))) if self.disk_cache_dir else 0,
            'hit_rate': self.stats['hits'] / (self.stats['hits'] + self.stats['misses']) if (self.stats['hits'] + self.stats['misses']) > 0 else 0,
            'disk_hit_rate': self.stats['disk_hits'] / (self.stats['disk_hits'] + self.stats['disk_misses']) if (self.stats['disk_hits'] + self.stats['disk_misses']) > 0 else 0
        }


def cache_result(strategy: CacheStrategy = CacheStrategy.MEMORY, ttl: Optional[float] = None):
    """Decorator for intelligent caching of function results."""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Get the cache instance from the function's context
            cache = getattr(func, '_cache_instance', None)
            if cache is None:
                # Create a default cache if none exists
                cache = IntelligentCache()
                func._cache_instance = cache

            # Compute cache key
            key = cache._compute_cache_key(func.__name__, args, kwargs)

            # Try to get from cache
            cached_result = cache.get(key)
            if cached_result is not None:
                return cached_result

            # Compute result
            result = func(*args, **kwargs)

            # Cache result
            cache.put(key, result, strategy, ttl)

            return result

        return wrapper

    return decorator


class ChunkedProcessor:
    """Process large datasets in chunks for memory efficiency."""

    def __init__(self, chunk_size: int = 10000, max_memory_mb: int = 1000):
        self.chunk_size = chunk_size
        self.memory_pool = MemoryPool(max_memory_mb)

    def process_dataframe_chunks(self,
                                df: pd.DataFrame,
                                process_func: Callable[[pd.DataFrame], pd.DataFrame],
                                combine_func: Optional[Callable[[List[pd.DataFrame]], pd.DataFrame]] = None) -> pd.DataFrame:
        """Process DataFrame in chunks with memory management."""
        results = []

        for i in range(0, len(df), self.chunk_size):
            chunk = df.iloc[i:i + self.chunk_size].copy()

            # Process chunk
            processed_chunk = process_func(chunk)

            # Use memory pool for intermediate results if needed
            if len(results) > 10:  # Keep only recent results in memory
                # Combine intermediate results to free memory
                if combine_func:
                    combined = combine_func(results[-5:])  # Keep last 5
                    results = results[:5] + [combined]

            results.append(processed_chunk)

            # Periodic memory cleanup
            if i % (self.chunk_size * 10) == 0:
                freed = self.memory_pool.cleanup()
                if freed > 0:
                    print(f"  Freed {freed / 1024 / 1024:.1f} MB from memory pool")

        # Final combination
        if combine_func:
            return combine_func(results)
        else:
            return pd.concat(results, ignore_index=True)


class PerformanceMonitor:
    """Monitor and profile pipeline performance."""

    def __init__(self):
        self.timers: Dict[str, float] = {}
        self.memory_snapshots: List[Dict[str, float]] = []
        self.function_calls: Dict[str, List[float]] = defaultdict(list)

    def start_timer(self, name: str) -> None:
        """Start timing a code section."""
        self.timers[name] = time.time()

    def end_timer(self, name: str) -> float:
        """End timing and return elapsed time."""
        if name in self.timers:
            elapsed = time.time() - self.timers[name]
            self.function_calls[name].append(elapsed)
            del self.timers[name]
            return elapsed
        return 0.0

    def record_memory_usage(self) -> None:
        """Record current memory usage."""
        process = psutil.Process()
        memory_info = process.memory_info()
        self.memory_snapshots.append({
            'timestamp': time.time(),
            'rss_mb': memory_info.rss / 1024 / 1024,
            'vms_mb': memory_info.vms / 1024 / 1024,
        })

    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        # Function timing statistics
        timing_stats = {}
        for func_name, times in self.function_calls.items():
            timing_stats[func_name] = {
                'calls': len(times),
                'total_time': sum(times),
                'avg_time': np.mean(times),
                'min_time': np.min(times),
                'max_time': np.max(times),
                'std_time': np.std(times)
            }

        # Memory usage over time
        memory_over_time = {
            'timestamps': [s['timestamp'] for s in self.memory_snapshots],
            'rss_mb': [s['rss_mb'] for s in self.memory_snapshots],
            'vms_mb': [s['vms_mb'] for s in self.memory_snapshots]
        }

        return {
            'timing_statistics': timing_stats,
            'memory_usage_over_time': memory_over_time,
            'peak_memory_mb': max(s['rss_mb'] for s in self.memory_snapshots) if self.memory_snapshots else 0,
            'total_functions_tracked': len(self.function_calls)
        }


# Global performance monitor instance
_performance_monitor = PerformanceMonitor()


def profile_function(func: Callable) -> Callable:
    """Decorator to profile function execution time and memory usage."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        func_name = f"{func.__module__}.{func.__name__}"

        _performance_monitor.start_timer(func_name)
        _performance_monitor.record_memory_usage()

        try:
            result = func(*args, **kwargs)
            return result
        finally:
            elapsed = _performance_monitor.end_timer(func_name)
            _performance_monitor.record_memory_usage()

            # Log slow operations
            if elapsed > 1.0:  # More than 1 second
                print(f"  SLOW: {func_name} took {elapsed:.2f}s")

    return wrapper


def get_performance_report() -> Dict[str, Any]:
    """Get the current performance monitoring report."""
    return _performance_monitor.get_performance_report()


def reset_performance_monitor() -> None:
    """Reset all performance monitoring data."""
    global _performance_monitor
    _performance_monitor = PerformanceMonitor()

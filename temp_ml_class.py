class IntelligentCache:
    """Intelligent caching strategy for performance optimization."""

    def __init__(self, cache_dir: str = None, strategy: str = "lru"):
        """Initialize intelligent cache.

        Args:
            cache_dir: Directory for cache storage
            strategy: Caching strategy (lru, lfu, adaptive)
        """
        self.strategy = strategy
        self.cache_dir = cache_dir

    def should_cache(self, func_name: str, args_hash: str) -> bool:
        """Determine if a function result should be cached."""
        return True  # Simple implementation

    def get_cache_key(self, func_name: str, args_hash: str) -> str:
        """Generate cache key."""
        return f"{func_name}_{args_hash}"


class CacheStrategy:
    """Cache strategy configurations."""

    LRU = "lru"
    LFU = "lfu"
    ADAPTIVE = "adaptive"


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
            model_name = getattr(func, "__name__", "unknown_model")
            tracker.record_training_time(model_name, training_time)

    return wrapper


def get_performance_report() -> Dict[str, Any]:
    """Get current performance report."""
    monitor = get_performance_monitor()
    return monitor.get_report()



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


# Global ML performance tracker
_ml_tracker = None


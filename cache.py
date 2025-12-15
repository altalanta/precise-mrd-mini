"""Caching layer for intermediate results to avoid recomputation."""

from __future__ import annotations

import hashlib
import json
import pickle
import time
from pathlib import Path
from typing import Any

from .config import PipelineConfig


class PipelineCache:
    """Cache for pipeline intermediate results."""

    def __init__(
        self,
        cache_dir: str | Path,
        enabled: bool = True,
        ttl_seconds: int = 86400,
    ):
        """Initialize cache.

        Args:
            cache_dir: Directory to store cached results
            enabled: Whether caching is enabled
            ttl_seconds: Time-to-live for cached results (default: 24 hours)
        """
        self.cache_dir = Path(cache_dir)
        self.enabled = enabled
        self.ttl_seconds = ttl_seconds

        if enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._metadata_file = self.cache_dir / "cache_metadata.json"
            self._load_metadata()

    def _load_metadata(self):
        """Load cache metadata."""
        if self._metadata_file.exists():
            try:
                with open(self._metadata_file) as f:
                    self._metadata = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                self._metadata = {}
        else:
            self._metadata = {}

    def _save_metadata(self):
        """Save cache metadata."""
        if self.enabled:
            with open(self._metadata_file, "w") as f:
                json.dump(self._metadata, f, indent=2)

    def _get_cache_key(self, func_name: str, args_hash: str, config_hash: str) -> str:
        """Generate cache key for a function call."""
        return hashlib.sha256(
            f"{func_name}:{args_hash}:{config_hash}".encode(),
        ).hexdigest()[:16]

    def _get_config_hash(self, config: PipelineConfig) -> str:
        """Get hash of pipeline configuration."""
        if config is None:
            return "no_config"
        return config.config_hash()

    def _get_args_hash(self, args: tuple, kwargs: dict) -> str:
        """Generate hash of function arguments."""
        # Convert args and kwargs to a hashable representation
        args_str = json.dumps(args, sort_keys=True, default=str)
        kwargs_str = json.dumps(kwargs, sort_keys=True, default=str)
        combined = f"{args_str}:{kwargs_str}"
        return hashlib.sha256(combined.encode()).hexdigest()[:16]

    def _is_expired(self, cache_key: str) -> bool:
        """Check if cached result is expired."""
        if cache_key not in self._metadata:
            return True

        metadata = self._metadata[cache_key]
        cache_time = metadata.get("timestamp", 0)
        return (time.time() - cache_time) > self.ttl_seconds

    def get(
        self,
        func_name: str,
        config: PipelineConfig,
        args: tuple = (),
        kwargs: dict = None,
    ) -> Any:
        """Get cached result if available and not expired.

        Args:
            func_name: Name of the function
            config: Pipeline configuration
            args: Function positional arguments
            kwargs: Function keyword arguments

        Returns:
            Cached result or None if not found/expired
        """
        if not self.enabled:
            return None

        kwargs = kwargs or {}
        config_hash = self._get_config_hash(config)
        args_hash = self._get_args_hash(args, kwargs)
        cache_key = self._get_cache_key(func_name, args_hash, config_hash)

        if self._is_expired(cache_key):
            return None

        cache_file = self.cache_dir / f"{cache_key}.pkl"
        if not cache_file.exists():
            return None

        try:
            with open(cache_file, "rb") as f:
                return pickle.load(f)
        except (pickle.PickleError, FileNotFoundError, EOFError):
            return None

    def put(
        self,
        func_name: str,
        config: PipelineConfig,
        args: tuple = (),
        kwargs: dict = None,
        result: Any = None,
    ):
        """Store result in cache.

        Args:
            func_name: Name of the function
            config: Pipeline configuration
            args: Function positional arguments
            kwargs: Function keyword arguments
            result: Result to cache
        """
        if not self.enabled:
            return

        kwargs = kwargs or {}
        config_hash = self._get_config_hash(config)
        args_hash = self._get_args_hash(args, kwargs)
        cache_key = self._get_cache_key(func_name, args_hash, config_hash)

        # Update metadata
        self._metadata[cache_key] = {
            "func_name": func_name,
            "timestamp": time.time(),
            "config_hash": config_hash,
            "args_hash": args_hash,
        }

        # Save result
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        try:
            with open(cache_file, "wb") as f:
                pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
        except (pickle.PickleError, OSError):
            # Remove from metadata if saving failed
            self._metadata.pop(cache_key, None)
            return

        self._save_metadata()

    def clear(self):
        """Clear all cached results."""
        if not self.enabled:
            return

        # Remove all cache files
        for cache_file in self.cache_dir.glob("*.pkl"):
            try:
                cache_file.unlink()
            except OSError:
                pass

        # Clear metadata
        self._metadata = {}
        self._save_metadata()

    def cleanup_expired(self):
        """Remove expired cache entries."""
        if not self.enabled:
            return

        expired_keys = []
        for cache_key, _metadata in self._metadata.items():
            if self._is_expired(cache_key):
                expired_keys.append(cache_key)

        for cache_key in expired_keys:
            # Remove cache file
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            try:
                cache_file.unlink()
            except OSError:
                pass

            # Remove from metadata
            del self._metadata[cache_key]

        if expired_keys:
            self._save_metadata()


def cached_pipeline_step(cache: PipelineCache = None):
    """Decorator to cache pipeline step results.

    Args:
        cache: PipelineCache instance to use. If None, no caching is performed.

    Returns:
        Decorator function
    """

    def decorator(func):
        def wrapper(config: PipelineConfig, *args, **kwargs):
            if cache is None:
                return func(config, *args, **kwargs)

            # Get cache key components
            func_name = func.__name__

            # Try to get from cache first
            cached_result = cache.get(func_name, config, args, kwargs)
            if cached_result is not None:
                print(f"  Using cached result for {func_name}")
                return cached_result

            # Compute result
            result = func(config, *args, **kwargs)

            # Cache result
            cache.put(func_name, config, args, kwargs, result)

            return result

        return wrapper

    return decorator

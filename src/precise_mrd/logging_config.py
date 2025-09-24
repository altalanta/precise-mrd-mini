"""
Logging configuration for the precise-mrd pipeline.

Provides structured logging with performance monitoring and
comprehensive error tracking capabilities.
"""

import functools
import logging
import time
from pathlib import Path
from typing import Optional, Callable, Any
import sys


class PerformanceLogger:
    """Context manager for performance monitoring."""
    
    def __init__(self, logger: logging.Logger, operation: str):
        self.logger = logger
        self.operation = operation
        self.start_time: Optional[float] = None
        
    def __enter__(self):
        self.start_time = time.perf_counter()
        self.logger.info(f"Starting {self.operation}")
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time is not None:
            duration = time.perf_counter() - self.start_time
            if exc_type is None:
                self.logger.info(f"Completed {self.operation} in {duration:.3f}s")
            else:
                self.logger.error(f"Failed {self.operation} after {duration:.3f}s: {exc_val}")
        return False


def time_it(operation: str = None):
    """Decorator for automatic performance logging."""
    def decorator(func: Callable) -> Callable:
        op_name = operation or f"{func.__module__}.{func.__name__}"
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger = logging.getLogger(func.__module__)
            with PerformanceLogger(logger, op_name):
                return func(*args, **kwargs)
        return wrapper
    return decorator


def setup_logging(
    level: str = "INFO",
    log_file: Optional[Path] = None,
    console_output: bool = True,
    format_string: Optional[str] = None
) -> logging.Logger:
    """Setup comprehensive logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file
        console_output: Whether to output to console
        format_string: Custom format string
        
    Returns:
        Configured logger instance
    """
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Convert string level to logging constant
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # Create logger
    logger = logging.getLogger("precise_mrd")
    logger.setLevel(numeric_level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(format_string)
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Prevent duplicate messages
    logger.propagate = False
    
    return logger


def log_system_info(logger: logging.Logger) -> None:
    """Log system information for reproducibility."""
    import platform
    import sys
    
    logger.info("System Information:")
    logger.info(f"  Platform: {platform.platform()}")
    logger.info(f"  Python: {sys.version}")
    logger.info(f"  Architecture: {platform.architecture()[0]}")
    
    # Log package versions
    try:
        import numpy
        import pandas
        import scipy
        
        logger.info("Package Versions:")
        logger.info(f"  NumPy: {numpy.__version__}")
        logger.info(f"  Pandas: {pandas.__version__}")
        logger.info(f"  SciPy: {scipy.__version__}")
    except ImportError:
        logger.warning("Could not determine package versions")


def get_logger(name: str = "precise_mrd") -> logging.Logger:
    """Get a logger instance with default configuration."""
    return logging.getLogger(name)


class LoggingContext:
    """Context manager for temporary logging configuration."""
    
    def __init__(self, logger: logging.Logger, level: int):
        self.logger = logger
        self.new_level = level
        self.old_level = logger.level
        
    def __enter__(self):
        self.logger.setLevel(self.new_level)
        return self.logger
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.setLevel(self.old_level)


def suppress_logging(logger: logging.Logger):
    """Context manager to temporarily suppress logging."""
    return LoggingContext(logger, logging.CRITICAL + 1)


def debug_logging(logger: logging.Logger):
    """Context manager to temporarily enable debug logging."""
    return LoggingContext(logger, logging.DEBUG)
"""Logging configuration for the Precise MRD pipeline.

This module provides structured logging using structlog, which outputs
JSON-formatted logs suitable for production monitoring and log aggregation
systems like ELK, Splunk, or CloudWatch.

Example:
    >>> from precise_mrd.logging_config import setup_logging, get_logger
    >>> setup_logging("DEBUG")
    >>> log = get_logger(__name__)
    >>> log.info("Pipeline started", run_id="test_001", seed=42)
"""

import logging
import sys

import structlog

from .settings import settings


def setup_logging(log_level: str | None = None) -> None:
    """Configure structured logging for the application.

    Sets up both Python's standard logging and structlog with JSON output,
    ISO timestamps, and automatic exception formatting.

    Args:
        log_level: The minimum log level to capture. One of "DEBUG", "INFO",
            "WARNING", "ERROR", or "CRITICAL". If None, uses the LOG_LEVEL
            from application settings (defaults to "INFO").

    Note:
        This function should be called once at application startup, typically
        in the FastAPI lifespan handler or CLI entry point.
    """
    # Use log level from settings if not explicitly provided
    level = log_level or settings.LOG_LEVEL

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=level.upper(),
    )

    # Configure structlog for JSON output with rich context
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a configured logger for a specific module.

    Args:
        name: The logger name, typically __name__ of the calling module.

    Returns:
        A bound structlog logger that outputs JSON with automatic context.

    Example:
        >>> log = get_logger(__name__)
        >>> log.info("Processing sample", sample_id="S001", depth=10000)
        {"event": "Processing sample", "sample_id": "S001", "depth": 10000, ...}
    """
    return structlog.get_logger(name)

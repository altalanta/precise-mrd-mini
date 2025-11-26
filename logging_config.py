"""Logging configuration for the Precise MRD pipeline."""

import logging
import sys

import structlog

from .settings import settings


def setup_logging(log_level: str = None):
    """Configure structured logging for the application."""

    # Use log level from settings if not explicitly provided
    level = log_level or settings.LOG_LEVEL

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=level.upper(),
    )

    # Configure structlog
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
    """Get a configured logger for a specific module."""
    return structlog.get_logger(name)

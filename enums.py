"""Type-safe enumerations for the Precise MRD pipeline.

This module provides StrEnum classes for constrained string values throughout
the application, enabling type safety, IDE autocompletion, and preventing
typos in status strings and configuration values.

Example:
    >>> from precise_mrd.enums import JobStatusEnum
    >>> status = JobStatusEnum.RUNNING
    >>> print(status)  # "running"
    >>> status == "running"  # True (StrEnum allows string comparison)
"""

from enum import StrEnum
from typing import Literal


class JobStatusEnum(StrEnum):
    """Enumeration of possible job statuses.

    Using StrEnum allows these values to be used directly as strings
    while still providing type safety and IDE autocompletion.

    Attributes:
        PENDING: Job created but not yet queued
        QUEUED: Job is in the Celery queue
        RUNNING: Job is currently executing
        COMPLETED: Job finished successfully
        FAILED: Job encountered an error
        CANCELLED: Job was manually cancelled
    """

    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class HealthStatusEnum(StrEnum):
    """Enumeration of possible health check statuses.

    Attributes:
        OK: Service is healthy and functioning normally
        ERROR: Service is unavailable or experiencing critical issues
        DEGRADED: Service is available but with reduced performance
    """

    OK = "ok"
    ERROR = "error"
    DEGRADED = "degraded"


# Type aliases for better documentation and type checking
# These provide IDE autocompletion for valid values

MLModelType = Literal["ensemble", "random_forest", "xgboost", "lightgbm"]
"""Supported machine learning model types for variant calling."""

DLModelType = Literal["cnn_lstm", "transformer", "attention"]
"""Supported deep learning model architectures."""

StatisticalTestType = Literal["poisson", "binomial", "fisher"]
"""Statistical tests available for variant calling."""

FDRMethod = Literal["benjamini_hochberg", "bonferroni", "holm"]
"""False discovery rate correction methods."""


__all__ = [
    "JobStatusEnum",
    "HealthStatusEnum",
    "MLModelType",
    "DLModelType",
    "StatisticalTestType",
    "FDRMethod",
]



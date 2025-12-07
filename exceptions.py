"""Custom exceptions for the Precise MRD pipeline.

This module provides a structured exception hierarchy with error codes for
better debugging, logging, and programmatic error handling.

Each exception includes:
- An error code for programmatic identification
- A human-readable message
- Optional context dictionary for additional details

Example:
    >>> raise ConfigurationError(
    ...     "Invalid allele fraction",
    ...     context={"value": -0.5, "field": "allele_fractions"}
    ... )
    ConfigurationError: [PMRD-CFG-001] Invalid allele fraction
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any


class ErrorCode(StrEnum):
    """Error codes for Precise MRD exceptions.

    Format: PMRD-{CATEGORY}-{NUMBER}
    Categories:
        - GEN: General errors
        - CFG: Configuration errors
        - DAT: Data processing errors
        - ART: Artifact validation errors
        - FIO: File I/O errors
        - JOB: Job management errors
        - API: API-related errors
    """

    # General errors (000-099)
    UNKNOWN = "PMRD-GEN-000"
    INTERNAL = "PMRD-GEN-001"

    # Configuration errors (100-199)
    CONFIG_INVALID = "PMRD-CFG-001"
    CONFIG_MISSING = "PMRD-CFG-002"
    CONFIG_PARSE = "PMRD-CFG-003"
    CONFIG_VERSION = "PMRD-CFG-004"

    # Data processing errors (200-299)
    DATA_INVALID = "PMRD-DAT-001"
    DATA_MISSING = "PMRD-DAT-002"
    PIPELINE_STAGE = "PMRD-DAT-003"
    SCHEMA_VALIDATION = "PMRD-DAT-004"

    # Artifact validation errors (300-399)
    ARTIFACT_INVALID = "PMRD-ART-001"
    ARTIFACT_MISSING = "PMRD-ART-002"
    ARTIFACT_HASH_MISMATCH = "PMRD-ART-003"

    # File I/O errors (400-499)
    FILE_NOT_FOUND = "PMRD-FIO-001"
    FILE_READ = "PMRD-FIO-002"
    FILE_WRITE = "PMRD-FIO-003"
    FILE_PERMISSION = "PMRD-FIO-004"

    # Job management errors (500-599)
    JOB_NOT_FOUND = "PMRD-JOB-001"
    JOB_ALREADY_EXISTS = "PMRD-JOB-002"
    JOB_STATE_INVALID = "PMRD-JOB-003"

    # API errors (600-699)
    API_VALIDATION = "PMRD-API-001"
    API_RATE_LIMIT = "PMRD-API-002"
    API_TIMEOUT = "PMRD-API-003"


class PreciseMRDError(Exception):
    """Base exception for all application-specific errors.

    Attributes:
        message: Human-readable error message
        error_code: Structured error code for programmatic handling
        context: Optional dictionary with additional error context
    """

    default_error_code: ErrorCode = ErrorCode.UNKNOWN

    def __init__(
        self,
        message: str,
        error_code: ErrorCode | None = None,
        context: dict[str, Any] | None = None,
    ):
        self.message = message
        self.error_code = error_code or self.default_error_code
        self.context = context or {}
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        """Format the exception message with error code."""
        return f"[{self.error_code}] {self.message}"

    def to_dict(self) -> dict[str, Any]:
        """Convert exception to dictionary for JSON serialization."""
        return {
            "error_code": str(self.error_code),
            "message": self.message,
            "context": self.context,
        }


class ConfigurationError(PreciseMRDError):
    """Raised for errors related to configuration loading, validation, or parsing.

    Example:
        >>> raise ConfigurationError(
        ...     "Missing required field 'seed'",
        ...     error_code=ErrorCode.CONFIG_MISSING,
        ...     context={"field": "seed", "config_path": "/path/to/config.yaml"}
        ... )
    """

    default_error_code = ErrorCode.CONFIG_INVALID


class DataProcessingError(PreciseMRDError):
    """Raised for errors that occur during a pipeline stage.

    Example:
        >>> raise DataProcessingError(
        ...     "UMI collapse failed due to insufficient reads",
        ...     error_code=ErrorCode.PIPELINE_STAGE,
        ...     context={"stage": "collapse_umis", "n_reads": 5}
        ... )
    """

    default_error_code = ErrorCode.PIPELINE_STAGE


class ArtifactValidationError(PreciseMRDError):
    """Raised when a generated artifact does not match its contract.

    Example:
        >>> raise ArtifactValidationError(
        ...     "Hash mismatch for metrics.json",
        ...     error_code=ErrorCode.ARTIFACT_HASH_MISMATCH,
        ...     context={"expected": "abc123", "actual": "def456"}
        ... )
    """

    default_error_code = ErrorCode.ARTIFACT_INVALID


class FileOperationError(PreciseMRDError):
    """Raised for errors related to file I/O.

    Example:
        >>> raise FileOperationError(
        ...     "Cannot write to output directory",
        ...     error_code=ErrorCode.FILE_PERMISSION,
        ...     context={"path": "/protected/dir", "operation": "write"}
        ... )
    """

    default_error_code = ErrorCode.FILE_READ


class JobError(PreciseMRDError):
    """Raised for errors related to job management.

    Example:
        >>> raise JobError(
        ...     "Job with ID 'abc-123' not found",
        ...     error_code=ErrorCode.JOB_NOT_FOUND,
        ...     context={"job_id": "abc-123"}
        ... )
    """

    default_error_code = ErrorCode.JOB_NOT_FOUND


class APIError(PreciseMRDError):
    """Raised for API-specific errors.

    Example:
        >>> raise APIError(
        ...     "Invalid request payload",
        ...     error_code=ErrorCode.API_VALIDATION,
        ...     context={"field": "run_id", "error": "cannot be empty"}
        ... )
    """

    default_error_code = ErrorCode.API_VALIDATION


__all__ = [
    "ErrorCode",
    "PreciseMRDError",
    "ConfigurationError",
    "DataProcessingError",
    "ArtifactValidationError",
    "FileOperationError",
    "JobError",
    "APIError",
]

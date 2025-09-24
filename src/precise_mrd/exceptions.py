"""
Custom exceptions for the precise-mrd pipeline.

This module provides specific exception types for better error handling
and debugging throughout the pipeline.
"""


class PreciseMRDError(Exception):
    """Base exception for precise-mrd pipeline errors."""
    
    def __init__(self, message: str, details: dict = None):
        super().__init__(message)
        self.details = details or {}


class ValidationError(PreciseMRDError):
    """Raised when input validation fails."""
    pass


class ProcessingError(PreciseMRDError):
    """Raised when data processing fails."""
    pass


class StatisticalError(PreciseMRDError):
    """Raised when statistical analysis fails."""
    pass


class ConfigurationError(PreciseMRDError):
    """Raised when configuration is invalid."""
    pass


class FileFormatError(PreciseMRDError):
    """Raised when file format is invalid."""
    pass


class InsufficientDataError(PreciseMRDError):
    """Raised when there is insufficient data for analysis."""
    pass


class QualityControlError(PreciseMRDError):
    """Raised when quality control checks fail."""
    pass


class UMIProcessingError(ProcessingError):
    """Raised when UMI processing fails."""
    pass


class ContextAnalysisError(ProcessingError):
    """Raised when context analysis fails."""
    pass


class LODEstimationError(StatisticalError):
    """Raised when LoD estimation fails."""
    pass
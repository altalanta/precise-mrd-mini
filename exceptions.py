"""Custom exceptions for the Precise MRD pipeline."""

class PreciseMRDError(Exception):
    """Base exception for all application-specific errors."""
    pass

class ConfigurationError(PreciseMRDError):
    """Raised for errors related to configuration loading, validation, or parsing."""
    pass

class DataProcessingError(PreciseMRDError):
    """Raised for errors that occur during a pipeline stage."""
    pass

class ArtifactValidationError(PreciseMRDError):
    """Raised when a generated artifact does not match its contract."""
    pass

class FileOperationError(PreciseMRDError):
    """Raised for errors related to file I/O."""
    pass





"""
Tests for custom exceptions and error handling.
"""

import pytest
from precise_mrd.exceptions import (
    PreciseMRDError,
    ValidationError,
    ProcessingError,
    StatisticalError,
    ConfigurationError,
    UMIProcessingError,
    LODEstimationError
)


class TestCustomExceptions:
    """Test custom exception hierarchy."""
    
    def test_base_exception(self):
        """Test base PreciseMRDError."""
        error = PreciseMRDError("Base error")
        assert str(error) == "Base error"
        assert error.details == {}
        
        # With details
        details = {"code": "E001", "context": "test"}
        error_with_details = PreciseMRDError("Error with details", details)
        assert error_with_details.details == details
        
    def test_validation_error(self):
        """Test ValidationError."""
        error = ValidationError("Invalid input")
        assert isinstance(error, PreciseMRDError)
        assert str(error) == "Invalid input"
        
    def test_processing_error(self):
        """Test ProcessingError."""
        error = ProcessingError("Processing failed")
        assert isinstance(error, PreciseMRDError)
        assert str(error) == "Processing failed"
        
    def test_statistical_error(self):
        """Test StatisticalError."""
        error = StatisticalError("Statistical test failed")
        assert isinstance(error, PreciseMRDError)
        assert str(error) == "Statistical test failed"
        
    def test_configuration_error(self):
        """Test ConfigurationError."""
        error = ConfigurationError("Invalid config")
        assert isinstance(error, PreciseMRDError)
        assert str(error) == "Invalid config"
        
    def test_umi_processing_error(self):
        """Test UMIProcessingError."""
        error = UMIProcessingError("UMI processing failed")
        assert isinstance(error, ProcessingError)
        assert isinstance(error, PreciseMRDError)
        assert str(error) == "UMI processing failed"
        
    def test_lod_estimation_error(self):
        """Test LODEstimationError."""
        error = LODEstimationError("LoD estimation failed")
        assert isinstance(error, StatisticalError)
        assert isinstance(error, PreciseMRDError)
        assert str(error) == "LoD estimation failed"
        
    def test_error_chaining(self):
        """Test exception chaining."""
        try:
            raise ValueError("Original error")
        except ValueError as e:
            processing_error = ProcessingError("Processing failed", {"original": str(e)})
            assert "original" in processing_error.details
            assert processing_error.details["original"] == "Original error"
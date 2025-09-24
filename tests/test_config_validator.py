"""
Tests for configuration validation.
"""

import pytest
from precise_mrd.config_validator import ConfigValidator
from precise_mrd.exceptions import ConfigurationError


class TestConfigValidator:
    """Test configuration validation."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.validator = ConfigValidator()
        self.valid_config = {
            'run_id': 'test_run',
            'seed': 42,
            'simulation': {
                'allele_fractions': [0.05, 0.01, 0.001],
                'umi_depths': [5000, 10000, 20000],
                'n_replicates': 1000,
                'n_bootstrap': 1000
            },
            'umi': {
                'min_family_size': 3,
                'max_family_size': 1000,
                'quality_threshold': 20,
                'consensus_threshold': 0.6
            },
            'stats': {
                'test_type': 'poisson',
                'alpha': 0.05,
                'fdr_method': 'benjamini_hochberg'
            },
            'lod': {
                'detection_threshold': 0.95,
                'confidence_level': 0.95
            }
        }
    
    def test_valid_config(self):
        """Test validation of a valid configuration."""
        is_valid, errors, warnings = self.validator.validate_config(self.valid_config)
        assert is_valid
        assert len(errors) == 0
        
    def test_missing_required_keys(self):
        """Test validation with missing required keys."""
        config = {'run_id': 'test'}  # Missing most required keys
        is_valid, errors, warnings = self.validator.validate_config(config)
        assert not is_valid
        assert any('Missing required configuration key' in error for error in errors)
        
    def test_invalid_allele_fractions(self):
        """Test validation of invalid allele fractions."""
        config = self.valid_config.copy()
        
        # Test non-list
        config['simulation']['allele_fractions'] = 0.05
        is_valid, errors, warnings = self.validator.validate_config(config)
        assert not is_valid
        assert any('must be a list' in error for error in errors)
        
        # Test invalid values
        config['simulation']['allele_fractions'] = [1.5, -0.1]  # Out of range
        is_valid, errors, warnings = self.validator.validate_config(config)
        assert not is_valid
        assert len([e for e in errors if 'must be between 0 and 1' in e]) == 2
        
    def test_invalid_umi_config(self):
        """Test validation of invalid UMI configuration."""
        config = self.valid_config.copy()
        
        # Test invalid family sizes
        config['umi']['min_family_size'] = 0
        is_valid, errors, warnings = self.validator.validate_config(config)
        assert not is_valid
        assert any('must be at least 1' in error for error in errors)
        
        # Test inconsistent sizes
        config['umi']['min_family_size'] = 10
        config['umi']['max_family_size'] = 5
        is_valid, errors, warnings = self.validator.validate_config(config)
        assert not is_valid
        assert any('cannot exceed' in error for error in errors)
        
    def test_invalid_stats_config(self):
        """Test validation of invalid statistics configuration."""
        config = self.valid_config.copy()
        
        # Test invalid test type
        config['stats']['test_type'] = 'invalid'
        is_valid, errors, warnings = self.validator.validate_config(config)
        assert not is_valid
        assert any("must be 'poisson' or 'binomial'" in error for error in errors)
        
        # Test invalid alpha
        config['stats']['alpha'] = 1.5
        is_valid, errors, warnings = self.validator.validate_config(config)
        assert not is_valid
        assert any('must be between 0.0 and 1.0' in error for error in errors)
        
    def test_warnings(self):
        """Test generation of warnings for suboptimal configurations."""
        config = self.valid_config.copy()
        
        # Configuration that should generate warnings
        config['umi']['min_family_size'] = 1  # Low family size
        config['stats']['alpha'] = 0.15  # High alpha
        config['simulation']['n_bootstrap'] = 50  # Low bootstrap
        
        is_valid, errors, warnings = self.validator.validate_config(config)
        assert is_valid  # Should still be valid
        assert len(warnings) > 0
        assert any('poor consensus quality' in warning for warning in warnings)
        assert any('quite high' in warning for warning in warnings)
        assert any('low' in warning and 'bootstrap' in warning for warning in warnings)
        
    def test_run_id_validation(self):
        """Test run ID validation."""
        config = self.valid_config.copy()
        
        # Test empty run_id
        config['run_id'] = ''
        is_valid, errors, warnings = self.validator.validate_config(config)
        assert not is_valid
        assert any('cannot be empty' in error for error in errors)
        
        # Test non-string run_id
        config['run_id'] = 123
        is_valid, errors, warnings = self.validator.validate_config(config)
        assert not is_valid
        assert any('must be a string' in error for error in errors)
        
    def test_seed_validation(self):
        """Test seed validation."""
        config = self.valid_config.copy()
        
        # Test non-integer seed
        config['seed'] = '42'
        is_valid, errors, warnings = self.validator.validate_config(config)
        assert not is_valid
        assert any('must be an integer' in error for error in errors)
        
        # Test negative seed (should generate warning)
        config['seed'] = -1
        is_valid, errors, warnings = self.validator.validate_config(config)
        assert is_valid  # Still valid
        assert any('negative seed' in warning for warning in warnings)
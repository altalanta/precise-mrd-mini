"""
Configuration validation for the precise-mrd pipeline.

Provides comprehensive validation of configuration parameters
with detailed error reporting and suggested fixes.
"""

from typing import Dict, Any, List, Tuple
import logging
from pathlib import Path

from .exceptions import ConfigurationError


class ConfigValidator:
    """Validate configuration parameters for the pipeline."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.errors = []
        self.warnings = []
        
    def validate_config(self, config: Dict[str, Any]) -> Tuple[bool, List[str], List[str]]:
        """Validate complete configuration.
        
        Args:
            config: Configuration dictionary to validate
            
        Returns:
            Tuple of (is_valid, errors, warnings)
        """
        self.errors = []
        self.warnings = []
        
        # Validate required top-level keys
        required_keys = ['run_id', 'seed', 'simulation', 'umi', 'stats', 'lod']
        for key in required_keys:
            if key not in config:
                self.errors.append(f"Missing required configuration key: {key}")
        
        # Validate specific sections
        if 'simulation' in config:
            self._validate_simulation_config(config['simulation'])
            
        if 'umi' in config:
            self._validate_umi_config(config['umi'])
            
        if 'stats' in config:
            self._validate_stats_config(config['stats'])
            
        if 'lod' in config:
            self._validate_lod_config(config['lod'])
            
        # Validate general parameters
        self._validate_general_config(config)
        
        is_valid = len(self.errors) == 0
        return is_valid, self.errors, self.warnings
    
    def _validate_simulation_config(self, sim_config: Dict[str, Any]) -> None:
        """Validate simulation configuration."""
        # Required keys
        required_keys = ['allele_fractions', 'umi_depths', 'n_replicates', 'n_bootstrap']
        for key in required_keys:
            if key not in sim_config:
                self.errors.append(f"Missing simulation.{key}")
        
        # Validate allele fractions
        if 'allele_fractions' in sim_config:
            af_list = sim_config['allele_fractions']
            if not isinstance(af_list, list):
                self.errors.append("simulation.allele_fractions must be a list")
            else:
                if len(af_list) == 0:
                    self.errors.append("simulation.allele_fractions cannot be empty")
                
                for i, af in enumerate(af_list):
                    if not isinstance(af, (int, float)):
                        self.errors.append(f"simulation.allele_fractions[{i}] must be numeric")
                    elif not 0 <= af <= 1:
                        self.errors.append(f"simulation.allele_fractions[{i}] must be between 0 and 1")
                
                # Check ordering
                sorted_af = sorted(af_list, reverse=True)
                if af_list != sorted_af:
                    self.warnings.append("simulation.allele_fractions should be sorted in descending order")
        
        # Validate UMI depths
        if 'umi_depths' in sim_config:
            depth_list = sim_config['umi_depths']
            if not isinstance(depth_list, list):
                self.errors.append("simulation.umi_depths must be a list")
            else:
                if len(depth_list) == 0:
                    self.errors.append("simulation.umi_depths cannot be empty")
                
                for i, depth in enumerate(depth_list):
                    if not isinstance(depth, int):
                        self.errors.append(f"simulation.umi_depths[{i}] must be an integer")
                    elif depth < 100:
                        self.warnings.append(f"simulation.umi_depths[{i}] is very low ({depth}), consider >= 100")
                    elif depth > 100000:
                        self.warnings.append(f"simulation.umi_depths[{i}] is very high ({depth}), may be slow")
        
        # Validate replicate counts
        for key in ['n_replicates', 'n_bootstrap']:
            if key in sim_config:
                value = sim_config[key]
                if not isinstance(value, int):
                    self.errors.append(f"simulation.{key} must be an integer")
                elif value < 1:
                    self.errors.append(f"simulation.{key} must be positive")
                elif value < 100 and key == 'n_bootstrap':
                    self.warnings.append(f"simulation.{key} is low ({value}), consider >= 1000 for stable CIs")
    
    def _validate_umi_config(self, umi_config: Dict[str, Any]) -> None:
        """Validate UMI configuration."""
        # Required keys
        required_keys = ['min_family_size', 'max_family_size', 'quality_threshold', 'consensus_threshold']
        for key in required_keys:
            if key not in umi_config:
                self.errors.append(f"Missing umi.{key}")
        
        # Validate family size thresholds
        if 'min_family_size' in umi_config:
            min_size = umi_config['min_family_size']
            if not isinstance(min_size, int):
                self.errors.append("umi.min_family_size must be an integer")
            elif min_size < 1:
                self.errors.append("umi.min_family_size must be at least 1")
            elif min_size < 3:
                self.warnings.append("umi.min_family_size < 3 may result in poor consensus quality")
        
        if 'max_family_size' in umi_config:
            max_size = umi_config['max_family_size']
            if not isinstance(max_size, int):
                self.errors.append("umi.max_family_size must be an integer")
            elif max_size < 1:
                self.errors.append("umi.max_family_size must be positive")
        
        # Check consistency
        if ('min_family_size' in umi_config and 'max_family_size' in umi_config and
            isinstance(umi_config['min_family_size'], int) and
            isinstance(umi_config['max_family_size'], int)):
            if umi_config['min_family_size'] > umi_config['max_family_size']:
                self.errors.append("umi.min_family_size cannot exceed umi.max_family_size")
        
        # Validate quality threshold
        if 'quality_threshold' in umi_config:
            qual = umi_config['quality_threshold']
            if not isinstance(qual, (int, float)):
                self.errors.append("umi.quality_threshold must be numeric")
            elif not 0 <= qual <= 60:
                self.errors.append("umi.quality_threshold must be between 0 and 60")
            elif qual < 20:
                self.warnings.append("umi.quality_threshold < 20 may include many low-quality reads")
        
        # Validate consensus threshold
        if 'consensus_threshold' in umi_config:
            consensus = umi_config['consensus_threshold']
            if not isinstance(consensus, (int, float)):
                self.errors.append("umi.consensus_threshold must be numeric")
            elif not 0.0 < consensus <= 1.0:
                self.errors.append("umi.consensus_threshold must be between 0.0 and 1.0")
            elif consensus < 0.5:
                self.warnings.append("umi.consensus_threshold < 0.5 may result in ambiguous consensus calls")
    
    def _validate_stats_config(self, stats_config: Dict[str, Any]) -> None:
        """Validate statistics configuration."""
        # Required keys
        required_keys = ['test_type', 'alpha', 'fdr_method']
        for key in required_keys:
            if key not in stats_config:
                self.errors.append(f"Missing stats.{key}")
        
        # Validate test type
        if 'test_type' in stats_config:
            test_type = stats_config['test_type']
            if not isinstance(test_type, str):
                self.errors.append("stats.test_type must be a string")
            elif test_type not in ['poisson', 'binomial']:
                self.errors.append("stats.test_type must be 'poisson' or 'binomial'")
        
        # Validate alpha
        if 'alpha' in stats_config:
            alpha = stats_config['alpha']
            if not isinstance(alpha, (int, float)):
                self.errors.append("stats.alpha must be numeric")
            elif not 0.0 < alpha < 1.0:
                self.errors.append("stats.alpha must be between 0.0 and 1.0")
            elif alpha > 0.1:
                self.warnings.append(f"stats.alpha is quite high ({alpha}), consider <= 0.05")
        
        # Validate FDR method
        if 'fdr_method' in stats_config:
            fdr_method = stats_config['fdr_method']
            if not isinstance(fdr_method, str):
                self.errors.append("stats.fdr_method must be a string")
            elif fdr_method not in ['benjamini_hochberg', 'bonferroni', 'holm']:
                self.errors.append("stats.fdr_method must be 'benjamini_hochberg', 'bonferroni', or 'holm'")
    
    def _validate_lod_config(self, lod_config: Dict[str, Any]) -> None:
        """Validate LoD configuration."""
        # Required keys
        required_keys = ['detection_threshold', 'confidence_level']
        for key in required_keys:
            if key not in lod_config:
                self.errors.append(f"Missing lod.{key}")
        
        # Validate detection threshold
        if 'detection_threshold' in lod_config:
            det_thresh = lod_config['detection_threshold']
            if not isinstance(det_thresh, (int, float)):
                self.errors.append("lod.detection_threshold must be numeric")
            elif not 0.0 < det_thresh <= 1.0:
                self.errors.append("lod.detection_threshold must be between 0.0 and 1.0")
            elif det_thresh < 0.8:
                self.warnings.append("lod.detection_threshold < 0.8 is quite low for clinical use")
        
        # Validate confidence level
        if 'confidence_level' in lod_config:
            conf_level = lod_config['confidence_level']
            if not isinstance(conf_level, (int, float)):
                self.errors.append("lod.confidence_level must be numeric")
            elif not 0.0 < conf_level < 1.0:
                self.errors.append("lod.confidence_level must be between 0.0 and 1.0")
    
    def _validate_general_config(self, config: Dict[str, Any]) -> None:
        """Validate general configuration parameters."""
        # Validate run_id
        if 'run_id' in config:
            run_id = config['run_id']
            if not isinstance(run_id, str):
                self.errors.append("run_id must be a string")
            elif not run_id.strip():
                self.errors.append("run_id cannot be empty")
            elif not run_id.replace('_', '').replace('-', '').isalnum():
                self.warnings.append("run_id should contain only alphanumeric characters, dashes, and underscores")
        
        # Validate seed
        if 'seed' in config:
            seed = config['seed']
            if not isinstance(seed, int):
                self.errors.append("seed must be an integer")
            elif seed < 0:
                self.warnings.append("negative seed may cause issues with some random number generators")


def validate_config_file(config_path: Path) -> Tuple[bool, List[str], List[str]]:
    """Validate a configuration file.
    
    Args:
        config_path: Path to configuration YAML file
        
    Returns:
        Tuple of (is_valid, errors, warnings)
        
    Raises:
        ConfigurationError: If file cannot be read or parsed
    """
    import yaml
    
    if not config_path.exists():
        raise ConfigurationError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigurationError(f"Invalid YAML in configuration file: {e}") from e
    except (IOError, OSError) as e:
        raise ConfigurationError(f"Cannot read configuration file: {e}") from e
    
    if not isinstance(config, dict):
        raise ConfigurationError("Configuration file must contain a dictionary")
    
    validator = ConfigValidator()
    return validator.validate_config(config)
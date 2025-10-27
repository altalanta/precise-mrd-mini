"""Enhanced configuration management for precise MRD pipeline."""

from __future__ import annotations

import hashlib
import json
import yaml
import re
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable, Type
from abc import ABC, abstractmethod
import copy


@dataclass
class SimulationConfig:
    """Configuration for simulation parameters."""
    allele_fractions: List[float]
    umi_depths: List[int]
    n_replicates: int
    n_bootstrap: int

    def __post_init__(self):
        """Validate simulation configuration after initialization."""
        self._validate()

    def _validate(self):
        """Validate simulation configuration parameters."""
        if not self.allele_fractions:
            raise ValueError("allele_fractions cannot be empty")
        if any(af < 0 or af > 1 for af in self.allele_fractions):
            raise ValueError("allele_fractions must be between 0 and 1")
        if not self.umi_depths:
            raise ValueError("umi_depths cannot be empty")
        if any(depth <= 0 for depth in self.umi_depths):
            raise ValueError("umi_depths must be positive")
        if self.n_replicates <= 0:
            raise ValueError("n_replicates must be positive")
        if self.n_bootstrap <= 0:
            raise ValueError("n_bootstrap must be positive")

    def get_estimated_runtime(self) -> float:
        """Estimate runtime in minutes based on configuration."""
        # Rough estimation based on empirical data
        total_samples = len(self.allele_fractions) * len(self.umi_depths) * self.n_replicates
        return total_samples * 0.01  # 0.01 minutes per sample

    def adapt_to_data_characteristics(self, data_stats: Dict[str, Any]) -> 'SimulationConfig':
        """Adapt configuration based on data characteristics."""
        adapted = copy.deepcopy(self)

        # Adjust allele fractions based on observed variant frequencies
        if 'variant_frequencies' in data_stats:
            observed_af = data_stats['variant_frequencies']
            min_af = min(observed_af) if observed_af else 0.001
            max_af = max(observed_af) if observed_af else 0.1

            # Adjust allele fractions to cover observed range
            adapted.allele_fractions = [min_af / 10, min_af, max_af, max_af * 10]

        # Adjust depths based on observed read depths
        if 'read_depths' in data_stats:
            observed_depths = data_stats['read_depths']
            if observed_depths:
                mean_depth = sum(observed_depths) / len(observed_depths)
                # Adjust depths to be around observed mean
                adapted.umi_depths = [int(mean_depth * 0.5), int(mean_depth), int(mean_depth * 2)]

        return adapted


@dataclass
class UMIConfig:
    """Configuration for UMI processing."""
    min_family_size: int
    max_family_size: int
    quality_threshold: int
    consensus_threshold: float

    def __post_init__(self):
        """Validate UMI configuration after initialization."""
        self._validate()

    def _validate(self):
        """Validate UMI configuration parameters."""
        if self.min_family_size <= 0:
            raise ValueError("min_family_size must be positive")
        if self.max_family_size < self.min_family_size:
            raise ValueError("max_family_size must be >= min_family_size")
        if not 0 <= self.quality_threshold <= 60:
            raise ValueError("quality_threshold must be between 0 and 60")
        if not 0 <= self.consensus_threshold <= 1:
            raise ValueError("consensus_threshold must be between 0 and 1")

    def adapt_to_data_quality(self, quality_stats: Dict[str, Any]) -> 'UMIConfig':
        """Adapt UMI configuration based on data quality characteristics."""
        adapted = copy.deepcopy(self)

        # Adjust quality threshold based on observed quality distribution
        if 'mean_quality' in quality_stats:
            observed_mean_quality = quality_stats['mean_quality']
            # Set threshold to be slightly below observed mean quality
            adapted.quality_threshold = max(10, int(observed_mean_quality * 0.8))

        # Adjust family size requirements based on observed family sizes
        if 'family_sizes' in quality_stats:
            family_sizes = quality_stats['family_sizes']
            if family_sizes:
                median_family_size = sorted(family_sizes)[len(family_sizes) // 2]
                # Ensure minimum family size is reasonable for the dataset
                adapted.min_family_size = max(1, min(self.min_family_size, median_family_size // 2))

        return adapted


@dataclass
class StatsConfig:
    """Configuration for statistical testing."""
    test_type: str
    alpha: float
    fdr_method: str

    def __post_init__(self):
        """Validate statistical configuration after initialization."""
        self._validate()

    def _validate(self):
        """Validate statistical configuration parameters."""
        valid_test_types = ['poisson', 'binomial', 'fisher']
        if self.test_type not in valid_test_types:
            raise ValueError(f"test_type must be one of {valid_test_types}")

        if not 0 < self.alpha < 1:
            raise ValueError("alpha must be between 0 and 1")

        valid_fdr_methods = ['benjamini_hochberg', 'bonferroni', 'holm']
        if self.fdr_method not in valid_fdr_methods:
            raise ValueError(f"fdr_method must be one of {valid_fdr_methods}")

    def get_power_analysis_config(self) -> Dict[str, Any]:
        """Get configuration for power analysis."""
        return {
            'test_type': self.test_type,
            'alpha': self.alpha,
            'fdr_method': self.fdr_method,
            'requires_multiple_testing_correction': self.fdr_method != 'none'
        }


@dataclass
class LODConfig:
    """Configuration for LoD estimation."""
    detection_threshold: float
    confidence_level: float

    def __post_init__(self):
        """Validate LOD configuration after initialization."""
        self._validate()

    def _validate(self):
        """Validate LOD configuration parameters."""
        if not 0 < self.detection_threshold < 1:
            raise ValueError("detection_threshold must be between 0 and 1")
        if not 0 < self.confidence_level < 1:
            raise ValueError("confidence_level must be between 0 and 1")

    def get_bootstrap_config(self) -> Dict[str, Any]:
        """Get bootstrap configuration for LOD estimation."""
        return {
            'detection_threshold': self.detection_threshold,
            'confidence_level': self.confidence_level,
            'bootstrap_confidence_interval': True
        }


@dataclass
class FASTQConfig:
    """Configuration for FASTQ file processing."""
    input_path: str
    max_reads: Optional[int] = None
    umi_pattern: Optional[str] = None
    quality_threshold: int = 20
    min_family_size: int = 3

    def __post_init__(self):
        """Validate FASTQ configuration after initialization."""
        self._validate()

    def _validate(self):
        """Validate FASTQ configuration parameters."""
        if not self.input_path:
            raise ValueError("input_path cannot be empty")
        if self.max_reads is not None and self.max_reads <= 0:
            raise ValueError("max_reads must be positive")
        if not 0 <= self.quality_threshold <= 60:
            raise ValueError("quality_threshold must be between 0 and 60")
        if self.min_family_size <= 0:
            raise ValueError("min_family_size must be positive")


class ConfigurationTemplate(ABC):
    """Abstract base class for configuration templates."""

    @abstractmethod
    def get_base_config(self) -> Dict[str, Any]:
        """Get the base configuration template."""
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Get the template name."""
        pass


@dataclass
class ConfigVersion:
    """Configuration version information."""
    major: int
    minor: int
    patch: int
    description: str = ""

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"

    def __lt__(self, other: 'ConfigVersion') -> bool:
        return (self.major, self.minor, self.patch) < (other.major, other.minor, other.patch)


@dataclass
class PipelineConfig:
    """Enhanced main pipeline configuration with inheritance and validation."""
    run_id: str
    seed: int
    umi: UMIConfig
    stats: StatsConfig
    lod: LODConfig
    simulation: Optional[SimulationConfig] = None
    fastq: Optional[FASTQConfig] = None
    # New fields for enhanced configuration management
    config_version: str = "2.0.0"
    parent_config: Optional[str] = None
    template: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    description: str = ""
    created_at: Optional[str] = None
    last_modified: Optional[str] = None

    def __post_init__(self):
        """Validate and enhance configuration after initialization."""
        self._validate()
        self._set_timestamps()

    def _validate(self):
        """Comprehensive validation of pipeline configuration."""
        if not self.run_id:
            raise ValueError("run_id cannot be empty")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")

        # Validate component configurations
        self.umi._validate()
        self.stats._validate()
        self.lod._validate()

        if self.simulation:
            self.simulation._validate()
        if self.fastq:
            self.fastq._validate()

        # Validate configuration consistency
        self._validate_consistency()

    def _validate_consistency(self):
        """Validate configuration consistency across components."""
        # Ensure simulation and FASTQ configs don't conflict
        if self.simulation and self.fastq:
            # If both are present, ensure they make sense together
            pass  # Add specific validation rules as needed

    def _set_timestamps(self):
        """Set creation and modification timestamps."""
        import datetime
        now = datetime.datetime.now().isoformat()
        if not self.created_at:
            self.created_at = now
        self.last_modified = now

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary with metadata."""
        result = asdict(self)
        result['config_version'] = str(self.config_version)
        return result

    def config_hash(self) -> str:
        """Compute deterministic hash of configuration."""
        # Include version and parent info for more robust hashing
        config_str = json.dumps({
            'version': self.config_version,
            'parent': self.parent_config,
            'core_config': {
                'run_id': self.run_id,
                'seed': self.seed,
                'umi': self.umi.__dict__,
                'stats': self.stats.__dict__,
                'lod': self.lod.__dict__,
                'simulation': self.simulation.__dict__ if self.simulation else None,
                'fastq': self.fastq.__dict__ if self.fastq else None,
            }
        }, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]

    def get_estimated_runtime(self) -> float:
        """Estimate total runtime in minutes."""
        runtime = 0.0

        if self.simulation:
            runtime += self.simulation.get_estimated_runtime()

        # Add runtime for other pipeline stages
        # This is a rough estimate - in practice you'd want more sophisticated modeling
        if self.fastq:
            # FASTQ processing is I/O bound, estimate based on file size
            runtime += 5.0  # Base estimate

        return runtime

    def adapt_to_data(self, data_characteristics: Dict[str, Any]) -> 'PipelineConfig':
        """Create an adapted configuration based on data characteristics."""
        adapted = copy.deepcopy(self)

        # Adapt simulation configuration if present
        if adapted.simulation and 'variant_frequencies' in data_characteristics:
            adapted.simulation = adapted.simulation.adapt_to_data_characteristics(data_characteristics)

        # Adapt UMI configuration based on quality data
        if 'quality_stats' in data_characteristics:
            adapted.umi = adapted.umi.adapt_to_data_quality(data_characteristics['quality_stats'])

        # Update metadata
        adapted.description = f"Auto-adapted from {self.run_id}"
        adapted.parent_config = self.run_id
        adapted.last_modified = None  # Will be set by __post_init__

        return adapted

    def validate_compatibility(self, other: 'PipelineConfig') -> List[str]:
        """Validate compatibility between two configurations."""
        issues = []

        # Check version compatibility
        if self.config_version != other.config_version:
            issues.append(f"Configuration version mismatch: {self.config_version} vs {other.config_version}")

        # Check for incompatible parameter combinations
        if (self.simulation and other.fastq) or (self.fastq and other.simulation):
            issues.append("Cannot mix simulation and FASTQ modes")

        return issues

    def merge_with(self, other: 'PipelineConfig', strategy: str = 'override') -> 'PipelineConfig':
        """Merge this configuration with another using specified strategy."""
        if strategy == 'override':
            # Other config overrides this config
            merged = copy.deepcopy(other)
            merged.run_id = f"{self.run_id}_merged_{other.run_id}"
        elif strategy == 'inherit':
            # Inherit non-conflicting settings from other config
            merged = copy.deepcopy(self)
            # Apply inheritance logic here
        else:
            raise ValueError(f"Unknown merge strategy: {strategy}")

        merged.parent_config = self.run_id
        merged.last_modified = None  # Will be set by __post_init__
        return merged

    def export_template(self) -> Dict[str, Any]:
        """Export configuration as a reusable template."""
        return {
            'template_name': self.run_id,
            'description': self.description,
            'base_config': {
                'umi': self.umi.__dict__,
                'stats': self.stats.__dict__,
                'lod': self.lod.__dict__,
                'simulation': self.simulation.__dict__ if self.simulation else None,
            },
            'tags': self.tags,
            'version': self.config_version
        }

    @classmethod
    def from_template(cls, template: Dict[str, Any], run_id: str = None) -> 'PipelineConfig':
        """Create configuration from template."""
        base_config = template['base_config']

        return cls(
            run_id=run_id or f"from_template_{template['template_name']}",
            seed=7,  # Default seed
            umi=UMIConfig(**base_config['umi']),
            stats=StatsConfig(**base_config['stats']),
            lod=LODConfig(**base_config['lod']),
            simulation=SimulationConfig(**base_config['simulation']) if base_config['simulation'] else None,
            template=template['template_name'],
            description=f"Generated from template: {template['template_name']}",
            tags=template.get('tags', [])
        )


def load_config(path: str | Path, auto_migrate: bool = True) -> PipelineConfig:
    """Load configuration from YAML file with validation and optional migration."""
    with open(path, 'r') as f:
        data = yaml.safe_load(f)

    # Handle legacy configurations without version info
    if 'config_version' not in data:
        data['config_version'] = "1.0.0"

    # Auto-migrate to latest version if requested
    if auto_migrate:
        data = ConfigVersionManager.migrate_config(data)

    # Map old format to new format if needed
    if 'simulation' in data and data['simulation']:
        data['simulation'] = _migrate_simulation_config(data['simulation'])

    return PipelineConfig(
        run_id=data.get('run_id', 'unnamed_run'),
        seed=data.get('seed', 7),
        simulation=SimulationConfig(**data['simulation']) if data.get('simulation') else None,
        umi=UMIConfig(**data['umi']),
        stats=StatsConfig(**data['stats']),
        lod=LODConfig(**data['lod']),
        fastq=FASTQConfig(**data['fastq']) if data.get('fastq') else None,
        config_version=data.get('config_version', '2.0.0'),
        description=data.get('description', ''),
        tags=data.get('tags', [])
    )


def dump_config(config: PipelineConfig, path: str | Path) -> None:
    """Save configuration to YAML file with full metadata."""
    with open(path, 'w') as f:
        yaml.safe_dump(config.to_dict(), f, default_flow_style=False, sort_keys=False)


def _migrate_simulation_config(old_config: Dict[str, Any]) -> Dict[str, Any]:
    """Migrate old simulation config format to new format."""
    # Add any migration logic here if needed
    return old_config


class ConfigVersionManager:
    """Manages configuration versioning and migration."""

    # Define migration paths between versions
    MIGRATION_PATHS = {
        "1.0.0": {
            "target": "2.0.0",
            "changes": [
                "Added config_version field",
                "Added parent_config field",
                "Added template field",
                "Added tags field",
                "Added description field",
                "Added timestamps",
                "Enhanced validation in all config classes",
                "Added dynamic adaptation methods"
            ]
        }
    }

    @staticmethod
    def get_latest_version() -> str:
        """Get the latest configuration version."""
        return "2.0.0"

    @staticmethod
    def migrate_config(config_data: Dict[str, Any], target_version: str = None) -> Dict[str, Any]:
        """Migrate configuration data to target version."""
        current_version = config_data.get('config_version', '1.0.0')
        target_version = target_version or ConfigVersionManager.get_latest_version()

        if current_version == target_version:
            return config_data

        # Apply migrations step by step
        migrated_data = config_data.copy()

        if current_version == "1.0.0" and target_version == "2.0.0":
            migrated_data = ConfigVersionManager._migrate_1_to_2(migrated_data)

        return migrated_data

    @staticmethod
    def _migrate_1_to_2(config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Migrate from version 1.0.0 to 2.0.0."""
        migrated = config_data.copy()

        # Add new fields with defaults
        migrated['config_version'] = "2.0.0"
        migrated['parent_config'] = None
        migrated['template'] = None
        migrated['tags'] = []
        migrated['description'] = ""
        migrated['created_at'] = None
        migrated['last_modified'] = None

        return migrated

    @staticmethod
    def get_migration_info(from_version: str, to_version: str) -> Dict[str, Any]:
        """Get information about migrating between versions."""
        if from_version == to_version:
            return {"status": "no_migration_needed", "changes": []}

        migration_key = f"{from_version}_to_{to_version}"
        if migration_key in ConfigVersionManager.MIGRATION_PATHS:
            return ConfigVersionManager.MIGRATION_PATHS[migration_key]

        return {"status": "migration_not_found", "changes": []}

    @staticmethod
    def validate_version_compatibility(config_version: str, required_version: str) -> bool:
        """Check if a configuration version is compatible with required version."""
        # Simple version comparison - in practice, you'd want more sophisticated logic
        config_ver = ConfigVersionManager._parse_version(config_version)
        required_ver = ConfigVersionManager._parse_version(required_version)

        return config_ver >= required_ver

    @staticmethod
    def _parse_version(version_str: str) -> ConfigVersion:
        """Parse version string into ConfigVersion object."""
        parts = version_str.split('.')
        if len(parts) != 3:
            raise ValueError(f"Invalid version format: {version_str}")

        return ConfigVersion(
            major=int(parts[0]),
            minor=int(parts[1]),
            patch=int(parts[2])
        )


class ConfigValidator:
    """Configuration validation and analysis tools."""

    @staticmethod
    def validate_config(config: PipelineConfig) -> Dict[str, Any]:
        """Comprehensive configuration validation."""
        issues = []
        warnings = []
        suggestions = []

        try:
            # Basic validation is already done in __post_init__
            config._validate()
        except ValueError as e:
            issues.append(f"Validation error: {e}")

        # Check for potential issues and improvements
        if config.simulation:
            sim_issues = ConfigValidator._validate_simulation_config(config.simulation)
            issues.extend(sim_issues)

        if config.fastq:
            fastq_issues = ConfigValidator._validate_fastq_config(config.fastq)
            issues.extend(fastq_issues)

        # Check for performance implications
        estimated_runtime = config.get_estimated_runtime()
        if estimated_runtime > 60:  # More than 1 hour
            warnings.append(f"Estimated runtime is {estimated_runtime:.1f} minutes - consider optimization")

        # Configuration suggestions
        if not config.description:
            suggestions.append("Add a description to document the purpose of this configuration")

        if not config.tags:
            suggestions.append("Add tags to categorize this configuration")

        return {
            'is_valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings,
            'suggestions': suggestions,
            'estimated_runtime_minutes': estimated_runtime,
            'config_hash': config.config_hash()
        }

    @staticmethod
    def _validate_simulation_config(sim_config: SimulationConfig) -> List[str]:
        """Validate simulation configuration for potential issues."""
        issues = []

        # Check for extreme parameter combinations
        if len(sim_config.allele_fractions) * len(sim_config.umi_depths) * sim_config.n_replicates > 10000:
            issues.append("Large parameter space may result in very long execution times")

        # Check for unrealistic allele frequencies
        if any(af > 0.5 for af in sim_config.allele_fractions):
            issues.append("Allele frequencies above 0.5 may be unrealistic for typical MRD scenarios")

        return issues

    @staticmethod
    def _validate_fastq_config(fastq_config: FASTQConfig) -> List[str]:
        """Validate FASTQ configuration for potential issues."""
        issues = []

        # Check if file exists
        if not Path(fastq_config.input_path).exists():
            issues.append(f"FASTQ file not found: {fastq_config.input_path}")

        return issues

    @staticmethod
    def suggest_optimizations(config: PipelineConfig) -> List[str]:
        """Suggest configuration optimizations."""
        suggestions = []

        if config.simulation:
            # Suggest reducing bootstrap iterations for faster runs
            if config.simulation.n_bootstrap > 1000:
                suggestions.append("Consider reducing n_bootstrap for faster execution")

            # Suggest appropriate allele frequency ranges
            if len(config.simulation.allele_fractions) > 5:
                suggestions.append("Consider reducing the number of allele fractions for faster analysis")

        return suggestions


# Predefined configuration templates
class PredefinedTemplates:
    """Predefined configuration templates for common use cases."""

    @staticmethod
    def get_smoke_test_template() -> Dict[str, Any]:
        """Get smoke test configuration template."""
        return {
            'template_name': 'smoke_test',
            'description': 'Minimal configuration for quick smoke testing',
            'base_config': {
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
                },
                'simulation': {
                    'allele_fractions': [0.01, 0.001, 0.0001],
                    'umi_depths': [1000, 5000],
                    'n_replicates': 10,
                    'n_bootstrap': 100
                }
            },
            'tags': ['smoke_test', 'quick', 'minimal'],
            'version': '2.0.0'
        }

    @staticmethod
    def get_production_template() -> Dict[str, Any]:
        """Get production-ready configuration template."""
        return {
            'template_name': 'production',
            'description': 'Production configuration with comprehensive validation',
            'base_config': {
                'umi': {
                    'min_family_size': 5,
                    'max_family_size': 10000,
                    'quality_threshold': 25,
                    'consensus_threshold': 0.7
                },
                'stats': {
                    'test_type': 'poisson',
                    'alpha': 0.01,
                    'fdr_method': 'benjamini_hochberg'
                },
                'lod': {
                    'detection_threshold': 0.95,
                    'confidence_level': 0.99
                },
                'simulation': {
                    'allele_fractions': [0.1, 0.01, 0.001, 0.0001, 0.00001],
                    'umi_depths': [1000, 5000, 10000, 25000],
                    'n_replicates': 50,
                    'n_bootstrap': 1000
                }
            },
            'tags': ['production', 'comprehensive', 'high_accuracy'],
            'version': '2.0.0'
        }
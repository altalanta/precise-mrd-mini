"""Enhanced configuration management for precise MRD pipeline."""

from __future__ import annotations

import copy
import hashlib
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .enums import FDRMethod, StatisticalTestType
from .exceptions import ConfigurationError


class SimulationConfig(BaseModel):
    """Configuration for simulation parameters."""

    allele_fractions: list[float]
    umi_depths: list[int]
    n_replicates: int
    n_bootstrap: int

    @field_validator("allele_fractions")
    @classmethod
    def validate_allele_fractions(cls, v: list[float]) -> list[float]:
        if not v:
            raise ConfigurationError("allele_fractions cannot be empty")
        if any(af < 0 or af > 1 for af in v):
            raise ConfigurationError("allele_fractions must be between 0 and 1")
        return v

    def get_estimated_runtime(self) -> float:
        """Estimate runtime in minutes based on configuration."""
        # Rough estimation based on empirical data
        total_samples = (
            len(self.allele_fractions) * len(self.umi_depths) * self.n_replicates
        )
        return total_samples * 0.01  # 0.01 minutes per sample

    def adapt_to_data_characteristics(
        self,
        data_stats: dict[str, Any],
    ) -> SimulationConfig:
        """Adapt configuration based on data characteristics."""
        adapted = copy.deepcopy(self)

        # Adjust allele fractions based on observed variant frequencies
        if "variant_frequencies" in data_stats:
            observed_af = data_stats["variant_frequencies"]
            min_af = min(observed_af) if observed_af else 0.001
            max_af = max(observed_af) if observed_af else 0.1

            # Adjust allele fractions to cover observed range
            adapted.allele_fractions = [min_af / 10, min_af, max_af, max_af * 10]

        # Adjust depths based on observed read depths
        if "read_depths" in data_stats:
            observed_depths = data_stats["read_depths"]
            if observed_depths:
                mean_depth = sum(observed_depths) / len(observed_depths)
                # Adjust depths to be around observed mean
                adapted.umi_depths = [
                    int(mean_depth * 0.5),
                    int(mean_depth),
                    int(mean_depth * 2),
                ]

        return adapted


class UMIConfig(BaseModel):
    """Configuration for UMI processing."""

    min_family_size: int
    max_family_size: int
    quality_threshold: int
    consensus_threshold: float

    @field_validator("min_family_size")
    @classmethod
    def validate_min_family_size(cls, v: int) -> int:
        if v <= 0:
            raise ConfigurationError("min_family_size must be positive")
        return v

    @field_validator("quality_threshold")
    @classmethod
    def validate_quality_threshold(cls, v: int) -> int:
        if not 0 <= v <= 60:
            raise ConfigurationError("quality_threshold must be between 0 and 60")
        return v

    @field_validator("consensus_threshold")
    @classmethod
    def validate_consensus_threshold(cls, v: float) -> float:
        if not 0 <= v <= 1:
            raise ConfigurationError("consensus_threshold must be between 0 and 1")
        return v

    @model_validator(mode="after")
    def validate_family_size_range(self) -> Self:
        """Validate that max_family_size >= min_family_size."""
        if self.max_family_size < self.min_family_size:
            raise ConfigurationError("max_family_size must be >= min_family_size")
        return self

    def adapt_to_data_quality(self, quality_stats: dict[str, Any]) -> UMIConfig:
        """Adapt UMI configuration based on data quality characteristics."""
        adapted = copy.deepcopy(self)

        # Adjust quality threshold based on observed quality distribution
        if "mean_quality" in quality_stats:
            observed_mean_quality = quality_stats["mean_quality"]
            # Set threshold to be slightly below observed mean quality
            adapted.quality_threshold = max(10, int(observed_mean_quality * 0.8))

        # Adjust family size requirements based on observed family sizes
        if "family_sizes" in quality_stats:
            family_sizes = quality_stats["family_sizes"]
            if family_sizes:
                median_family_size = sorted(family_sizes)[len(family_sizes) // 2]
                # Ensure minimum family size is reasonable for the dataset
                adapted.min_family_size = max(
                    1,
                    min(self.min_family_size, median_family_size // 2),
                )

        return adapted


class StatsConfig(BaseModel):
    """Configuration for statistical testing.

    Attributes:
        test_type: Statistical test to use for variant calling.
            Must be one of: "poisson", "binomial", "fisher".
        alpha: Significance level for hypothesis testing (0 < alpha < 1).
        fdr_method: False discovery rate correction method.
            Must be one of: "benjamini_hochberg", "bonferroni", "holm".
    """

    test_type: StatisticalTestType = Field(
        ...,
        description="Statistical test type: poisson, binomial, or fisher",
    )
    alpha: float = Field(
        ...,
        gt=0,
        lt=1,
        description="Significance level (0 < alpha < 1)",
    )
    fdr_method: FDRMethod = Field(
        ...,
        description="FDR correction method: benjamini_hochberg, bonferroni, or holm",
    )

    def get_power_analysis_config(self) -> dict[str, Any]:
        """Get configuration for power analysis."""
        return {
            "test_type": self.test_type,
            "alpha": self.alpha,
            "fdr_method": self.fdr_method,
            "requires_multiple_testing_correction": self.fdr_method != "none",
        }


class LODConfig(BaseModel):
    """Configuration for LoD estimation."""

    detection_threshold: float
    confidence_level: float

    @field_validator("detection_threshold")
    @classmethod
    def validate_detection_threshold(cls, v: float) -> float:
        if not 0 < v < 1:
            raise ConfigurationError("detection_threshold must be between 0 and 1")
        return v

    @field_validator("confidence_level")
    @classmethod
    def validate_confidence_level(cls, v: float) -> float:
        if not 0 < v < 1:
            raise ConfigurationError("confidence_level must be between 0 and 1")
        return v

    def get_bootstrap_config(self) -> dict[str, Any]:
        """Get bootstrap configuration for LOD estimation."""
        return {
            "detection_threshold": self.detection_threshold,
            "confidence_level": self.confidence_level,
            "bootstrap_confidence_interval": True,
        }


class FASTQConfig(BaseModel):
    """Configuration for FASTQ file processing."""

    input_path: str
    max_reads: int | None = None
    umi_pattern: str | None = None
    quality_threshold: int = 20
    min_family_size: int = 3

    @field_validator("input_path")
    @classmethod
    def validate_input_path(cls, v: str) -> str:
        if not v:
            raise ConfigurationError("input_path cannot be empty")
        return v

    @field_validator("max_reads")
    @classmethod
    def validate_max_reads(cls, v: int | None) -> int | None:
        if v is not None and v <= 0:
            raise ConfigurationError("max_reads must be positive")
        return v

    @field_validator("quality_threshold")
    @classmethod
    def validate_quality_threshold(cls, v: int) -> int:
        if not 0 <= v <= 60:
            raise ConfigurationError("quality_threshold must be between 0 and 60")
        return v

    @field_validator("min_family_size")
    @classmethod
    def validate_min_family_size(cls, v: int) -> int:
        if v <= 0:
            raise ConfigurationError("min_family_size must be positive")
        return v


class ConfigurationTemplate(ABC):
    """Abstract base class for configuration templates."""

    @abstractmethod
    def get_base_config(self) -> dict[str, Any]:
        """Get the base configuration template."""
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Get the template name."""
        pass


class ConfigVersion(BaseModel):
    """Configuration version information."""

    major: int
    minor: int
    patch: int
    description: str = ""

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"

    def __lt__(self, other: ConfigVersion) -> bool:
        return (self.major, self.minor, self.patch) < (
            other.major,
            other.minor,
            other.patch,
        )


class PipelineConfig(BaseModel):
    """Enhanced main pipeline configuration with inheritance and validation."""

    model_config = ConfigDict(validate_assignment=True)

    run_id: str
    seed: int
    umi: UMIConfig
    stats: StatsConfig
    lod: LODConfig
    simulation: SimulationConfig | None = None
    fastq: FASTQConfig | None = None
    config_version: str = "2.0.0"
    parent_config: str | None = None
    template: str | None = None
    tags: list[str] = Field(default_factory=list)
    description: str = ""
    created_at: str | None = None
    last_modified: str | None = None

    def __init__(self, **data):
        super().__init__(**data)
        self._set_timestamps()

    def _set_timestamps(self):
        """Set creation and modification timestamps using UTC."""
        from datetime import datetime, timezone

        now = datetime.now(timezone.utc).isoformat()
        if not self.created_at:
            self.created_at = now
        self.last_modified = now

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary with metadata."""
        return self.model_dump(mode="json")

    def config_hash(self) -> str:
        """Compute deterministic hash of configuration."""
        config_str = self.model_dump_json(exclude={"created_at", "last_modified"})
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

    def adapt_to_data(self, data_characteristics: dict[str, Any]) -> PipelineConfig:
        """Create an adapted configuration based on data characteristics."""
        adapted = copy.deepcopy(self)

        # Adapt simulation configuration if present
        if adapted.simulation and "variant_frequencies" in data_characteristics:
            adapted.simulation = adapted.simulation.adapt_to_data_characteristics(
                data_characteristics,
            )

        # Adapt UMI configuration based on quality data
        if "quality_stats" in data_characteristics:
            adapted.umi = adapted.umi.adapt_to_data_quality(
                data_characteristics["quality_stats"],
            )

        # Update metadata
        adapted.description = f"Auto-adapted from {self.run_id}"
        adapted.parent_config = self.run_id
        adapted.last_modified = None  # Will be set by __post_init__

        return adapted

    def validate_compatibility(self, other: PipelineConfig) -> list[str]:
        """Validate compatibility between two configurations."""
        issues = []

        # Check version compatibility
        if self.config_version != other.config_version:
            issues.append(
                f"Configuration version mismatch: {self.config_version} vs {other.config_version}",
            )

        # Check for incompatible parameter combinations
        if (self.simulation and other.fastq) or (self.fastq and other.simulation):
            issues.append("Cannot mix simulation and FASTQ modes")

        return issues

    def merge_with(
        self,
        other: PipelineConfig,
        strategy: str = "override",
    ) -> PipelineConfig:
        """Merge this configuration with another using specified strategy."""
        if strategy == "override":
            # Other config overrides this config
            merged = copy.deepcopy(other)
            merged.run_id = f"{self.run_id}_merged_{other.run_id}"
        elif strategy == "inherit":
            # Inherit non-conflicting settings from other config
            merged = copy.deepcopy(self)
            # Apply inheritance logic here
        else:
            raise ConfigurationError(f"Unknown merge strategy: {strategy}")

        merged.parent_config = self.run_id
        merged.last_modified = None  # Will be set by __post_init__
        return merged

    def export_template(self) -> dict[str, Any]:
        """Export configuration as a reusable template."""
        return {
            "template_name": self.run_id,
            "description": self.description,
            "base_config": {
                "umi": self.umi.__dict__,
                "stats": self.stats.__dict__,
                "lod": self.lod.__dict__,
                "simulation": self.simulation.__dict__ if self.simulation else None,
            },
            "tags": self.tags,
            "version": self.config_version,
        }

    @classmethod
    def from_template(
        cls,
        template: dict[str, Any],
        run_id: str = None,
    ) -> PipelineConfig:
        """Create configuration from template."""
        base_config = template["base_config"]

        return cls(
            run_id=run_id or f"from_template_{template['template_name']}",
            seed=7,  # Default seed
            umi=UMIConfig(**base_config["umi"]),
            stats=StatsConfig(**base_config["stats"]),
            lod=LODConfig(**base_config["lod"]),
            simulation=SimulationConfig(**base_config["simulation"])
            if base_config["simulation"]
            else None,
            template=template["template_name"],
            description=f"Generated from template: {template['template_name']}",
            tags=template.get("tags", []),
        )


def load_config(path: str | Path, auto_migrate: bool = True) -> PipelineConfig:
    """Load configuration from YAML file with validation and optional migration.

    Args:
        path: Path to the YAML configuration file.
        auto_migrate: If True, automatically migrates older configuration versions
            to the latest format. Defaults to True.

    Returns:
        A validated PipelineConfig instance.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        ConfigurationError: If the configuration is invalid or cannot be parsed.
        yaml.YAMLError: If the file contains invalid YAML syntax.
    """
    with open(path) as f:
        data = yaml.safe_load(f)

    if auto_migrate:
        data = ConfigVersionManager.migrate_config(data)

    return PipelineConfig(**data)


def dump_config(config: PipelineConfig, path: str | Path | None) -> str | None:
    """Save configuration to YAML file with full metadata.

    If path is None, returns the YAML string instead of writing to file.

    Args:
        config: The PipelineConfig instance to serialize.
        path: Path to write the YAML file, or None to return as string.

    Returns:
        The YAML string if path is None, otherwise None.

    Raises:
        PermissionError: If the file cannot be written to the specified path.
    """
    yaml_str = yaml.safe_dump(
        config.to_dict(),
        default_flow_style=False,
        sort_keys=False,
    )
    if path is None:
        return yaml_str
    with open(path, "w") as f:
        f.write(yaml_str)
    return None


def _migrate_simulation_config(old_config: dict[str, Any]) -> dict[str, Any]:
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
                "Added dynamic adaptation methods",
            ],
        },
    }

    @staticmethod
    def get_latest_version() -> str:
        """Get the latest configuration version."""
        return "2.0.0"

    @staticmethod
    def migrate_config(
        config_data: dict[str, Any],
        target_version: str = None,
    ) -> dict[str, Any]:
        """Migrate configuration data to target version."""
        current_version = config_data.get("config_version", "1.0.0")
        target_version = target_version or ConfigVersionManager.get_latest_version()

        if current_version == target_version:
            return config_data

        # Apply migrations step by step
        migrated_data = config_data.copy()

        if current_version == "1.0.0" and target_version == "2.0.0":
            migrated_data = ConfigVersionManager._migrate_1_to_2(migrated_data)

        return migrated_data

    @staticmethod
    def _migrate_1_to_2(config_data: dict[str, Any]) -> dict[str, Any]:
        """Migrate from version 1.0.0 to 2.0.0."""
        migrated = config_data.copy()

        # Add new fields with defaults
        migrated["config_version"] = "2.0.0"
        migrated["parent_config"] = None
        migrated["template"] = None
        migrated["tags"] = []
        migrated["description"] = ""
        migrated["created_at"] = None
        migrated["last_modified"] = None

        return migrated

    @staticmethod
    def get_migration_info(from_version: str, to_version: str) -> dict[str, Any]:
        """Get information about migrating between versions."""
        if from_version == to_version:
            return {"status": "no_migration_needed", "changes": []}

        migration_key = f"{from_version}_to_{to_version}"
        if migration_key in ConfigVersionManager.MIGRATION_PATHS:
            return ConfigVersionManager.MIGRATION_PATHS[migration_key]

        return {"status": "migration_not_found", "changes": []}

    @staticmethod
    def validate_version_compatibility(
        config_version: str,
        required_version: str,
    ) -> bool:
        """Check if a configuration version is compatible with required version."""
        # Simple version comparison - in practice, you'd want more sophisticated logic
        config_ver = ConfigVersionManager._parse_version(config_version)
        required_ver = ConfigVersionManager._parse_version(required_version)

        return config_ver >= required_ver

    @staticmethod
    def _parse_version(version_str: str) -> ConfigVersion:
        """Parse version string into ConfigVersion object."""
        parts = version_str.split(".")
        if len(parts) != 3:
            raise ConfigurationError(f"Invalid version format: {version_str}")

        return ConfigVersion(
            major=int(parts[0]),
            minor=int(parts[1]),
            patch=int(parts[2]),
        )


class ConfigValidator:
    """Configuration validation and analysis tools."""

    @staticmethod
    def validate_config(config: PipelineConfig) -> dict[str, Any]:
        """Comprehensive configuration validation."""
        issues = []
        warnings = []
        suggestions = []

        try:
            # Basic validation is already done in __post_init__
            config._set_timestamps()  # Ensure timestamps are set for validation
            config.model_dump()  # Use model_dump for validation
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
            warnings.append(
                f"Estimated runtime is {estimated_runtime:.1f} minutes - consider optimization",
            )

        # Configuration suggestions
        if not config.description:
            suggestions.append(
                "Add a description to document the purpose of this configuration",
            )

        if not config.tags:
            suggestions.append("Add tags to categorize this configuration")

        return {
            "is_valid": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
            "suggestions": suggestions,
            "estimated_runtime_minutes": estimated_runtime,
            "config_hash": config.config_hash(),
        }

    @staticmethod
    def _validate_simulation_config(sim_config: SimulationConfig) -> list[str]:
        """Validate simulation configuration for potential issues."""
        issues = []

        # Check for extreme parameter combinations
        if (
            len(sim_config.allele_fractions)
            * len(sim_config.umi_depths)
            * sim_config.n_replicates
            > 10000
        ):
            issues.append(
                "Large parameter space may result in very long execution times",
            )

        # Check for unrealistic allele frequencies
        if any(af > 0.5 for af in sim_config.allele_fractions):
            issues.append(
                "Allele frequencies above 0.5 may be unrealistic for typical MRD scenarios",
            )

        return issues

    @staticmethod
    def _validate_fastq_config(fastq_config: FASTQConfig) -> list[str]:
        """Validate FASTQ configuration for potential issues."""
        issues = []

        # Check if file exists
        if not Path(fastq_config.input_path).exists():
            issues.append(f"FASTQ file not found: {fastq_config.input_path}")

        return issues

    @staticmethod
    def suggest_optimizations(config: PipelineConfig) -> list[str]:
        """Suggest configuration optimizations."""
        suggestions = []

        if config.simulation:
            # Suggest reducing bootstrap iterations for faster runs
            if config.simulation.n_bootstrap > 1000:
                suggestions.append("Consider reducing n_bootstrap for faster execution")

            # Suggest appropriate allele frequency ranges
            if len(config.simulation.allele_fractions) > 5:
                suggestions.append(
                    "Consider reducing the number of allele fractions for faster analysis",
                )

        return suggestions


# Predefined configuration templates
class PredefinedTemplates:
    """Predefined configuration templates for common use cases."""

    @staticmethod
    def get_smoke_test_template() -> dict[str, Any]:
        """Get smoke test configuration template."""
        return {
            "template_name": "smoke_test",
            "description": "Minimal configuration for quick smoke testing",
            "base_config": {
                "umi": {
                    "min_family_size": 3,
                    "max_family_size": 1000,
                    "quality_threshold": 20,
                    "consensus_threshold": 0.6,
                },
                "stats": {
                    "test_type": "poisson",
                    "alpha": 0.05,
                    "fdr_method": "benjamini_hochberg",
                },
                "lod": {"detection_threshold": 0.95, "confidence_level": 0.95},
                "simulation": {
                    "allele_fractions": [0.01, 0.001, 0.0001],
                    "umi_depths": [1000, 5000],
                    "n_replicates": 10,
                    "n_bootstrap": 100,
                },
            },
            "tags": ["smoke_test", "quick", "minimal"],
            "version": "2.0.0",
        }

    @staticmethod
    def get_production_template() -> dict[str, Any]:
        """Get production-ready configuration template."""
        return {
            "template_name": "production",
            "description": "Production configuration with comprehensive validation",
            "base_config": {
                "umi": {
                    "min_family_size": 5,
                    "max_family_size": 10000,
                    "quality_threshold": 25,
                    "consensus_threshold": 0.7,
                },
                "stats": {
                    "test_type": "poisson",
                    "alpha": 0.01,
                    "fdr_method": "benjamini_hochberg",
                },
                "lod": {"detection_threshold": 0.95, "confidence_level": 0.99},
                "simulation": {
                    "allele_fractions": [0.1, 0.01, 0.001, 0.0001, 0.00001],
                    "umi_depths": [1000, 5000, 10000, 25000],
                    "n_replicates": 50,
                    "n_bootstrap": 1000,
                },
            },
            "tags": ["production", "comprehensive", "high_accuracy"],
            "version": "2.0.0",
        }

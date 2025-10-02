"""Configuration models for the MRD pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, ValidationError, field_validator


class SimulationSettings(BaseModel):
    allele_fractions: list[float] = Field(default_factory=lambda: [0.02, 0.01, 0.005, 0.002, 0.001])
    umi_depths: list[int] = Field(default_factory=lambda: [2000, 5000, 10000])
    n_replicates: int = 12
    n_bootstrap: int = 200
    controls: int = 6
    umi_per_variant: int = 48
    umi_family_min: int = 6
    umi_family_mean: int = 10
    
    # Backward compatibility
    @property
    def replicates(self) -> int:
        return self.n_replicates

    @field_validator("allele_fractions")
    @classmethod
    def _check_fractions(cls, values: list[float]) -> list[float]:
        if not values:
            msg = "allele_fractions must not be empty"
            raise ValueError(msg)
        if any(v <= 0 or v >= 1 for v in values):
            msg = "allele_fractions must be within (0, 1)"
            raise ValueError(msg)
        return values

    @field_validator("umi_depths")
    @classmethod
    def _check_depths(cls, values: list[int]) -> list[int]:
        if not values:
            msg = "umi_depths must not be empty"
            raise ValueError(msg)
        if any(v <= 0 for v in values):
            msg = "umi_depths must be positive"
            raise ValueError(msg)
        return values


class UMISettings(BaseModel):
    min_family_size: int = 5
    consensus_threshold: float = Field(default=0.7, ge=0.0, le=1.0)


class ErrorModelSettings(BaseModel):
    alpha_prior: float = Field(default=1.0, gt=0.0)
    beta_prior: float = Field(default=1000.0, gt=0.0)
    bootstrap_samples: int = Field(default=200, ge=10, le=1000)
    ci_level: float = Field(default=0.95, gt=0.5, lt=1.0)


class StatsSettings(BaseModel):
    alpha: float = Field(default=0.05, gt=0.0, lt=1.0)
    test_type: str = Field(default="poisson")
    fdr_method: str = Field(default="benjamini_hochberg")


class LODSettings(BaseModel):
    detection_threshold: float = Field(default=0.95, gt=0.0, le=1.0)
    confidence_level: float = Field(default=0.95, gt=0.0, le=1.0)


class CallSettings(BaseModel):
    pvalue_threshold: float = Field(default=1e-2, gt=0.0, lt=1.0)
    calibration_bins: int = Field(default=10, ge=3, le=50)


class ReportSettings(BaseModel):
    template: str | None = None
    include_plots: bool = True


class PipelineConfig(BaseModel):
    run_id: str = "run"
    simulation: SimulationSettings = Field(default_factory=SimulationSettings)
    umi: UMISettings = Field(default_factory=UMISettings)
    error_model: ErrorModelSettings = Field(default_factory=ErrorModelSettings)
    stats: StatsSettings = Field(default_factory=StatsSettings)
    lod: LODSettings = Field(default_factory=LODSettings)
    call: CallSettings = Field(default_factory=CallSettings)
    report: ReportSettings = Field(default_factory=ReportSettings)

    model_config = {
        "extra": "ignore",
    }


class ConfigError(RuntimeError):
    """Raised when configuration loading fails."""


def load_config(path: str | Path | None) -> PipelineConfig:
    """Load configuration from YAML or return defaults when not provided."""

    if path is None:
        return PipelineConfig()

    config_path = Path(path)
    if not config_path.exists():
        msg = f"configuration file not found: {config_path}"
        raise ConfigError(msg)

    try:
        data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:  # pragma: no cover - depends on parser internals
        raise ConfigError(f"invalid YAML in {config_path}: {exc}") from exc

    try:
        return PipelineConfig.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(str(exc)) from exc


def dump_config(config: PipelineConfig) -> dict[str, Any]:
    """Return a JSON-serialisable view of the configuration."""

    return config.model_dump(mode="json")


def create_default_config(template: str = "default") -> PipelineConfig:
    """Create a default configuration with optional template variations.
    
    Args:
        template: Configuration template ("default", "small", "large")
        
    Returns:
        PipelineConfig instance
    """
    config = PipelineConfig()
    
    if template == "small":
        # Fast configuration for testing
        config.simulation.n_replicates = 10
        config.simulation.n_bootstrap = 50
        config.simulation.allele_fractions = [0.01, 0.005, 0.001]
        config.simulation.umi_depths = [1000, 2000]
        config.error_model.bootstrap_samples = 50
    elif template == "large":
        # Comprehensive configuration for research
        config.simulation.n_replicates = 1000
        config.simulation.n_bootstrap = 1000
        config.simulation.allele_fractions = [0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0001]
        config.simulation.umi_depths = [1000, 2000, 5000, 10000, 20000, 50000]
        config.error_model.bootstrap_samples = 500
    # "default" template uses the class defaults
    
    return config

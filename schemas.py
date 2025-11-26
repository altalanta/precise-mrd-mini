from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


# Pydantic models for API requests/responses
class PipelineConfigRequest(BaseModel):
    """Request model for pipeline configuration."""

    run_id: str = Field(..., description="Unique run identifier")
    seed: int = Field(7, description="Random seed for reproducibility")
    config_override: str | None = Field(
        None, description="Custom configuration as YAML string"
    )
    use_parallel: bool = Field(False, description="Enable parallel processing")
    use_ml_calling: bool = Field(False, description="Enable ML-based variant calling")
    use_deep_learning: bool = Field(
        False, description="Enable deep learning variant calling"
    )
    ml_model_type: str = Field("ensemble", description="ML model type")
    dl_model_type: str = Field("cnn_lstm", description="Deep learning model type")


class JobStatus(BaseModel):
    """Job status response model."""

    job_id: str
    status: str  # pending, running, completed, failed
    progress: float
    start_time: datetime | None
    end_time: datetime | None = None
    error_message: str | None = None
    results: dict[str, Any] | None = None


class JobListResponse(BaseModel):
    """Response model for listing jobs."""

    jobs: list[JobStatus]


class ValidationResponse(BaseModel):
    """Response model for configuration validation."""

    is_valid: bool
    issues: list[str]
    warnings: list[str]
    suggestions: list[str]
    estimated_runtime_minutes: float
    config_hash: str


class TemplateListResponse(BaseModel):
    """Response model for listing configuration templates."""

    templates: list[dict[str, Any]]


class ConfigFromTemplateResponse(BaseModel):
    """Response model for creating a configuration from a template."""

    config_yaml: str
    config_summary: dict[str, Any]


class PipelineResults(BaseModel):
    """Pipeline execution results."""

    job_id: str
    run_id: str
    config_hash: str
    status: str
    metrics: dict[str, Any]
    artifacts: dict[str, str]
    run_context: dict[str, Any]


class ServiceStatus(BaseModel):
    """Health status of a single downstream service."""

    name: str = Field(
        ..., description="Name of the service (e.g., 'database', 'redis')."
    )
    status: str = Field(..., description="Service status ('ok' or 'error').")
    message: str | None = Field(
        None, description="Additional details about the service status."
    )


class HealthStatus(BaseModel):
    """Overall health status of the API."""

    status: str = Field(..., description="Overall status ('ok' or 'error').")
    services: list[ServiceStatus] = Field(
        ..., description="Status of individual downstream services."
    )

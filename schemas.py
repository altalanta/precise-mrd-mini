from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from .enums import (
    DLModelType,
    HealthStatusEnum,
    JobStatusEnum,
    MLModelType,
)


# Pydantic models for API requests/responses
class PipelineConfigRequest(BaseModel):
    """Request model for pipeline configuration."""

    run_id: str = Field(..., description="Unique run identifier")
    seed: int = Field(7, description="Random seed for reproducibility")
    config_override: str | None = Field(
        None,
        description="Custom configuration as YAML string",
    )
    use_parallel: bool = Field(False, description="Enable parallel processing")
    use_ml_calling: bool = Field(False, description="Enable ML-based variant calling")
    use_deep_learning: bool = Field(
        False,
        description="Enable deep learning variant calling",
    )
    ml_model_type: MLModelType = Field("ensemble", description="ML model type")
    dl_model_type: DLModelType = Field(
        "cnn_lstm",
        description="Deep learning model type",
    )


class JobStatus(BaseModel):
    """Job status response model."""

    job_id: str
    status: JobStatusEnum | str = Field(
        ...,
        description="Job status: pending, queued, running, completed, failed, or cancelled",
    )
    progress: float = Field(..., ge=0.0, le=1.0, description="Progress from 0.0 to 1.0")
    start_time: datetime | None = None
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
        ...,
        description="Name of the service (e.g., 'database', 'redis').",
    )
    status: HealthStatusEnum | str = Field(
        ...,
        description="Service status ('ok', 'error', or 'degraded').",
    )
    message: str | None = Field(
        None,
        description="Additional details about the service status.",
    )
    response_time_ms: float | None = Field(
        None,
        description="Response time of the health check in milliseconds.",
    )


class HealthStatus(BaseModel):
    """Overall health status of the API."""

    status: HealthStatusEnum | str = Field(
        ...,
        description="Overall status ('ok', 'error', or 'degraded').",
    )
    version: str = Field(..., description="API version.")
    services: list[ServiceStatus] = Field(
        ...,
        description="Status of individual downstream services.",
    )

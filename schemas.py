
from __future__ import annotations
from typing import Dict, List, Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field


# Pydantic models for API requests/responses
class PipelineConfigRequest(BaseModel):
    """Request model for pipeline configuration."""
    run_id: str = Field(..., description="Unique run identifier")
    seed: int = Field(7, description="Random seed for reproducibility")
    config_override: Optional[str] = Field(None, description="Custom configuration as YAML string")
    use_parallel: bool = Field(False, description="Enable parallel processing")
    use_ml_calling: bool = Field(False, description="Enable ML-based variant calling")
    use_deep_learning: bool = Field(False, description="Enable deep learning variant calling")
    ml_model_type: str = Field("ensemble", description="ML model type")
    dl_model_type: str = Field("cnn_lstm", description="Deep learning model type")


class JobStatus(BaseModel):
    """Job status response model."""
    job_id: str
    status: str  # pending, running, completed, failed
    progress: float
    start_time: Optional[datetime]
    end_time: Optional[datetime] = None
    error_message: Optional[str] = None
    results: Optional[Dict[str, Any]] = None


class JobListResponse(BaseModel):
    """Response model for listing jobs."""
    jobs: List[JobStatus]


class ValidationResponse(BaseModel):
    """Response model for configuration validation."""
    is_valid: bool
    issues: List[str]
    warnings: List[str]
    suggestions: List[str]
    estimated_runtime_minutes: float
    config_hash: str


class TemplateListResponse(BaseModel):
    """Response model for listing configuration templates."""
    templates: List[Dict[str, Any]]


class ConfigFromTemplateResponse(BaseModel):
    """Response model for creating a configuration from a template."""
    config_yaml: str
    config_summary: Dict[str, Any]


class PipelineResults(BaseModel):
    """Pipeline execution results."""
    job_id: str
    run_id: str
    config_hash: str
    status: str
    metrics: Dict[str, Any]
    artifacts: Dict[str, str]
    run_context: Dict[str, Any]


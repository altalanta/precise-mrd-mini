"""REST API for the Precise MRD pipeline with cloud-native deployment support."""

from __future__ import annotations

import asyncio
import json
import uuid
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import shutil

from fastapi import FastAPI, HTTPException, BackgroundTasks, File, UploadFile, Form, Query
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel, Field
import pandas as pd

from .config import PipelineConfig, load_config, dump_config, ConfigValidator
from .simulate import simulate_reads
from .collapse import collapse_umis
from .call import call_mrd
from .error_model import fit_error_model
from .metrics import calculate_metrics
from .reporting import render_report
from .determinism_utils import set_global_seed
from .utils import PipelineIO


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
    end_time: Optional[datetime]
    error_message: Optional[str]
    results: Optional[Dict[str, Any]]


class PipelineResults(BaseModel):
    """Pipeline execution results."""
    job_id: str
    run_id: str
    config_hash: str
    status: str
    metrics: Dict[str, Any]
    artifacts: Dict[str, str]
    run_context: Dict[str, Any]


# Global job management
class JobManager:
    """Manages asynchronous pipeline jobs."""

    def __init__(self):
        """Initialize job manager."""
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self.results_dir = Path("api_results")
        self.results_dir.mkdir(exist_ok=True)

    def create_job(self, config_request: PipelineConfigRequest) -> str:
        """Create a new pipeline job."""
        job_id = str(uuid.uuid4())

        self.jobs[job_id] = {
            'job_id': job_id,
            'status': 'pending',
            'progress': 0.0,
            'start_time': None,
            'end_time': None,
            'error_message': None,
            'results': None,
            'config_request': config_request.dict()
        }

        return job_id

    def update_job_status(self, job_id: str, status: str, progress: float = None, error: str = None):
        """Update job status and progress."""
        if job_id not in self.jobs:
            raise ValueError(f"Job {job_id} not found")

        job = self.jobs[job_id]
        job['status'] = status

        if progress is not None:
            job['progress'] = progress

        if error:
            job['error_message'] = error
            job['status'] = 'failed'

        if status == 'running' and job['start_time'] is None:
            job['start_time'] = datetime.now()

        if status in ['completed', 'failed'] and job['end_time'] is None:
            job['end_time'] = datetime.now()

    def get_job_status(self, job_id: str) -> JobStatus:
        """Get job status."""
        if job_id not in self.jobs:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

        job = self.jobs[job_id]
        return JobStatus(
            job_id=job['job_id'],
            status=job['status'],
            progress=job['progress'],
            start_time=job['start_time'],
            end_time=job['end_time'],
            error_message=job['error_message'],
            results=job.get('results')
        )

    def set_job_results(self, job_id: str, results: Dict[str, Any]):
        """Set job results."""
        if job_id in self.jobs:
            self.jobs[job_id]['results'] = results

    def cleanup_old_jobs(self, max_age_hours: int = 24):
        """Clean up old completed/failed jobs."""
        cutoff_time = datetime.now().timestamp() - (max_age_hours * 3600)

        to_remove = []
        for job_id, job in self.jobs.items():
            if job['end_time'] and job['end_time'].timestamp() < cutoff_time:
                to_remove.append(job_id)

        for job_id in to_remove:
            del self.jobs[job_id]


# Global job manager instance
job_manager = JobManager()


# FastAPI application
app = FastAPI(
    title="Precise MRD Pipeline API",
    description="REST API for the Precise MRD (Minimal Residual Disease) detection pipeline",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for web integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Precise MRD Pipeline API",
        "version": "2.0.0",
        "description": "REST API for MRD detection with parallel processing, ML, and deep learning",
        "endpoints": {
            "submit_job": "/submit",
            "job_status": "/status/{job_id}",
            "job_results": "/results/{job_id}",
            "health": "/health"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0"
    }


@app.post("/submit", response_model=JobStatus)
async def submit_pipeline_job(
    background_tasks: BackgroundTasks,
    run_id: str = Form(..., description="Unique run identifier"),
    seed: int = Form(7, description="Random seed for reproducibility"),
    config_override: Optional[str] = Form(None, description="Custom configuration as YAML string"),
    use_parallel: bool = Form(False, description="Enable parallel processing"),
    use_ml_calling: bool = Form(False, description="Enable ML-based variant calling"),
    use_deep_learning: bool = Form(False, description="Enable deep learning variant calling"),
    ml_model_type: str = Form("ensemble", description="ML model type"),
    dl_model_type: str = Form("cnn_lstm", description="Deep learning model type")
):
    """Submit a pipeline job for execution."""
    try:
        # Create configuration request
        config_request = PipelineConfigRequest(
            run_id=run_id,
            seed=seed,
            config_override=config_override,
            use_parallel=use_parallel,
            use_ml_calling=use_ml_calling,
            use_deep_learning=use_deep_learning,
            ml_model_type=ml_model_type,
            dl_model_type=dl_model_type
        )

        # Create job
        job_id = job_manager.create_job(config_request)
        job_manager.update_job_status(job_id, 'running', 0.0)

        # Run pipeline in background
        background_tasks.add_task(run_pipeline_job, job_id, config_request)

        return job_manager.get_job_status(job_id)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to submit job: {str(e)}")


@app.get("/status/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    """Get the status of a pipeline job."""
    return job_manager.get_job_status(job_id)


@app.get("/results/{job_id}")
async def get_job_results(job_id: str):
    """Get the results of a completed pipeline job."""
    status = job_manager.get_job_status(job_id)

    if status.status == 'pending':
        raise HTTPException(status_code=404, detail="Job not found or not yet submitted")

    if status.status == 'running':
        raise HTTPException(status_code=202, detail="Job is still running")

    if status.status == 'failed':
        raise HTTPException(status_code=500, detail=f"Job failed: {status.error_message}")

    if status.results is None:
        raise HTTPException(status_code=404, detail="Results not available")

    return JSONResponse(content=status.results)


@app.get("/download/{job_id}/{artifact_type}")
async def download_artifact(job_id: str, artifact_type: str):
    """Download a specific artifact from a completed job."""
    status = job_manager.get_job_status(job_id)

    if status.status != 'completed' or status.results is None:
        raise HTTPException(status_code=404, detail="Artifact not available")

    artifacts = status.results.get('artifacts', {})
    if artifact_type not in artifacts:
        raise HTTPException(status_code=404, detail=f"Artifact type '{artifact_type}' not found")

    artifact_path = artifacts[artifact_type]
    if not Path(artifact_path).exists():
        raise HTTPException(status_code=404, detail="Artifact file not found")

    return FileResponse(
        artifact_path,
        media_type='application/octet-stream',
        filename=f"{job_id}_{artifact_type}"
    )


@app.get("/jobs")
async def list_jobs(limit: int = Query(50, description="Maximum number of jobs to return")):
    """List recent jobs."""
    job_manager.cleanup_old_jobs()

    # Get recent jobs (most recent first)
    recent_jobs = []
    for job_id in sorted(job_manager.jobs.keys(), reverse=True):
        job = job_manager.jobs[job_id]
        recent_jobs.append({
            'job_id': job['job_id'],
            'status': job['status'],
            'run_id': job['config_request'].get('run_id', 'unknown'),
            'start_time': job['start_time'].isoformat() if job['start_time'] else None,
            'end_time': job['end_time'].isoformat() if job['end_time'] else None
        })

        if len(recent_jobs) >= limit:
            break

    return {"jobs": recent_jobs}


async def run_pipeline_job(job_id: str, config_request: PipelineConfigRequest):
    """Run the MRD pipeline for a job (background task)."""
    try:
        # Update progress
        job_manager.update_job_status(job_id, 'running', 10.0)

        # Load or create configuration
        if config_request.config_override:
            # Use custom configuration
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
                f.write(config_request.config_override)
                config_path = f.name

            config = load_config(config_path)
            # Clean up temp file
            Path(config_path).unlink()
        else:
            # Use default configuration
            config = PipelineConfig(
                run_id=config_request.run_id,
                seed=config_request.seed,
                simulation=None,  # Will be auto-generated for API
                umi=None,  # Use defaults
                stats=None,  # Use defaults
                lod=None,  # Use defaults
            )

        job_manager.update_job_status(job_id, 'running', 20.0)

        # Set up output directories
        job_dir = job_manager.results_dir / job_id
        job_dir.mkdir(exist_ok=True)

        reports_dir = job_dir / "reports"
        reports_dir.mkdir(exist_ok=True)

        data_dir = job_dir / "data"
        data_dir.mkdir(exist_ok=True)

        # Run pipeline stages
        rng = set_global_seed(config_request.seed, deterministic_ops=True)

        job_manager.update_job_status(job_id, 'running', 30.0)

        # Simulate reads (if no FASTQ data provided)
        reads_path = data_dir / "simulated_reads.parquet"
        reads_df = simulate_reads(config, rng, output_path=str(reads_path))

        job_manager.update_job_status(job_id, 'running', 50.0)

        # Collapse UMIs
        collapsed_path = data_dir / "collapsed_umis.parquet"
        collapsed_df = collapse_umis(
            reads_df, config, rng,
            output_path=str(collapsed_path),
            use_parallel=config_request.use_parallel
        )

        job_manager.update_job_status(job_id, 'running', 70.0)

        # Fit error model
        error_model_path = data_dir / "error_model.parquet"
        error_model_df = fit_error_model(
            collapsed_df, config, rng,
            output_path=str(error_model_path)
        )

        job_manager.update_job_status(job_id, 'running', 80.0)

        # Call variants
        calls_path = data_dir / "mrd_calls.parquet"
        calls_df = call_mrd(
            collapsed_df, error_model_df, config, rng,
            output_path=str(calls_path),
            use_ml_calling=config_request.use_ml_calling,
            ml_model_type=config_request.ml_model_type,
            use_deep_learning=config_request.use_deep_learning,
            dl_model_type=config_request.dl_model_type
        )

        job_manager.update_job_status(job_id, 'running', 90.0)

        # Calculate metrics
        metrics = calculate_metrics(calls_df, rng)

        # Generate run context
        run_context = {
            "schema_version": "2.0.0",
            "job_id": job_id,
            "run_id": config_request.run_id,
            "seed": config_request.seed,
            "timestamp": datetime.now().isoformat(),
            "config_hash": config.config_hash(),
            "api_version": "2.0.0",
            "processing_options": {
                "parallel": config_request.use_parallel,
                "ml_calling": config_request.use_ml_calling,
                "ml_model_type": config_request.ml_model_type,
                "deep_learning": config_request.use_deep_learning,
                "dl_model_type": config_request.dl_model_type
            }
        }

        # Save results
        metrics_path = reports_dir / "metrics.json"
        context_path = reports_dir / "run_context.json"
        report_path = reports_dir / "auto_report.html"

        PipelineIO.save_json(metrics, str(metrics_path))
        PipelineIO.save_json(run_context, str(context_path))

        # Generate HTML report
        render_report(calls_df, metrics, config.to_dict(), run_context, str(report_path))

        # Create manifest
        manifest_path = reports_dir / "hash_manifest.txt"
        write_manifest([metrics_path, context_path, report_path], out_manifest=manifest_path)

        job_manager.update_job_status(job_id, 'completed', 100.0)

        # Store results
        results = {
            'job_id': job_id,
            'run_id': config_request.run_id,
            'config_hash': config.config_hash(),
            'status': 'completed',
            'metrics': metrics,
            'run_context': run_context,
            'artifacts': {
                'reads': str(reads_path),
                'collapsed': str(collapsed_path),
                'error_model': str(error_model_path),
                'calls': str(calls_path),
                'metrics': str(metrics_path),
                'run_context': str(context_path),
                'report': str(report_path),
                'manifest': str(manifest_path)
            }
        }

        job_manager.set_job_results(job_id, results)

    except Exception as e:
        job_manager.update_job_status(job_id, 'failed', error=str(e))
        raise


@app.post("/validate-config")
async def validate_configuration(config_yaml: str = Form(...)):
    """Validate a pipeline configuration."""
    try:
        # Create temporary config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(config_yaml)
            config_path = f.name

        try:
            config = load_config(config_path)
            validation_result = ConfigValidator.validate_config(config)

            return {
                'is_valid': validation_result['is_valid'],
                'issues': validation_result['issues'],
                'warnings': validation_result['warnings'],
                'suggestions': validation_result['suggestions'],
                'estimated_runtime_minutes': validation_result['estimated_runtime_minutes'],
                'config_hash': validation_result['config_hash']
            }

        finally:
            Path(config_path).unlink()

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Configuration validation failed: {str(e)}")


@app.get("/config-templates")
async def get_config_templates():
    """Get available configuration templates."""
    from .config import PredefinedTemplates

    templates = [
        PredefinedTemplates.get_smoke_test_template(),
        PredefinedTemplates.get_production_template()
    ]

    return {"templates": templates}


@app.post("/config-from-template")
async def create_config_from_template(
    template_name: str = Form(...),
    run_id: str = Form(...),
    seed: int = Form(7)
):
    """Create configuration from template."""
    from .config import PredefinedTemplates

    template = None
    if template_name == "smoke_test":
        template = PredefinedTemplates.get_smoke_test_template()
    elif template_name == "production":
        template = PredefinedTemplates.get_production_template()
    else:
        raise HTTPException(status_code=400, detail=f"Unknown template: {template_name}")

    config = PipelineConfig.from_template(template, run_id)
    config.seed = seed

    return {
        'config_yaml': dump_config(config, None),  # Returns YAML string
        'config_summary': {
            'run_id': config.run_id,
            'seed': config.seed,
            'config_version': config.config_version,
            'template': template_name
        }
    }


def create_api_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    return app


def run_api_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """Run the API server."""
    uvicorn.run(
        "precise_mrd.api:app",
        host=host,
        port=port,
        reload=reload,
        access_log=True
    )


if __name__ == "__main__":
    run_api_server()

"""REST API for the Precise MRD pipeline with cloud-native deployment support."""

from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import (
    Depends,
    FastAPI,
    Form,
    HTTPException,
    Query,
    Request,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from .celery_app import celery_app
from .config import ConfigValidator, PipelineConfig, dump_config, load_config
from .database import get_db, init_db
from .exceptions import ConfigurationError
from .job_manager import JobManager, get_job_manager
from .logging_config import get_logger, setup_logging
from .schemas import (
    ConfigFromTemplateResponse,
    HealthStatus,
    JobListResponse,
    JobStatus,
    PipelineConfigRequest,
    PipelineResults,
    ServiceStatus,
    TemplateListResponse,
    ValidationResponse,
)
from .tasks import run_pipeline_task
from .tracing import instrument_fastapi_app, setup_telemetry

log = get_logger(__name__)


class ConnectionManager:
    """Manages active WebSocket connections for real-time updates."""

    def __init__(self):
        self.active_connections: dict[str, list[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, job_id: str):
        """Accepts a new WebSocket connection and associates it with a job_id."""
        await websocket.accept()
        if job_id not in self.active_connections:
            self.active_connections[job_id] = []
        self.active_connections[job_id].append(websocket)
        log.info(f"WebSocket connected for job_id: {job_id}")

    def disconnect(self, websocket: WebSocket, job_id: str):
        """Disconnects a WebSocket and removes it from the active connections."""
        if job_id in self.active_connections:
            self.active_connections[job_id].remove(websocket)
            if not self.active_connections[job_id]:
                del self.active_connections[job_id]
        log.info(f"WebSocket disconnected for job_id: {job_id}")

    async def broadcast(self, job_id: str, message: dict[str, Any]):
        """Broadcasts a message to all clients connected for a specific job_id."""
        if job_id in self.active_connections:
            for connection in self.active_connections[job_id]:
                await connection.send_json(message)


# Create a single instance of the connection manager to be used across the application
manager = ConnectionManager()


# FastAPI application
app = FastAPI(
    title="Precise MRD Pipeline API",
    description="""
A robust, scalable, and reproducible REST API for the Precise MRD (Minimal Residual Disease)
detection pipeline.
    """,
    version="0.2.0",
    contact={
        "name": "Development Team",
        "url": "https://github.com/your-repo/precise-mrd-mini",
        "email": "dev@example.com",
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT",
    },
)


@app.on_event("startup")
async def startup_event():
    """Configure logging and telemetry on application startup."""
    setup_logging()
    setup_telemetry()
    instrument_fastapi_app(app)
    init_db()
    log.info("Application startup complete. Logging, Telemetry, and DB configured.")


# CORS middleware for web integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def structured_logging_middleware(request: Request, call_next):
    """
    Middleware to add structured logging for every API request.
    """
    start_time = time.time()

    response = await call_next(request)

    process_time = (time.time() - start_time) * 1000

    log.info(
        "api_request",
        method=request.method,
        path=request.url.path,
        query_params=str(request.query_params),
        status_code=response.status_code,
        processing_time_ms=f"{process_time:.2f}",
        client_host=request.client.host,
    )

    return response


@app.get(
    "/health",
    response_model=HealthStatus,
    summary="Get Service Health Status",
    description="""
    Performs a health check on the API and its downstream services,
    including the database and Redis cache.
    """,
    tags=["Health"],
)
def get_health_status(db: Session = Depends(get_db)):
    """Check the health of the service and its dependencies."""
    db_status = "ok"
    db_message = "Database connection successful."
    try:
        # Perform a simple query to check DB connection
        db.execute("SELECT 1")
    except Exception as e:
        db_status = "error"
        db_message = f"Database connection failed: {e}"

    redis_status = "ok"
    redis_message = "Redis connection successful."
    try:
        # Perform a simple ping to check Redis connection
        celery_app.backend.client.ping()
    except Exception as e:
        redis_status = "error"
        redis_message = f"Redis connection failed: {e}"

    overall_status = "ok" if db_status == "ok" and redis_status == "ok" else "error"

    return HealthStatus(
        status=overall_status,
        services=[
            ServiceStatus(name="database", status=db_status, message=db_message),
            ServiceStatus(name="redis", status=redis_status, message=redis_message),
        ],
    )


@app.websocket("/ws/status/{job_id}", name="job_status_ws")
async def websocket_endpoint(websocket: WebSocket, job_id: str):
    """
    WebSocket endpoint to stream real-time status updates for a given job.
    """
    await manager.connect(websocket, job_id)
    try:
        # Keep the connection alive
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket, job_id)


@app.post(
    "/submit",
    response_model=JobStatus,
    summary="Submit a New Pipeline Job",
    description="""
    Submits a new MRD pipeline job for asynchronous execution.

    The job is added to the Celery queue and will be picked up by an available worker.
    You can specify the run configuration using a combination of form fields and an optional
    YAML override.
    """,
    tags=["Jobs"],
)
async def submit_pipeline_job(
    job_manager: JobManager = Depends(get_job_manager),
    run_id: str = Form(
        ...,
        description="A unique identifier for this specific run, e.g., 'patient_123_run_1'.",
    ),
    seed: int = Form(
        7,
        description="The random seed for ensuring reproducibility of the simulation and analysis.",
    ),
    config_override: str | None = Form(
        None,
        description="A complete YAML configuration string to override all default parameters.",
    ),
    use_parallel: bool = Form(
        False,
        description="Enable parallel processing for performance improvement on multi-core systems.",
    ),
    use_ml_calling: bool = Form(
        False, description="Enable the machine learning-based variant calling model."
    ),
    use_deep_learning: bool = Form(
        False, description="Enable the deep learning (CNN-LSTM) variant calling model."
    ),
    ml_model_type: str = Form(
        "ensemble",
        description="Specify the type of ML model to use (e.g., 'ensemble', 'random_forest').",
    ),
    dl_model_type: str = Form(
        "cnn_lstm",
        description="Specify the type of deep learning model to use (e.g., 'cnn_lstm', 'transformer').",
    ),
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
            dl_model_type=dl_model_type,
        )

        # Create job
        job = job_manager.create_job(config_request)
        # Dispatch the task to Celery
        run_pipeline_task.delay(job.id, config_request.dict())
        job_manager.update_job_status(job.id, "queued", 0.0)

        return JobStatus(
            job_id=job.id, status="queued", progress=0.0, start_time=job.created_at
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to submit job: {str(e)}") from e
    except ConfigurationError as e:
        raise HTTPException(
            status_code=400, detail=f"Invalid configuration provided: {e}"
        ) from e


@app.get(
    "/status/{job_id}",
    response_model=JobStatus,
    summary="Get Job Status",
    description="Retrieves the current status, progress, and metadata for a specific job.",
    tags=["Jobs"],
)
async def get_job_status(
    job_id: str, job_manager: JobManager = Depends(get_job_manager)
):
    """Get the status of a pipeline job."""
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return JobStatus(
        job_id=job.id,
        status=job.status,
        progress=job.progress,
        start_time=job.created_at,
        end_time=job.updated_at,
        results=job.get_results(),
    )


@app.get(
    "/results/{job_id}",
    response_model=PipelineResults,
    summary="Get Job Results",
    description="""
    Retrieves the full results of a completed job, including metrics, run context,
    and paths to all generated artifacts. Returns an error if the job is not yet completed.
    """,
    tags=["Jobs"],
)
async def get_job_results(
    job_id: str, job_manager: JobManager = Depends(get_job_manager)
):
    """Get the results of a completed pipeline job."""
    job = job_manager.get_job(job_id)

    if not job or job.status == "pending":
        raise HTTPException(
            status_code=404, detail="Job not found or not yet submitted"
        )

    if job.status == "running":
        raise HTTPException(status_code=202, detail="Job is still running")

    if job.status == "failed":
        raise HTTPException(status_code=500, detail="Job failed")

    results = job.get_results()
    if results is None:
        raise HTTPException(status_code=404, detail="Results not available")

    return results


@app.get(
    "/download/{job_id}/{artifact_type}",
    summary="Download a Job Artifact",
    description="Downloads a specific data artifact (e.g., reads, calls, report) from a completed job.",
    tags=["Artifacts"],
)
async def download_artifact(
    job_id: str, artifact_type: str, job_manager: JobManager = Depends(get_job_manager)
):
    """Download a specific artifact from a completed job."""
    job = job_manager.get_job(job_id)

    if not job or job.status != "completed" or not job.results:
        raise HTTPException(status_code=404, detail="Artifact not available")

    results = job.get_results()
    artifacts = results.get("artifacts", {}) if results else {}
    if artifact_type not in artifacts:
        raise HTTPException(
            status_code=404, detail=f"Artifact type '{artifact_type}' not found"
        )

    artifact_path = artifacts[artifact_type]
    if not Path(artifact_path).exists():
        raise HTTPException(status_code=404, detail="Artifact file not found")

    return FileResponse(
        artifact_path,
        media_type="application/octet-stream",
        filename=f"{job_id}_{artifact_type}",
    )


@app.get(
    "/jobs",
    response_model=JobListResponse,
    summary="List Recent Jobs",
    description="Retrieves a list of the most recent jobs, sorted by creation time.",
    tags=["Jobs"],
)
async def list_jobs(
    job_manager: JobManager = Depends(get_job_manager),
    limit: int = Query(50, description="The maximum number of jobs to return."),
):
    """List recent jobs."""
    jobs = job_manager.get_all_jobs(limit=limit)
    job_statuses = [
        JobStatus(
            job_id=job.id,
            status=job.status,
            progress=job.progress,
            start_time=job.created_at,
            end_time=job.updated_at,
        )
        for job in jobs
    ]
    return {"jobs": job_statuses}


@app.post(
    "/validate-config",
    response_model=ValidationResponse,
    summary="Validate a Configuration",
    description="""
    Validates a given YAML configuration string without submitting a job.

    This is useful for checking the correctness of a configuration and getting an estimate
    of the runtime before committing to a full pipeline run.
    """,
    tags=["Configuration"],
)
async def validate_configuration(
    config_yaml: str = Form(
        ..., description="The YAML configuration string to validate."
    ),
):
    """Validate a pipeline configuration."""
    try:
        # Create temporary config file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(config_yaml)
            config_path = f.name

        try:
            config = load_config(config_path)
            validation_result = ConfigValidator.validate_config(config)

            return validation_result

        finally:
            Path(config_path).unlink()

    except ConfigurationError as e:
        raise HTTPException(
            status_code=400, detail=f"Configuration validation failed: {e}"
        ) from e
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Configuration validation failed: {str(e)}"
        ) from e


@app.get(
    "/config-templates",
    response_model=TemplateListResponse,
    summary="Get Configuration Templates",
    description="Retrieves a list of available pre-defined configuration templates (e.g., 'smoke_test', 'production').",
    tags=["Configuration"],
)
async def get_config_templates():
    """Get available configuration templates."""
    from .config import PredefinedTemplates

    templates = [
        PredefinedTemplates.get_smoke_test_template(),
        PredefinedTemplates.get_production_template(),
    ]

    return {"templates": templates}


@app.post(
    "/config-from-template",
    response_model=ConfigFromTemplateResponse,
    summary="Create Configuration from Template",
    description="Generates a full YAML configuration string from a pre-defined template and a given run_id.",
    tags=["Configuration"],
)
async def create_config_from_template(
    template_name: str = Form(
        ..., description="The name of the template to use (e.g., 'smoke_test')."
    ),
    run_id: str = Form(
        ..., description="The unique run identifier to embed in the configuration."
    ),
    seed: int = Form(7, description="The random seed to embed in the configuration."),
):
    """Create configuration from template."""
    from .config import PredefinedTemplates

    template = None
    if template_name == "smoke_test":
        template = PredefinedTemplates.get_smoke_test_template()
    elif template_name == "production":
        template = PredefinedTemplates.get_production_template()
    else:
        raise HTTPException(
            status_code=400, detail=f"Unknown template: {template_name}"
        )
    try:
        config = PipelineConfig.from_template(template, run_id)
        config.seed = seed

        return {
            "config_yaml": dump_config(config, None),  # Returns YAML string
            "config_summary": {
                "run_id": config.run_id,
                "seed": config.seed,
                "config_version": config.config_version,
                "template": template_name,
            },
        }
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Failed to create config from template: {e}"
        ) from e


def create_api_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    return app


def run_api_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """Run the API server."""
    uvicorn.run(
        "precise_mrd.api:app", host=host, port=port, reload=reload, access_log=True
    )


if __name__ == "__main__":
    run_api_server()

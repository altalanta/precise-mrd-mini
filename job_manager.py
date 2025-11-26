import uuid
from typing import TYPE_CHECKING

from fastapi import Depends
from sqlalchemy.orm import Session

from .database import Job, get_db
from .schemas import PipelineConfigRequest

# This type hint helps with auto-completion, but we avoid a circular import
if TYPE_CHECKING:
    from .api import ConnectionManager


class JobManager:
    """Manages asynchronous pipeline jobs using a persistent database."""

    def __init__(self, db: Session, ws_manager: "ConnectionManager" = None):
        self.db = db
        self.ws_manager = ws_manager

    def create_job(self, config_request: PipelineConfigRequest) -> Job:
        """Create a new pipeline job in the database."""
        job_id = str(uuid.uuid4())
        new_job = Job(
            id=job_id, run_id=config_request.run_id, status="pending", progress=0.0
        )
        self.db.add(new_job)
        self.db.commit()
        self.db.refresh(new_job)
        return new_job

    def get_job(self, job_id: str) -> Job | None:
        """Get a job by its ID."""
        return self.db.query(Job).filter(Job.id == job_id).first()

    def update_job_status(
        self, job_id: str, status: str, progress: float = None, error: str = None
    ):
        """Update job status and progress, and broadcast the update."""
        job = self.get_job(job_id)
        if not job:
            raise ValueError(f"Job {job_id} not found")

        job.status = status
        if progress is not None:
            job.progress = progress

        if error:
            job.status = "failed"
            job.set_results({"error": error})

        self.db.commit()

        # Broadcast the update via WebSocket
        if self.ws_manager:
            import asyncio

            from .schemas import JobStatus

            job_status = JobStatus(
                job_id=job.id,
                status=job.status,
                progress=job.progress,
                start_time=job.created_at,
                end_time=job.updated_at,
                results=job.get_results(),
            )

            # Run the async broadcast function in the background
            asyncio.create_task(self.ws_manager.broadcast(job_id, job_status.dict()))

    def set_job_results(self, job_id: str, results_path: str):
        """Set job results from a directory path."""
        job = self.get_job(job_id)
        if job:
            # Create a serializable summary of results
            results_summary = {
                "results_path": results_path,
                "report_url": f"/download/{job_id}/report",
                "metrics_url": f"/download/{job_id}/metrics",
            }
            job.set_results(results_summary)
            self.db.commit()

    def get_all_jobs(self, skip: int = 0, limit: int = 100) -> list[Job]:
        """Get all jobs."""
        return (
            self.db.query(Job)
            .order_by(Job.created_at.desc())
            .offset(skip)
            .limit(limit)
            .all()
        )


def get_job_manager(db: Session = Depends(get_db)) -> JobManager:
    # The websocket manager is injected here from the global scope of the api module.
    # This is a simple approach for dependency injection without complex frameworks.
    from .api import manager

    return JobManager(db, manager)

"""Job management with both sync and async database support."""

import uuid
from typing import TYPE_CHECKING

from fastapi import Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session

from .database import Job, SessionLocal, get_async_db
from .enums import JobStatusEnum
from .schemas import PipelineConfigRequest

# This type hint helps with auto-completion, but we avoid a circular import
if TYPE_CHECKING:
    from .api import ConnectionManager


class JobManager:
    """
    Synchronous job manager for Celery tasks.

    Uses synchronous database sessions for background task processing.
    """

    def __init__(self, db: Session, ws_manager: "ConnectionManager" = None):
        self.db = db
        self.ws_manager = ws_manager

    def create_job(self, config_request: PipelineConfigRequest) -> Job:
        """Create a new pipeline job in the database."""
        job_id = str(uuid.uuid4())
        new_job = Job(
            id=job_id,
            run_id=config_request.run_id,
            status=JobStatusEnum.PENDING,
            progress=0.0,
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
            job.status = JobStatusEnum.FAILED
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


class AsyncJobManager:
    """
    Asynchronous job manager for FastAPI endpoints.

    Uses async database sessions to avoid blocking the event loop.
    """

    def __init__(self, db: AsyncSession, ws_manager: "ConnectionManager" = None):
        self.db = db
        self.ws_manager = ws_manager

    async def create_job(self, config_request: PipelineConfigRequest) -> Job:
        """Create a new pipeline job in the database."""
        job_id = str(uuid.uuid4())
        new_job = Job(
            id=job_id,
            run_id=config_request.run_id,
            status=JobStatusEnum.PENDING,
            progress=0.0,
        )
        self.db.add(new_job)
        await self.db.commit()
        await self.db.refresh(new_job)
        return new_job

    async def get_job(self, job_id: str) -> Job | None:
        """Get a job by its ID."""
        result = await self.db.execute(select(Job).filter(Job.id == job_id))
        return result.scalar_one_or_none()

    async def update_job_status(
        self, job_id: str, status: str, progress: float = None, error: str = None
    ):
        """Update job status and progress, and broadcast the update."""
        job = await self.get_job(job_id)
        if not job:
            raise ValueError(f"Job {job_id} not found")

        job.status = status
        if progress is not None:
            job.progress = progress

        if error:
            job.status = JobStatusEnum.FAILED
            job.set_results({"error": error})

        await self.db.commit()

        # Broadcast the update via WebSocket
        if self.ws_manager:
            from .schemas import JobStatus

            job_status = JobStatus(
                job_id=job.id,
                status=job.status,
                progress=job.progress,
                start_time=job.created_at,
                end_time=job.updated_at,
                results=job.get_results(),
            )

            await self.ws_manager.broadcast(job_id, job_status.dict())

    async def set_job_results(self, job_id: str, results_path: str):
        """Set job results from a directory path."""
        job = await self.get_job(job_id)
        if job:
            # Create a serializable summary of results
            results_summary = {
                "results_path": results_path,
                "report_url": f"/download/{job_id}/report",
                "metrics_url": f"/download/{job_id}/metrics",
            }
            job.set_results(results_summary)
            await self.db.commit()

    async def get_all_jobs(self, skip: int = 0, limit: int = 100) -> list[Job]:
        """Get all jobs."""
        result = await self.db.execute(
            select(Job).order_by(Job.created_at.desc()).offset(skip).limit(limit)
        )
        return list(result.scalars().all())


def get_sync_job_manager() -> JobManager:
    """
    Get a synchronous job manager for Celery tasks.

    Creates its own database session that must be closed by the caller.
    """
    db = SessionLocal()
    return JobManager(db)


async def get_async_job_manager(
    db: AsyncSession = Depends(get_async_db),
) -> AsyncJobManager:
    """
    FastAPI dependency that provides an async job manager.

    The database session is automatically managed by the dependency injection.
    """
    # The websocket manager is injected here from the global scope of the api module.
    from .api import manager

    return AsyncJobManager(db, manager)


import uuid
from typing import Dict, Any, Optional, List
from sqlalchemy.orm import Session
from fastapi import Depends

from .database import get_db, Job
from .schemas import PipelineConfigRequest


class JobManager:
    """Manages asynchronous pipeline jobs using a persistent database."""

    def __init__(self, db: Session):
        self.db = db

    def create_job(self, config_request: PipelineConfigRequest) -> Job:
        """Create a new pipeline job in the database."""
        job_id = str(uuid.uuid4())
        new_job = Job(
            id=job_id,
            run_id=config_request.run_id,
            status='pending',
            progress=0.0
        )
        self.db.add(new_job)
        self.db.commit()
        self.db.refresh(new_job)
        return new_job

    def get_job(self, job_id: str) -> Optional[Job]:
        """Get a job by its ID."""
        return self.db.query(Job).filter(Job.id == job_id).first()

    def update_job_status(self, job_id: str, status: str, progress: float = None, error: str = None):
        """Update job status and progress."""
        job = self.get_job(job_id)
        if not job:
            raise ValueError(f"Job {job_id} not found")

        job.status = status
        if progress is not None:
            job.progress = progress

        if error:
            job.status = 'failed'

        self.db.commit()

    def set_job_results(self, job_id: str, results: Dict[str, Any]):
        """Set job results."""
        job = self.get_job(job_id)
        if job:
            job.set_results(results)
            self.db.commit()

    def get_all_jobs(self, skip: int = 0, limit: int = 100) -> List[Job]:
        """Get all jobs."""
        return self.db.query(Job).order_by(Job.created_at.desc()).offset(skip).limit(limit).all()


def get_job_manager(db: Session = Depends(get_db)) -> JobManager:
    return JobManager(db)

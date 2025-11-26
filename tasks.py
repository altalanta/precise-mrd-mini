from .celery_app import celery_app
from .database import SessionLocal
from .job_manager import JobManager
from .logging_config import get_logger
from .pipeline_service import PipelineService
from .schemas import PipelineConfigRequest

log = get_logger(__name__)


@celery_app.task(bind=True)
def run_pipeline_task(self, job_id: str, config_request_dict: dict):
    """Celery task to run the MRD pipeline."""
    db = SessionLocal()
    job_manager = JobManager(db)
    pipeline_service = PipelineService()
    config_request = PipelineConfigRequest(**config_request_dict)

    log.info(f"Celery task {self.request.id} started for job_id: {job_id}")

    try:
        pipeline_service.run(
            job_id=job_id, config_request=config_request, job_manager=job_manager
        )
    except Exception as e:
        log.error(f"Celery task for job {job_id} failed", error=str(e), exc_info=True)
        job_manager.update_job_status(job_id, "failed", error=str(e))
        raise
    finally:
        db.close()

    log.info(f"Celery task {self.request.id} for job_id: {job_id} completed.")
    job = job_manager.get_job(job_id)
    return job.get_results() if job else None

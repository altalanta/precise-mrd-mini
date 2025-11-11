
import os
from celery import Celery

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

celery_app = Celery(
    "precise_mrd",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=["src.precise_mrd.tasks"],
)

celery_app.conf.update(
    task_track_started=True,
    broker_connection_retry_on_startup=True,
)

from celery import Celery

from .settings import settings

REDIS_URL = settings.REDIS_URL

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

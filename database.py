import json
from typing import Any

from sqlalchemy import Column, DateTime, Float, String, Text, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.sql import func

from .settings import settings

DATABASE_URL = settings.DATABASE_URL

engine = create_engine(
    DATABASE_URL,
    # The 'check_same_thread' argument is only needed for SQLite.
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {},
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Job(Base):
    __tablename__ = "jobs"

    id = Column(String, primary_key=True, index=True)
    run_id = Column(String, index=True)
    status = Column(String, default="pending")
    progress = Column(Float, default=0.0)
    results = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    def get_results(self) -> dict[str, Any] | None:
        if self.results:
            return json.loads(self.results)
        return None

    def set_results(self, results_dict: dict[str, Any]):
        self.results = json.dumps(results_dict)


def init_db():
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

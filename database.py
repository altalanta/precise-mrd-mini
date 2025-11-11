
import os
from sqlalchemy import create_engine, Column, String, Float, DateTime, Text, Integer
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.sql import func
import json
from typing import Dict, Any, Optional, List

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./jobs.db")

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
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

    def get_results(self) -> Optional[Dict[str, Any]]:
        if self.results:
            return json.loads(self.results)
        return None

    def set_results(self, results_dict: Dict[str, Any]):
        self.results = json.dumps(results_dict)


def init_db():
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

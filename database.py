"""Database configuration with both sync and async session support."""

import json
from collections.abc import AsyncGenerator
from typing import Any

from sqlalchemy import Column, DateTime, Float, String, Text, create_engine
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.sql import func

from .settings import settings

# --- Sync Engine (for Celery tasks and migrations) ---
sync_engine = create_engine(
    settings.DATABASE_URL,
    # The 'check_same_thread' argument is only needed for SQLite.
    connect_args={"check_same_thread": False}
    if "sqlite" in settings.DATABASE_URL
    else {},
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=sync_engine)

# --- Async Engine (for FastAPI endpoints) ---
async_engine = create_async_engine(
    settings.ASYNC_DATABASE_URL,
    # For aiosqlite, we don't need check_same_thread as it handles this internally
)
AsyncSessionLocal = async_sessionmaker(
    async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)

Base = declarative_base()


class Job(Base):
    """Database model for pipeline jobs."""

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
    """Initialize the database schema using the sync engine."""
    Base.metadata.create_all(bind=sync_engine)


async def init_db_async():
    """Initialize the database schema using the async engine."""
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


def get_db():
    """
    Sync database session dependency (for Celery tasks).

    Yields a synchronous SQLAlchemy session.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


async def get_async_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Async database session dependency (for FastAPI endpoints).

    Yields an asynchronous SQLAlchemy session that doesn't block the event loop.
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()

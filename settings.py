"""
Centralized, environment-aware configuration management for the application.

This module uses pydantic-settings to load configuration from environment
variables and .env files, allowing for a flexible and secure setup across
different environments (local development, CI/CD, production).
"""
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

# Base directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent.parent

class Settings(BaseSettings):
    """
    Defines the application's configuration settings, loaded from environment variables.
    """
    # --- Core Application Settings ---
    APP_ENV: str = "development"
    LOG_LEVEL: str = "INFO"

    # --- Database Configuration ---
    # Default to a local SQLite database file in the project root.
    DATABASE_URL: str = f"sqlite:///{BASE_DIR / 'jobs.db'}"

    # --- Redis Configuration for Celery ---
    # Default to a standard local Redis instance.
    REDIS_URL: str = "redis://localhost:6379/0"

    # --- MLflow Configuration ---
    MLFLOW_TRACKING_URI: str = (BASE_DIR / "mlruns").as_uri()
    MLFLOW_EXPERIMENT_NAME: str = "precise-mrd-pipeline"

    # Pydantic-settings configuration
    model_config = SettingsConfigDict(
        env_file=(".env", ".env.local"),  # Load from .env files
        env_file_encoding='utf-8',
        extra='ignore'  # Ignore extra fields from the environment
    )

# Create a single, importable instance of the settings
settings = Settings()

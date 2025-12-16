"""
Utilities for MLOps, focusing on MLflow experiment tracking.
"""

import json
from pathlib import Path
from typing import Any

import mlflow

from .logging_config import get_logger
from .settings import settings

log = get_logger(__name__)


def setup_mlflow():
    """
    Configures the MLflow tracking URI and sets the experiment based on application settings.
    """
    tracking_uri = settings.MLFLOW_TRACKING_URI
    experiment_name = settings.MLFLOW_EXPERIMENT_NAME

    # Handle local file paths correctly
    if not tracking_uri.startswith(("http", "file:", "sqlite:")):
        tracking_uri_path = Path(tracking_uri).resolve()
        tracking_uri_path.mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(tracking_uri_path.as_uri())
    else:
        mlflow.set_tracking_uri(tracking_uri)

    mlflow.set_experiment(experiment_name)
    log.info(
        "MLflow tracking enabled. URI: '%s', Experiment: '%s'",
        mlflow.get_tracking_uri(),
        experiment_name,
    )


def log_pipeline_run(
    run_name: str,
    params: dict[str, Any],
    metrics_path: Path,
    artifacts_dir: Path,
    tags: dict[str, str] = None,
):
    """
    Logs a complete pipeline run to MLflow, including parameters, metrics, and all artifacts.

    Args:
        run_name: A descriptive name for the MLflow run.
        params: A dictionary of parameters to log.
        metrics_path: Path to the metrics.json file.
        artifacts_dir: Path to the directory containing all output artifacts to log.
        tags: Optional dictionary of tags to set for the run.
    """
    with mlflow.start_run(run_name=run_name) as run:
        log.info("Starting MLflow run: %s (%s)", run.info.run_name, run.info.run_id)

        # Log parameters
        mlflow.log_params(params)

        # Set tags
        if tags:
            mlflow.set_tags(tags)

        # Log metrics from metrics.json
        if metrics_path.exists():
            with open(metrics_path) as f:
                metrics_data = json.load(f)

            # Log overall metrics and metrics per allele fraction
            if "overall" in metrics_data:
                mlflow.log_metrics(metrics_data["overall"])
            if "per_af" in metrics_data:
                # MLflow prefers flat key-value pairs
                for af_metrics in metrics_data["per_af"]:
                    af = af_metrics.get("allele_fraction", "unknown_af")
                    for key, value in af_metrics.items():
                        if key != "allele_fraction" and isinstance(value, (int, float)):
                            mlflow.log_metric(f"af_{af}_{key}", value)
        else:
            log.warning(
                "Metrics file not found at %s, skipping metric logging.",
                metrics_path,
            )

        # Log all artifacts from the output directory
        if artifacts_dir.is_dir():
            mlflow.log_artifacts(str(artifacts_dir), artifact_path="results")
            log.info("Logged artifacts from directory: %s", artifacts_dir)
        else:
            log.warning(
                "Artifacts directory not found at %s, skipping artifact logging.",
                artifacts_dir,
            )

        log.info(f"Completed MLflow run: {run.info.run_name}")
        return run.info.run_id

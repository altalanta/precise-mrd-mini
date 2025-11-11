"""MLOps utilities for experiment tracking with MLflow."""

import mlflow
from pathlib import Path
from typing import Dict, Any

def setup_mlflow(experiment_name: str = "precise-mrd", tracking_uri: str = "mlruns"):
    """
    Configure MLflow tracking.
    Sets the experiment and ensures the tracking URI is set up.
    """
    mlflow.set_tracking_uri(Path(tracking_uri).as_uri())
    mlflow.set_experiment(experiment_name)

def log_experiment(params: Dict[str, Any], metrics: Dict[str, Any], artifacts: Dict[str, Any]):
    """
    Log a complete experiment run to MLflow.

    Args:
        params: Dictionary of parameters to log.
        metrics: Dictionary of metrics to log.
        artifacts: Dictionary of artifacts to log (local file paths).
    """
    with mlflow.start_run() as run:
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        for name, path in artifacts.items():
            mlflow.log_artifact(path, artifact_path=name)
        
        return run.info.run_id



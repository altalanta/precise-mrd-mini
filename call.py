"""MRD calling and statistical testing module."""

from __future__ import annotations

from typing import TYPE_CHECKING

import mlflow
import pandera as pa

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd

from .config import PipelineConfig
from .data_schemas import (
    CollapsedUmisSchema,
    ErrorModelSchema,
    MLCallsSchema,
)
from .models.base import VariantCaller
from .models.deep_learning import DLVariantCaller
from .models.machine_learning import MLVariantCaller
from .models.statistical import StatisticalVariantCaller


def get_variant_caller(
    config: PipelineConfig,
    model_type: str,
    model_subtype: str,
) -> VariantCaller:
    """Factory function to get the appropriate variant caller."""
    if model_type == "ml":
        return MLVariantCaller(config, model_subtype)
    elif model_type == "dl":
        return DLVariantCaller(config, model_subtype)
    else:  # 'statistical'
        return StatisticalVariantCaller(config)


@pa.check_input(
    pa.DataFrameSchema(CollapsedUmisSchema.to_schema().columns, strict=False),
    "collapsed_df",
)
def train_model(
    collapsed_df: pd.DataFrame,
    config: PipelineConfig,
    rng: np.random.Generator,
    model_type: str,
    model_subtype: str,
) -> dict:
    """
    Train a variant calling model and register it in MLflow.
    """
    caller = get_variant_caller(config, model_type, model_subtype)
    training_results = caller.train(collapsed_df, rng)
    return training_results


@pa.check_input(
    pa.DataFrameSchema(CollapsedUmisSchema.to_schema().columns, strict=False),
    "collapsed_df",
)
@pa.check_input(
    pa.DataFrameSchema(ErrorModelSchema.to_schema().columns, strict=False),
    "error_model_df",
)
def predict_from_model(
    collapsed_df: pd.DataFrame,
    error_model_df: pd.DataFrame,
    config: PipelineConfig,
    model_uri: str,
    output_path: str | None = None,
) -> pd.DataFrame:
    """
    Perform MRD calling using a pre-trained model from MLflow.
    """
    # Note: For this refactoring, we assume the model type from the URI.
    # A more robust solution might inspect the model or take type as an arg.

    # Load the model from MLflow Registry
    loaded_model = mlflow.sklearn.load_model(model_uri)

    # The loaded_model is the raw sklearn model. We need to wrap it
    # or create a prediction logic here. For simplicity, we'll
    # replicate the essential parts of the original predict logic.

    # This part is simplified. A full implementation would need to
    # reconstruct the feature engineering pipeline if it exists.
    features = ["family_size", "quality_score", "consensus_agreement"]
    X_predict = collapsed_df[features]

    ml_probabilities = loaded_model.predict_proba(X_predict)[:, 1]

    # Thresholding would be loaded from MLflow run associated with the model artifact
    # For now, we use a default.
    client = mlflow.tracking.MlflowClient()
    model_version = client.get_model_version_by_alias(
        name=model_uri.split("/")[1],
        alias="latest",
    )  # Simplified
    run_info = client.get_run(model_version.run_id)
    optimal_threshold = float(run_info.data.metrics.get("optimal_threshold", 0.5))

    ml_calls = (ml_probabilities > optimal_threshold).astype(int)

    results_df = collapsed_df.copy()
    results_df["predicted_is_variant"] = ml_calls
    results_df["prediction_prob"] = ml_probabilities

    # Validate the output against the appropriate schema
    results_df = MLCallsSchema.validate(results_df)

    if output_path and not results_df.empty:
        results_df.to_parquet(output_path, index=False)

    return results_df

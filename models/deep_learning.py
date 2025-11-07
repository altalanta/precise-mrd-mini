"""Deep learning models for variant calling."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict

import numpy as np
import pandas as pd

from ..deep_learning_models import DeepLearningVariantCaller as DLModel
from .base import VariantCaller
from ..mlops import setup_mlflow
import mlflow

if TYPE_CHECKING:
    from ..config import PipelineConfig


class DLVariantCaller(VariantCaller):
    """Variant caller using deep learning models."""

    def __init__(self, config: "PipelineConfig", dl_model_type: str = 'cnn_lstm'):
        super().__init__(config)
        self.dl_model_type = dl_model_type
        self.model = DLModel(config, dl_model_type)
        self.training_results: Dict[str, Any] = {}

    def train(self, collapsed_df: pd.DataFrame, rng: np.random.Generator) -> Dict[str, Any]:
        """Train the DL model and log the experiment to MLflow."""
        setup_mlflow()

        with mlflow.start_run() as run:
            mlflow.log_param("dl_model_type", self.dl_model_type)
            mlflow.log_param("seed", self.config.seed)

            self.training_results = self.model.train_model(collapsed_df, rng)
            
            # Log metrics
            metrics_to_log = {
                "optimal_threshold": self.training_results.get("optimal_threshold", 0.0),
                "validation_loss": self.training_results.get("validation_loss", 0.0),
                "validation_accuracy": self.training_results.get("validation_accuracy", 0.0),
            }
            mlflow.log_metrics(metrics_to_log)

            # Log the trained model (assuming the model object is serializable)
            # Note: Logging PyTorch models may require a specific flavor like mlflow.pytorch
            mlflow.log_param("model_summary", self.model.get_model_summary())
            
            self.training_results['mlflow_run_id'] = run.info.run_id
            return self.training_results

    def predict(self, collapsed_df: pd.DataFrame, error_model_df: pd.DataFrame = None) -> pd.DataFrame:
        """Predict variants using the trained DL model."""
        dl_probabilities = self.model.predict_variants(collapsed_df)
        optimal_threshold = self.training_results.get('optimal_threshold', np.median(dl_probabilities))
        dl_calls = (dl_probabilities > optimal_threshold).astype(int)

        results_df = pd.DataFrame({
            'sample_id': collapsed_df['sample_id'],
            'family_id': collapsed_df['family_id'],
            'family_size': collapsed_df['family_size'],
            'quality_score': collapsed_df['quality_score'],
            'consensus_agreement': collapsed_df['consensus_agreement'],
            'passes_quality': collapsed_df['passes_quality'],
            'passes_consensus': collapsed_df['passes_consensus'],
            'is_variant': dl_calls,
            'p_value': 1.0 - dl_probabilities,
            'dl_probability': dl_probabilities,
            'dl_threshold': optimal_threshold,
            'calling_method': f'dl_{self.dl_model_type}',
            'config_hash': self.config.config_hash()
        })
        return results_df


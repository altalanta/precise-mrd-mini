"""Machine learning models for variant calling."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict

import numpy as np
import pandas as pd

from ..ml_models import EnsembleVariantCaller, GradientBoostedVariantCaller
from .base import VariantCaller

if TYPE_CHECKING:
    from ..config import PipelineConfig


class MLVariantCaller(VariantCaller):
    """Variant caller using traditional machine learning models."""

    def __init__(self, config: "PipelineConfig", ml_model_type: str = 'ensemble'):
        super().__init__(config)
        self.ml_model_type = ml_model_type
        if self.ml_model_type == 'ensemble':
            self.model = EnsembleVariantCaller(config)
        else:
            self.model = GradientBoostedVariantCaller(config, ml_model_type)
        self.training_results: Dict[str, Any] = {}

    def train(self, collapsed_df: pd.DataFrame, rng: np.random.Generator) -> Dict[str, Any]:
        """Train the ML model."""
        if self.ml_model_type == 'ensemble':
            self.training_results = self.model.train_ensemble(collapsed_df, rng)
        else:
            self.training_results = self.model.train_model(collapsed_df, rng)
        return self.training_results

    def predict(self, collapsed_df: pd.DataFrame, error_model_df: pd.DataFrame = None) -> pd.DataFrame:
        """Predict variants using the trained ML model."""
        ml_probabilities = self.model.predict_variants(collapsed_df)
        optimal_threshold = self.training_results.get('optimal_threshold', np.median(ml_probabilities))
        ml_calls = (ml_probabilities > optimal_threshold).astype(int)

        results_df = pd.DataFrame({
            'sample_id': collapsed_df['sample_id'],
            'family_id': collapsed_df['family_id'],
            'family_size': collapsed_df['family_size'],
            'quality_score': collapsed_df['quality_score'],
            'consensus_agreement': collapsed_df['consensus_agreement'],
            'passes_quality': collapsed_df['passes_quality'],
            'passes_consensus': collapsed_df['passes_consensus'],
            'is_variant': ml_calls,
            'p_value': 1.0 - ml_probabilities,
            'ml_probability': ml_probabilities,
            'ml_threshold': optimal_threshold,
            'calling_method': f'ml_{self.ml_model_type}',
            'config_hash': self.config.config_hash()
        })
        return results_df

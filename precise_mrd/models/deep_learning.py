"""Deep learning models for variant calling."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict

import numpy as np
import pandas as pd

from ..deep_learning_models import DeepLearningVariantCaller as DLModel
from .base import VariantCaller

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
        """Train the DL model."""
        self.training_results = self.model.train_model(collapsed_df, rng)
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

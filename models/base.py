"""Base classes for pluggable variant calling models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from ..config import PipelineConfig


class VariantCaller(ABC):
    """Abstract base class for all variant calling models."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.model = None

    def train(
        self, collapsed_df: pd.DataFrame, rng: np.random.Generator
    ) -> dict[str, Any]:
        """
        Train the variant calling model.

        For models that do not require explicit training (e.g., statistical tests),
        this method can be a no-op.

        Returns:
            A dictionary of training results, such as model metrics or thresholds.
        """
        return {}

    @abstractmethod
    def predict(
        self, collapsed_df: pd.DataFrame, error_model_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Predict variants for the given data.

        Returns:
            A DataFrame with variant calls and associated probabilities/statistics.
        """
        raise NotImplementedError

"""Variant calling models for the Precise MRD pipeline.

This package provides different model implementations for variant calling:

- **VariantCaller**: Abstract base class defining the interface
- **StatisticalVariantCaller**: Classical statistical testing approaches
- **MLVariantCaller**: Machine learning models (XGBoost, LightGBM, Random Forest)
- **DLVariantCaller**: Deep learning models (CNN-LSTM, Transformer)

Example:
    >>> from precise_mrd.models import MLVariantCaller
    >>> caller = MLVariantCaller(config, "ensemble")
    >>> results = caller.train(collapsed_df, rng)
"""

from .base import VariantCaller
from .deep_learning import DLVariantCaller
from .machine_learning import MLVariantCaller
from .statistical import StatisticalVariantCaller

__all__ = [
    "VariantCaller",
    "StatisticalVariantCaller",
    "MLVariantCaller",
    "DLVariantCaller",
]

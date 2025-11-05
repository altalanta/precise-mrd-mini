"""MRD calling and statistical testing module."""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Optional

from .config import PipelineConfig
from .data_schemas import (
    CollapsedUmisSchema, ErrorModelSchema, StatisticalCallsSchema,
    MLCallsSchema, DLCallsSchema
)
import pandera as pa
from .models.base import VariantCaller
from .models.statistical import StatisticalVariantCaller
from .models.machine_learning import MLVariantCaller
from .models.deep_learning import DLVariantCaller


def get_variant_caller(
    config: PipelineConfig,
    use_ml_calling: bool,
    ml_model_type: str,
    use_deep_learning: bool,
    dl_model_type: str
) -> VariantCaller:
    """Factory function to get the appropriate variant caller."""
    if use_ml_calling:
        return MLVariantCaller(config, ml_model_type)
    elif use_deep_learning:
        return DLVariantCaller(config, dl_model_type)
    else:
        return StatisticalVariantCaller(config)


@pa.check_input(pa.DataFrameSchema(CollapsedUmisSchema.to_schema().columns,
                                 filter_ignore_na=True,
                                 strict=False), "collapsed_df")
@pa.check_input(pa.DataFrameSchema(ErrorModelSchema.to_schema().columns,
                                 filter_ignore_na=True,
                                 strict=False), "error_model_df")
def call_mrd(
    collapsed_df: pd.DataFrame,
    error_model_df: pd.DataFrame,
    config: PipelineConfig,
    rng: np.random.Generator,
    output_path: Optional[str] = None,
    use_ml_calling: bool = False,
    ml_model_type: str = 'ensemble',
    use_deep_learning: bool = False,
    dl_model_type: str = 'cnn_lstm'
) -> pd.DataFrame:
    """
    Perform MRD calling by dispatching to the appropriate pluggable model.
    """
    caller = get_variant_caller(
        config, use_ml_calling, ml_model_type, use_deep_learning, dl_model_type
    )

    # Train the model (if applicable) and predict
    caller.train(collapsed_df, rng)
    results_df = caller.predict(collapsed_df, error_model_df)

    # Validate the output against the appropriate schema
    if isinstance(caller, MLVariantCaller):
        results_df = MLCallsSchema.validate(results_df)
    elif isinstance(caller, DLVariantCaller):
        results_df = DLCallsSchema.validate(results_df)
    elif isinstance(caller, StatisticalVariantCaller):
        results_df = StatisticalCallsSchema.validate(results_df)

    # Save results if output path specified
    if output_path and not results_df.empty:
        results_df.to_parquet(output_path, index=False)

    return results_df

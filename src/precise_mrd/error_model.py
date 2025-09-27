"""Background error model fitting."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import PipelineConfig
from .schemas import COLLAPSED_UMI_SCHEMA, ERROR_MODEL_SCHEMA


def fit_error_model(collapsed: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    """Fit beta-binomial parameters using control samples."""

    COLLAPSED_UMI_SCHEMA.validate(collapsed)
    controls = collapsed[collapsed["sample_type"] == "control"].copy()
    if controls.empty:
        msg = "no control samples available for error model"
        raise ValueError(msg)

    controls = controls.assign(
        proportion=lambda df: np.where(
            df["family_size"] > 0,
            df["alt_reads"] / df["family_size"],
            0.0,
        )
    )

    rows = []
    err_cfg = config.error_model
    for (variant_id, depth), frame in controls.groupby(["variant_id", "depth"]):
        mean = frame["proportion"].mean()
        var = frame["proportion"].var(ddof=1)

        if var <= 0:
            alpha = err_cfg.alpha_prior
            beta = err_cfg.beta_prior
        else:
            common = mean * (1 - mean) / var - 1
            if common <= 0:
                alpha = err_cfg.alpha_prior
                beta = err_cfg.beta_prior
            else:
                alpha = max(mean * common, err_cfg.alpha_prior)
                beta = max((1 - mean) * common, err_cfg.beta_prior)

        rows.append(
            {
                "variant_id": variant_id,
                "depth": depth,
                "alpha": float(alpha),
                "beta": float(beta),
                "mean_error": float(mean),
            }
        )

    error_df = pd.DataFrame(rows)
    ERROR_MODEL_SCHEMA.validate(error_df)
    return error_df

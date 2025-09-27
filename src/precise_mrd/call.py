"""MRD calling stage."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import betabinom

from .config import PipelineConfig
from .lod import lod_grid
from .metrics import (
    average_precision,
    brier_score,
    bootstrap_metric,
    calibration_curve,
    roc_auc_score,
)
from .rng import RandomState
from .schemas import COLLAPSED_UMI_SCHEMA, ERROR_MODEL_SCHEMA, MRD_CALL_SCHEMA


def call_mrd(
    collapsed: pd.DataFrame,
    error_model: pd.DataFrame,
    config: PipelineConfig,
    rng: RandomState,
) -> tuple[pd.DataFrame, dict[str, object], pd.DataFrame]:
    """Perform MRD calling and compute evaluation metrics."""

    COLLAPSED_UMI_SCHEMA.validate(collapsed)
    ERROR_MODEL_SCHEMA.validate(error_model)

    merged = collapsed.merge(error_model, on=["variant_id", "depth"], how="left")
    merged[["alpha", "beta"]] = merged[["alpha", "beta"]].fillna(
        {
            "alpha": config.error_model.alpha_prior,
            "beta": config.error_model.beta_prior,
        }
    )

    merged["total_reads"] = merged["alt_reads"] + merged["ref_reads"]
    merged["pvalue"] = betabinom.sf(
        merged["alt_reads"] - 1,
        merged["total_reads"],
        merged["alpha"],
        merged["beta"],
    )
    merged["detected"] = merged["pvalue"] < config.call.pvalue_threshold
    merged["truth_positive"] = merged["allele_fraction"] > 0

    MRD_CALL_SCHEMA.validate(merged)

    labels = merged["truth_positive"].astype(int).to_numpy()
    scores = 1.0 - merged["pvalue"].to_numpy()
    probs = np.clip(scores, 1e-9, 1 - 1e-9)

    roc = roc_auc_score(labels, scores)
    pr = average_precision(labels, scores)
    brier = brier_score(labels, probs)
    curve = calibration_curve(labels, probs, bins=config.call.calibration_bins)

    ci_roc = bootstrap_metric(
        labels,
        scores,
        roc_auc_score,
        samples=config.error_model.bootstrap_samples,
        ci_level=config.error_model.ci_level,
        rng=rng.generator,
    )
    ci_pr = bootstrap_metric(
        labels,
        scores,
        average_precision,
        samples=config.error_model.bootstrap_samples,
        ci_level=config.error_model.ci_level,
        rng=rng.generator,
    )

    lod = lod_grid(collapsed)

    case_detection = (
        merged[merged["sample_type"] == "case"]
        .groupby("sample_id", as_index=False)["detected"]
        .any()
    )

    metrics_payload = {
        "roc_auc": roc,
        "roc_auc_ci": ci_roc,
        "average_precision": pr,
        "average_precision_ci": ci_pr,
        "brier_score": brier,
        "detected_cases": int(case_detection["detected"].sum()),
        "total_cases": int(case_detection.shape[0]),
        "calibration": curve.to_dict(orient="records"),
    }

    return merged, metrics_payload, lod

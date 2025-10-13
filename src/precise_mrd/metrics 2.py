"""Metrics calculation for MRD performance evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score as sklearn_roc_auc
from sklearn.metrics import average_precision_score as sklearn_ap
from typing import Dict, Any, Optional, Tuple


def roc_auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Calculate ROC AUC score with error handling."""
    try:
        return sklearn_roc_auc(y_true, y_score)
    except ValueError:
        # Handle edge cases (all same class)
        return 0.5


def average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Calculate average precision score with error handling."""
    try:
        return sklearn_ap(y_true, y_score)
    except ValueError:
        # Handle edge cases
        return np.mean(y_true)


def bootstrap_metric(
    y_true: np.ndarray, 
    y_score: np.ndarray, 
    metric_func,
    n_bootstrap: int = 1000,
    rng: Optional[np.random.Generator] = None
) -> Dict[str, float]:
    """Bootstrap confidence intervals for a metric.
    
    Args:
        y_true: True binary labels
        y_score: Prediction scores
        metric_func: Metric function to bootstrap
        n_bootstrap: Number of bootstrap samples
        rng: Random number generator
        
    Returns:
        Dictionary with mean, lower, upper CI, and std
    """
    if rng is None:
        rng = np.random.default_rng(42)
    
    n_samples = len(y_true)
    bootstrap_scores = []
    
    for _ in range(n_bootstrap):
        # Bootstrap sample with replacement
        indices = rng.choice(n_samples, size=n_samples, replace=True)
        boot_y_true = y_true[indices]
        boot_y_score = y_score[indices]
        
        # Calculate metric on bootstrap sample
        try:
            score = metric_func(boot_y_true, boot_y_score)
            bootstrap_scores.append(score)
        except Exception:
            # Skip failed bootstrap samples
            continue
    
    if not bootstrap_scores:
        return {"mean": 0.0, "lower": 0.0, "upper": 0.0, "std": 0.0}
    
    bootstrap_scores = np.array(bootstrap_scores)
    
    return {
        "mean": np.mean(bootstrap_scores),
        "lower": np.percentile(bootstrap_scores, 2.5),
        "upper": np.percentile(bootstrap_scores, 97.5),
        "std": np.std(bootstrap_scores)
    }


def calibration_analysis(
    y_true: np.ndarray, 
    y_prob: np.ndarray, 
    n_bins: int = 10
) -> list[Dict[str, Any]]:
    """Analyze prediction calibration.
    
    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        n_bins: Number of calibration bins
        
    Returns:
        List of calibration bin statistics
    """
    bins = np.linspace(0, 1, n_bins + 1)
    calibration_data = []
    
    for i in range(n_bins):
        bin_lower = bins[i]
        bin_upper = bins[i + 1]
        
        # Find predictions in this bin
        in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper)
        
        if i == n_bins - 1:  # Last bin includes upper bound
            in_bin = (y_prob >= bin_lower) & (y_prob <= bin_upper)
        
        if np.sum(in_bin) == 0:
            continue
        
        bin_count = np.sum(in_bin)
        bin_event_rate = np.mean(y_true[in_bin])
        bin_confidence = np.mean(y_prob[in_bin])
        
        calibration_data.append({
            "bin": i,
            "lower": bin_lower,
            "upper": bin_upper,
            "count": int(bin_count),
            "event_rate": float(bin_event_rate),
            "confidence": float(bin_confidence)
        })
    
    return calibration_data


def calculate_metrics(
    calls_df: pd.DataFrame,
    rng: np.random.Generator,
    n_bootstrap: int = 1000
) -> Dict[str, Any]:
    """Calculate comprehensive performance metrics.
    
    Args:
        calls_df: DataFrame with MRD calls
        rng: Random number generator for bootstrap
        n_bootstrap: Number of bootstrap samples
        
    Returns:
        Dictionary with all metrics
    """
    if len(calls_df) == 0:
        return {
            "roc_auc": 0.0,
            "average_precision": 0.0,
            "detected_cases": 0,
            "total_cases": 0,
            "calibration": []
        }
    
    # Define truth labels (high AF = positive case)
    y_true = (calls_df['allele_fraction'] > 0.001).astype(int)
    
    # Use variant fraction as prediction score
    y_score = calls_df['variant_fraction'].values
    
    # Basic metrics
    roc_auc = roc_auc_score(y_true, y_score)
    avg_precision = average_precision(y_true, y_score)
    
    # Bootstrap confidence intervals
    roc_ci = bootstrap_metric(y_true, y_score, roc_auc_score, n_bootstrap, rng)
    ap_ci = bootstrap_metric(y_true, y_score, average_precision, n_bootstrap, rng)
    
    # Detection statistics
    detected_cases = int(np.sum(calls_df['significant'] & (y_true == 1)))
    total_cases = int(np.sum(y_true))
    
    # Calibration analysis
    calibration = calibration_analysis(y_true, y_score)
    
    # Brier score
    brier_score = float(np.mean((y_score - y_true) ** 2))
    
    return {
        "roc_auc": float(roc_auc),
        "roc_auc_ci": roc_ci,
        "average_precision": float(avg_precision),
        "average_precision_ci": ap_ci,
        "brier_score": brier_score,
        "detected_cases": detected_cases,
        "total_cases": total_cases,
        "calibration": calibration
    }
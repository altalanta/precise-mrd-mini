"""Metrics calculation for MRD performance evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score as sklearn_roc_auc
from sklearn.metrics import average_precision_score as sklearn_ap
from typing import Dict, Any, Optional, Tuple
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

from .config import PipelineConfig
from .advanced_stats import AdvancedConfidenceIntervals
from .statistical_validation import CrossValidator, UncertaintyQuantifier, StatisticalTester, RobustnessAnalyzer, ModelValidator


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


def _bootstrap_worker(args):
    """Worker function for parallel bootstrap computation."""
    y_true, y_score, indices, metric_func = args
    boot_y_true = y_true[indices]
    boot_y_score = y_score[indices]

    try:
        return metric_func(boot_y_true, boot_y_score)
    except Exception:
        return None


def bootstrap_metric(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric_func,
    n_bootstrap: int = 1000,
    rng: Optional[np.random.Generator] = None,
    n_jobs: int = -1
) -> Dict[str, float]:
    """Bootstrap confidence intervals for a metric with parallel processing.

    Args:
        y_true: True binary labels
        y_score: Prediction scores
        metric_func: Metric function to bootstrap
        n_bootstrap: Number of bootstrap samples
        rng: Random number generator
        n_jobs: Number of parallel jobs (-1 for all cores)

    Returns:
        Dictionary with mean, lower, upper CI, and std
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n_samples = len(y_true)
    n_jobs = mp.cpu_count() if n_jobs == -1 else n_jobs

    # Generate all bootstrap indices upfront
    bootstrap_indices = [
        rng.choice(n_samples, size=n_samples, replace=True)
        for _ in range(n_bootstrap)
    ]

    # Parallel computation
    bootstrap_scores = []
    if n_jobs == 1:
        # Single-threaded fallback
        for indices in bootstrap_indices:
            score = _bootstrap_worker((y_true, y_score, indices, metric_func))
            if score is not None:
                bootstrap_scores.append(score)
    else:
        # Parallel execution with deterministic ordering
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            # Submit all tasks and maintain deterministic order
            future_to_indices = {
                executor.submit(_bootstrap_worker, (y_true, y_score, indices, metric_func)): i
                for i, indices in enumerate(bootstrap_indices)
            }

            # Collect results in submission order for determinism
            results = [None] * len(bootstrap_indices)
            for future in as_completed(future_to_indices):
                indices_idx = future_to_indices[future]
                score = future.result()
                results[indices_idx] = score

            # Filter out None results and maintain order
            bootstrap_scores = [score for score in results if score is not None]

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
    n_bootstrap: int = 1000,
    n_jobs: int = -1,
    config: Optional[PipelineConfig] = None,
    use_advanced_ci: bool = False,
    run_validation: bool = False
) -> Dict[str, Any]:
    """Calculate comprehensive performance metrics with optional advanced confidence intervals and validation.

    Args:
        calls_df: DataFrame with MRD calls
        rng: Random number generator for bootstrap
        n_bootstrap: Number of bootstrap samples
        n_jobs: Number of parallel jobs for bootstrap (-1 for all cores)
        config: Pipeline configuration for advanced statistics
        use_advanced_ci: Whether to use advanced confidence interval methods
        run_validation: Whether to run comprehensive statistical validation

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
    if 'allele_fraction' in calls_df.columns:
        y_true = (calls_df['allele_fraction'] > 0.001).astype(int)
    else:
        # For real data or ML-based calling, we don't have ground truth
        # Use a default approach or skip certain metrics
        print("  Warning: No allele_fraction column found, using simplified metrics")
        y_true = np.zeros(len(calls_df))  # All negative for compatibility
    
    # Use variant fraction as prediction score
    if 'variant_fraction' in calls_df.columns:
        y_score = calls_df['variant_fraction'].values
    elif 'ml_probability' in calls_df.columns:
        y_score = calls_df['ml_probability'].values
    else:
        # Fallback to p-value if available
        if 'p_value' in calls_df.columns:
            y_score = 1.0 - calls_df['p_value'].values  # Convert p-value to score
        else:
            y_score = np.random.random(len(calls_df))  # Random fallback
    
    # Basic metrics
    roc_auc = roc_auc_score(y_true, y_score)
    avg_precision = average_precision(y_true, y_score)
    
    # Confidence intervals
    if use_advanced_ci and config:
        print("  Using advanced confidence interval methods...")
        adv_ci = AdvancedConfidenceIntervals(config)

        # Advanced bootstrap CI for ROC AUC
        roc_ci = adv_ci.bootstrap_confidence_interval(
            y_score, lambda x: roc_auc_score(y_true, x), n_bootstrap, rng
        )

        # Advanced bootstrap CI for Average Precision
        ap_ci = adv_ci.bootstrap_confidence_interval(
            y_score, lambda x: average_precision(y_true, x), n_bootstrap, rng
        )

        # Add method information
        roc_ci['method'] = 'advanced_bootstrap'
        ap_ci['method'] = 'advanced_bootstrap'
    else:
        # Standard bootstrap CI (parallel processing)
        roc_ci = bootstrap_metric(y_true, y_score, roc_auc_score, n_bootstrap, rng, n_jobs)
        ap_ci = bootstrap_metric(y_true, y_score, average_precision, n_bootstrap, rng, n_jobs)
    
    # Detection statistics
    if 'significant' in calls_df.columns:
        if 'allele_fraction' in calls_df.columns:
            detected_cases = int(np.sum(calls_df['significant'] & (y_true == 1)))
            total_cases = int(np.sum(y_true))
        else:
            # For real data, just count significant calls
            detected_cases = int(np.sum(calls_df['significant']))
            total_cases = len(calls_df)  # All samples are "cases" in real data context
    else:
        detected_cases = 0
        total_cases = len(calls_df)
    
    # Calibration analysis
    calibration = calibration_analysis(y_true, y_score)
    
    # Brier score
    brier_score = float(np.mean((y_score - y_true) ** 2))
    
    # Add statistical validation if requested
    validation_results = {}
    if run_validation and config:
        print("  Running comprehensive statistical validation...")

        # Cross-validation for model performance
        if 'ml_probability' in calls_df.columns:
            X = calls_df[['family_size', 'quality_score', 'consensus_agreement']].values
            y = calls_df['is_variant'].values

            # Simple model function for demonstration
            def simple_model_func(X_train, y_train):
                from sklearn.ensemble import RandomForestClassifier
                model = RandomForestClassifier(n_estimators=10, random_state=config.seed)
                model.fit(X_train, y_train)
                return model

            cv = CrossValidator(config)
            cv_results = cv.k_fold_cross_validation(X, y, simple_model_func, k_folds=3)
            validation_results['cross_validation'] = cv_results

        # Calibration analysis
        if 'ml_probability' in calls_df.columns:
            calibrator = ModelValidator(config)
            calibration_results = calibrator.calibration_analysis(y_true, calls_df['ml_probability'].values)
            validation_results['calibration_analysis'] = calibration_results

        # Robustness analysis
        robustness = RobustnessAnalyzer(config)
        robustness_results = robustness.bootstrap_robustness(calls_df, n_bootstrap=50)
        validation_results['robustness_analysis'] = robustness_results

        # Uncertainty quantification
        uncertainty = UncertaintyQuantifier(config)
        # Example: quantify uncertainty in variant rate estimates
        variant_rates = calls_df['is_variant'].values
        uncertainty_results = uncertainty.bayesian_uncertainty([variant_rates])
        validation_results['uncertainty_quantification'] = uncertainty_results

    return {
        "roc_auc": float(roc_auc),
        "roc_auc_ci": roc_ci,
        "average_precision": float(avg_precision),
        "average_precision_ci": ap_ci,
        "brier_score": brier_score,
        "detected_cases": detected_cases,
        "total_cases": total_cases,
        "calibration": calibration,
        "statistical_validation": validation_results if validation_results else None
    }
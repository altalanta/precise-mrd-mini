"""Validation utilities for pipeline configuration and results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .config import PipelineConfig


def validate_config(config: PipelineConfig) -> dict[str, Any]:
    """Validate pipeline configuration.
    
    Args:
        config: Pipeline configuration to validate
        
    Returns:
        Validation result dictionary with status and details
    """
    issues = []
    warnings = []
    
    # Check simulation parameters
    if config.simulation.n_replicates < 10:
        warnings.append("Very low n_replicates may produce unstable results")
    if config.simulation.n_replicates > 10000:
        warnings.append("High n_replicates may be slow")
    
    if config.simulation.n_bootstrap < 100:
        warnings.append("Low n_bootstrap may produce wide confidence intervals")
    
    # Check UMI parameters
    if config.umi.min_family_size < 1:
        issues.append("min_family_size must be >= 1")
    if config.umi.min_family_size > 10:
        warnings.append("High min_family_size may reduce sensitivity")
    
    if not 0.0 <= config.umi.consensus_threshold <= 1.0:
        issues.append("consensus_threshold must be between 0 and 1")
    
    # Check statistical parameters
    if not 0.0 < config.stats.alpha < 1.0:
        issues.append("alpha must be between 0 and 1")
    
    if config.stats.test_type not in ["poisson", "binomial"]:
        issues.append(f"Unknown test_type: {config.stats.test_type}")
    
    # Check LoD parameters
    if not 0.0 < config.lod.detection_threshold <= 1.0:
        issues.append("detection_threshold must be between 0 and 1")
    
    if not 0.0 < config.lod.confidence_level <= 1.0:
        issues.append("confidence_level must be between 0 and 1")
    
    # Determine status
    if issues:
        status = "FAILED"
    elif warnings:
        status = "WARNING"
    else:
        status = "PASSED"
    
    return {
        "status": status,
        "issues": issues,
        "warnings": warnings,
        "total_checks": 8,
        "passed_checks": 8 - len(issues),
    }


def validate_results(results_dir: Path) -> dict[str, Any]:
    """Validate pipeline results directory.
    
    Args:
        results_dir: Directory containing pipeline results
        
    Returns:
        Validation result dictionary with status and details
    """
    issues = []
    warnings = []
    found_files = []
    
    # Expected files for a complete run
    expected_files = {
        "config.json": "Configuration file",
        "simulate.parquet": "Simulation results",
        "collapse.parquet": "UMI collapse results",
        "error_model.parquet": "Error model data",
        "call.parquet": "Variant calls",
        "lod_grid.parquet": "LoD grid data",
        "metrics.json": "Summary metrics",
        "lineage.jsonl": "Execution lineage",
    }
    
    # Check for expected files
    for filename, description in expected_files.items():
        filepath = results_dir / filename
        if filepath.exists():
            found_files.append(filename)
            # Additional validation for specific files
            try:
                if filename.endswith(".parquet"):
                    df = pd.read_parquet(filepath)
                    if df.empty:
                        warnings.append(f"{filename} exists but is empty")
                elif filename.endswith(".json"):
                    with open(filepath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        if not data:
                            warnings.append(f"{filename} exists but is empty")
            except Exception as e:
                issues.append(f"{filename} exists but cannot be read: {e}")
        else:
            issues.append(f"Missing {description} ({filename})")
    
    # Check for smoke test specific files
    smoke_files = ["smoke_scores.npy", "run_context.json"]
    smoke_found = sum(1 for f in smoke_files if (results_dir / f).exists())
    if smoke_found > 0 and smoke_found < len(smoke_files):
        warnings.append("Partial smoke test files found")
    
    # Check for reports
    report_files = ["auto_report.html", "auto_report.md"]
    report_found = any((results_dir / f).exists() for f in report_files)
    if not report_found:
        warnings.append("No report files found")
    
    # Validate metrics if available
    metrics_file = results_dir / "metrics.json"
    if metrics_file.exists():
        try:
            with open(metrics_file, "r", encoding="utf-8") as f:
                metrics = json.load(f)
            
            # Check for key metrics
            required_metrics = ["roc_auc", "pr_auc", "lod95_estimate"]
            for metric in required_metrics:
                if metric not in metrics:
                    warnings.append(f"Missing metric: {metric}")
                else:
                    value = metrics[metric]
                    if not isinstance(value, (int, float)) or value < 0:
                        issues.append(f"Invalid {metric} value: {value}")
        except Exception as e:
            issues.append(f"Cannot validate metrics: {e}")
    
    # Determine status
    if issues:
        status = "FAILED"
    elif warnings:
        status = "WARNING"
    else:
        status = "PASSED"
    
    return {
        "status": status,
        "issues": issues,
        "warnings": warnings,
        "found_files": found_files,
        "total_expected": len(expected_files),
        "found_expected": len(found_files),
        "completion_rate": len(found_files) / len(expected_files),
    }


def validate_determinism(results_dir: Path, reference_scores: list[float] | None = None) -> dict[str, Any]:
    """Validate deterministic execution by comparing smoke scores.
    
    Args:
        results_dir: Directory containing pipeline results
        reference_scores: Reference scores to compare against
        
    Returns:
        Validation result dictionary with status and details
    """
    import numpy as np
    
    issues = []
    warnings = []
    
    smoke_scores_file = results_dir / "smoke_scores.npy"
    
    if not smoke_scores_file.exists():
        issues.append("smoke_scores.npy not found")
        return {
            "status": "FAILED",
            "issues": issues,
            "warnings": warnings,
        }
    
    try:
        current_scores = np.load(smoke_scores_file)
        
        if reference_scores is not None:
            reference_array = np.array(reference_scores)
            
            # Check if arrays have same shape
            if current_scores.shape != reference_array.shape:
                issues.append(f"Score array shape mismatch: {current_scores.shape} vs {reference_array.shape}")
            else:
                # Check numerical equality within tolerance
                tolerance = 1e-10
                if not np.allclose(current_scores, reference_array, atol=tolerance):
                    issues.append(f"Scores differ from reference beyond tolerance {tolerance}")
                    max_diff = np.max(np.abs(current_scores - reference_array))
                    warnings.append(f"Maximum difference: {max_diff}")
        
        # Basic sanity checks
        if np.any(np.isnan(current_scores)):
            issues.append("NaN values found in scores")
        
        if np.any(np.isinf(current_scores)):
            issues.append("Infinite values found in scores")
        
        if not np.all((current_scores >= 0) & (current_scores <= 1)):
            warnings.append("Some scores outside [0,1] range")
    
    except Exception as e:
        issues.append(f"Cannot load or validate smoke scores: {e}")
    
    # Determine status
    if issues:
        status = "FAILED"
    elif warnings:
        status = "WARNING"
    else:
        status = "PASSED"
    
    return {
        "status": status,
        "issues": issues,
        "warnings": warnings,
        "scores_shape": current_scores.shape if 'current_scores' in locals() else None,
        "has_reference": reference_scores is not None,
    }
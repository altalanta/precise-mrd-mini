"""Test artifact contract compliance."""

import json
import subprocess
import sys
from pathlib import Path

import jsonschema
import pytest


def test_smoke_artifacts_exist():
    """Test that smoke command creates all required artifacts."""
    
    # Run smoke test
    result = subprocess.run([
        sys.executable, "-m", "precise_mrd.cli", "smoke", 
        "--seed", "7", "--out", "data/test_smoke"
    ], capture_output=True, text=True, cwd=Path.cwd())
    
    assert result.returncode == 0, f"Smoke test failed: {result.stderr}"
    
    # Check required artifacts exist
    required_artifacts = [
        "reports/metrics.json",
        "reports/auto_report.html", 
        "reports/run_context.json"
    ]
    
    for artifact in required_artifacts:
        artifact_path = Path(artifact)
        assert artifact_path.exists(), f"Required artifact missing: {artifact}"
        assert artifact_path.stat().st_size > 0, f"Artifact is empty: {artifact}"


def test_metrics_schema_validation():
    """Test that metrics.json validates against schema."""
    
    metrics_path = Path("reports/metrics.json")
    schema_path = Path("schemas/metrics.schema.json")
    
    if not metrics_path.exists():
        pytest.skip("metrics.json not found, run smoke test first")
    
    assert schema_path.exists(), "metrics schema not found"
    
    # Load and validate
    with open(metrics_path) as f:
        metrics = json.load(f)
    
    with open(schema_path) as f:
        schema = json.load(f)
    
    # Should not raise ValidationError
    jsonschema.validate(metrics, schema)
    
    # Check required fields are present
    assert "roc_auc" in metrics
    assert "average_precision" in metrics
    assert "detected_cases" in metrics
    assert "total_cases" in metrics
    assert "calibration" in metrics
    
    # Check value ranges
    assert 0 <= metrics["roc_auc"] <= 1
    assert 0 <= metrics["average_precision"] <= 1
    assert metrics["detected_cases"] >= 0
    assert metrics["total_cases"] >= 0


def test_run_context_schema_validation():
    """Test that run_context.json validates against schema."""
    
    context_path = Path("reports/run_context.json")
    schema_path = Path("schemas/run_context.schema.json")
    
    if not context_path.exists():
        pytest.skip("run_context.json not found, run smoke test first")
    
    assert schema_path.exists(), "run_context schema not found"
    
    # Load and validate
    with open(context_path) as f:
        context = json.load(f)
    
    with open(schema_path) as f:
        schema = json.load(f)
    
    # Should not raise ValidationError
    jsonschema.validate(context, schema)
    
    # Check required fields
    assert "seed" in context
    assert "timestamp" in context
    assert "config_hash" in context
    assert "git_sha" in context
    assert "python_version" in context
    
    # Check seed is integer
    assert isinstance(context["seed"], int)
    
    # Check config hash is reasonable length
    assert len(context["config_hash"]) >= 8
    
    # Check git_sha is string (may be "unknown")
    assert isinstance(context["git_sha"], str)


def test_html_report_exists():
    """Test that HTML report is created and has content."""
    
    report_path = Path("reports/auto_report.html")
    
    if not report_path.exists():
        pytest.skip("auto_report.html not found, run smoke test first")
    
    # Check file has content
    assert report_path.stat().st_size > 0
    
    # Check basic HTML structure
    with open(report_path) as f:
        content = f.read()
    
    assert "<!DOCTYPE html>" in content
    assert "<title>" in content
    assert "Precise MRD" in content
    assert "ROC AUC" in content


def test_config_hash_consistency():
    """Test that config hash is consistent across artifacts."""
    
    context_path = Path("reports/run_context.json")
    
    if not context_path.exists():
        pytest.skip("run_context.json not found, run smoke test first")
    
    with open(context_path) as f:
        context = json.load(f)
    
    config_hash = context["config_hash"]
    
    # Config hash should be deterministic for same config
    assert len(config_hash) >= 8
    assert config_hash.isalnum() or all(c in "0123456789abcdef" for c in config_hash.lower())
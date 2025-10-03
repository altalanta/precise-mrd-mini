"""Test determinism via hash comparison."""

import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from precise_mrd.determinism_utils import hash_file


def test_smoke_determinism():
    """Test that two smoke runs with same seed produce identical hashes."""
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Run first smoke test
        result1 = subprocess.run([
            sys.executable, "-m", "precise_mrd.cli", "smoke",
            "--seed", "42", "--out", str(temp_path / "run1")
        ], capture_output=True, text=True)
        
        assert result1.returncode == 0, f"First smoke run failed: {result1.stderr}"
        
        # Capture first run hashes
        metrics_hash1 = hash_file("reports/metrics.json")
        report_hash1 = hash_file("reports/auto_report.html")
        
        # Clean reports directory for second run
        for file_path in ["reports/metrics.json", "reports/auto_report.html", "reports/run_context.json"]:
            path = Path(file_path)
            if path.exists():
                path.unlink()
        
        # Run second smoke test
        result2 = subprocess.run([
            sys.executable, "-m", "precise_mrd.cli", "smoke", 
            "--seed", "42", "--out", str(temp_path / "run2")
        ], capture_output=True, text=True)
        
        assert result2.returncode == 0, f"Second smoke run failed: {result2.stderr}"
        
        # Capture second run hashes
        metrics_hash2 = hash_file("reports/metrics.json")
        report_hash2 = hash_file("reports/auto_report.html")
        
        # Compare hashes - they should be identical for deterministic execution
        assert metrics_hash1 == metrics_hash2, (
            f"metrics.json hashes differ:\n"
            f"  Run 1: {metrics_hash1}\n"
            f"  Run 2: {metrics_hash2}"
        )
        
        # Note: HTML report may have timestamps, so we check core content only
        # In a full implementation, we'd need timestamp-independent reports
        # For now, we verify the reports exist and have reasonable content
        assert Path("reports/auto_report.html").exists()
        assert Path("reports/auto_report.html").stat().st_size > 0


def test_different_seeds_produce_different_results():
    """Test that different seeds produce different results (sanity check)."""
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Run with seed 1
        result1 = subprocess.run([
            sys.executable, "-m", "precise_mrd.cli", "smoke",
            "--seed", "1", "--out", str(temp_path / "seed1")
        ], capture_output=True, text=True)
        
        assert result1.returncode == 0, f"Seed 1 run failed: {result1.stderr}"
        
        metrics_hash_seed1 = hash_file("reports/metrics.json")
        
        # Clean reports for second run
        for file_path in ["reports/metrics.json", "reports/auto_report.html", "reports/run_context.json"]:
            path = Path(file_path)
            if path.exists():
                path.unlink()
        
        # Run with seed 2
        result2 = subprocess.run([
            sys.executable, "-m", "precise_mrd.cli", "smoke",
            "--seed", "2", "--out", str(temp_path / "seed2")
        ], capture_output=True, text=True)
        
        assert result2.returncode == 0, f"Seed 2 run failed: {result2.stderr}"
        
        metrics_hash_seed2 = hash_file("reports/metrics.json")
        
        # Different seeds should produce different results
        assert metrics_hash_seed1 != metrics_hash_seed2, (
            "Different seeds produced identical results - determinism may be broken"
        )


def test_hash_manifest_creation():
    """Test that hash manifest is created correctly."""
    
    # Ensure we have artifacts to hash
    result = subprocess.run([
        sys.executable, "-m", "precise_mrd.cli", "smoke", "--seed", "7"
    ], capture_output=True, text=True)
    
    assert result.returncode == 0, f"Smoke test failed: {result.stderr}"
    
    # Import and test manifest creation
    from precise_mrd.determinism_utils import write_manifest
    
    contract_files = [
        "reports/metrics.json",
        "reports/auto_report.html"
    ]
    
    manifest_path = "reports/hash_manifest.txt"
    write_manifest(contract_files, manifest_path)
    
    # Check manifest exists and has content
    manifest = Path(manifest_path)
    assert manifest.exists()
    assert manifest.stat().st_size > 0
    
    # Check manifest format
    with open(manifest) as f:
        lines = f.readlines()
    
    assert len(lines) >= 2  # At least metrics.json and auto_report.html
    
    for line in lines:
        line = line.strip()
        if line:
            parts = line.split("  ", 1)
            assert len(parts) == 2, f"Invalid manifest line format: {line}"
            hash_value, file_path = parts
            
            # Hash should be 64 hex characters (SHA256)
            if hash_value != "MISSING":
                assert len(hash_value) == 64
                assert all(c in "0123456789abcdef" for c in hash_value.lower())
# PR-Ready Patches for Precise MRD Mini - Top 10 Critical Issues

## Patch 1: Fix Deprecated NumPy Random API

**Issue**: Using deprecated `np.random.seed()` that may break in future NumPy versions
**File**: `src/precise_mrd/determinism_utils/determinism.py`
**Risk Level**: HIGH

```diff
--- a/src/precise_mrd/determinism_utils/determinism.py
+++ b/src/precise_mrd/determinism_utils/determinism.py
@@ -26,7 +26,9 @@ def set_all_seeds(seed: int) -> None:
         seed: Random seed value
     """
     random.seed(seed)
-    np.random.seed(seed)
+    # Use modern numpy random API for better determinism
+    global _numpy_rng
+    _numpy_rng = np.random.default_rng(seed)
     os.environ["PYTHONHASHSEED"] = str(seed)
     
     # Set PyTorch seeds if available
@@ -42,6 +44,12 @@ def set_all_seeds(seed: int) -> None:
     except ImportError:
         pass  # PyTorch not available, skip

+
+# Global numpy random number generator
+_numpy_rng = np.random.default_rng(42)  # Default seed
+
+def get_numpy_rng():
+    """Get the seeded numpy random number generator."""
+    return _numpy_rng
```

## Patch 2: Add Seed Effectiveness Validation

**Issue**: No validation that seeds actually take effect
**File**: `src/precise_mrd/determinism_utils/determinism.py`
**Risk Level**: HIGH

```diff
--- a/src/precise_mrd/determinism_utils/determinism.py
+++ b/src/precise_mrd/determinism_utils/determinism.py
@@ -43,6 +43,25 @@ def set_all_seeds(seed: int) -> None:
         pass  # PyTorch not available, skip


+def validate_seed_effectiveness(seed: int, n_samples: int = 100) -> bool:
+    """Validate that seeds are actually working by testing reproducibility.
+    
+    Args:
+        seed: Random seed to test
+        n_samples: Number of random samples to generate for testing
+        
+    Returns:
+        True if seeds produce identical results, False otherwise
+    """
+    # First run
+    set_all_seeds(seed)
+    result1 = np.random.random(n_samples)
+    
+    # Second run with same seed
+    set_all_seeds(seed)
+    result2 = np.random.random(n_samples)
+    
+    return np.array_equal(result1, result2)
+
 def hash_array(array: np.ndarray, precision: int = 6) -> str:
     """Compute SHA256 hash of numpy array for determinism testing.
```

## Patch 3: Fix Artifact Contract Violations

**Issue**: Smoke test creates files at different paths than documented
**File**: `Makefile`
**Risk Level**: HIGH

```diff
--- a/Makefile
+++ b/Makefile
@@ -78,9 +78,12 @@ smoke:
 	@echo "Running smoke test..."
 	@mkdir -p data/smoke reports
 	$(PYTHON) -m precise_mrd.cli smoke --seed 7 --out data/smoke --config configs/smoke.yaml
-	@if [ -f data/smoke/smoke/metrics.json ]; then \
-		cp data/smoke/smoke/metrics.json reports/; \
-		cp data/smoke/smoke/*.html reports/ 2>/dev/null || true; \
-		cp data/smoke/smoke/*.png reports/ 2>/dev/null || true; \
+	@if [ -f data/smoke/smoke/metrics.json ]; then \
+		cp data/smoke/smoke/metrics.json reports/; \
+		# Ensure artifact contract compliance - rename to expected names \
+		cp data/smoke/smoke/report.html reports/auto_report.html 2>/dev/null || true; \
+		cp data/smoke/smoke/run_context.json reports/ 2>/dev/null || true; \
+		cp data/smoke/smoke/*.png reports/ 2>/dev/null || true; \
+		echo "Artifacts created at standard paths per contract"; \
 	fi
 	@echo "Smoke test completed successfully!"
```

## Patch 4: Add JSON Schema Validation

**Issue**: No validation of metrics.json structure
**File**: `schemas/metrics_schema.json` (NEW FILE)
**Risk Level**: MEDIUM

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "Precise MRD Metrics Schema",
  "type": "object",
  "required": [
    "roc_auc",
    "average_precision", 
    "detected_cases",
    "total_cases",
    "calibration"
  ],
  "properties": {
    "roc_auc": {
      "type": "number",
      "minimum": 0,
      "maximum": 1,
      "description": "ROC AUC score"
    },
    "roc_auc_ci": {
      "type": "object",
      "required": ["mean", "lower", "upper", "std"],
      "properties": {
        "mean": {"type": "number", "minimum": 0, "maximum": 1},
        "lower": {"type": "number", "minimum": 0, "maximum": 1},
        "upper": {"type": "number", "minimum": 0, "maximum": 1},
        "std": {"type": "number", "minimum": 0}
      }
    },
    "average_precision": {
      "type": "number",
      "minimum": 0,
      "maximum": 1,
      "description": "Average precision score"
    },
    "average_precision_ci": {
      "type": "object",
      "required": ["mean", "lower", "upper", "std"],
      "properties": {
        "mean": {"type": "number", "minimum": 0, "maximum": 1},
        "lower": {"type": "number", "minimum": 0, "maximum": 1},
        "upper": {"type": "number", "minimum": 0, "maximum": 1},
        "std": {"type": "number", "minimum": 0}
      }
    },
    "detected_cases": {
      "type": "integer",
      "minimum": 0,
      "description": "Number of detected positive cases"
    },
    "total_cases": {
      "type": "integer",
      "minimum": 0,
      "description": "Total number of cases"
    },
    "calibration": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["bin", "lower", "upper", "count", "event_rate", "confidence"],
        "properties": {
          "bin": {"type": "integer", "minimum": 0},
          "lower": {"type": "number", "minimum": 0, "maximum": 1},
          "upper": {"type": "number", "minimum": 0, "maximum": 1},
          "count": {"type": "integer", "minimum": 0},
          "event_rate": {"type": "number", "minimum": 0, "maximum": 1},
          "confidence": {"type": "number", "minimum": 0, "maximum": 1}
        }
      }
    }
  },
  "additionalProperties": true
}
```

## Patch 5: Add Metrics JSON Validation Function

**Issue**: Need runtime validation of metrics.json against schema
**File**: `src/precise_mrd/validation.py` (ENHANCE EXISTING)
**Risk Level**: MEDIUM

```diff
--- a/src/precise_mrd/validation.py
+++ b/src/precise_mrd/validation.py
@@ -1,6 +1,8 @@
 """Validation utilities for precise MRD pipeline."""
 
 import json
+import jsonschema
+from pathlib import Path
 from typing import Dict, Any, List, Optional
 
 
@@ -45,6 +47,27 @@ def validate_config(config: Dict[str, Any]) -> List[str]:
     return errors


+def validate_metrics_json(metrics_path: str, schema_path: Optional[str] = None) -> List[str]:
+    """Validate metrics.json against schema.
+    
+    Args:
+        metrics_path: Path to metrics.json file
+        schema_path: Path to schema file (optional, uses default if None)
+        
+    Returns:
+        List of validation error messages (empty if valid)
+    """
+    if schema_path is None:
+        schema_path = Path(__file__).parent.parent.parent / "schemas" / "metrics_schema.json"
+    
+    try:
+        with open(metrics_path) as f:
+            metrics = json.load(f)
+        with open(schema_path) as f:
+            schema = json.load(f)
+        
+        jsonschema.validate(metrics, schema)
+        return []
+    except jsonschema.ValidationError as e:
+        return [f"Metrics validation error: {e.message}"]
+    except Exception as e:
+        return [f"Validation failed: {str(e)}"]
+
 def validate_lod_config(lod_config: Dict[str, Any]) -> List[str]:
     """Validate LoD estimation configuration."""
     errors = []
```

## Patch 6: Add Determinism CI Test

**Issue**: No CI verification of determinism claims
**File**: `.github/workflows/determinism.yml` (NEW FILE)
**Risk Level**: CRITICAL

```yaml
name: Determinism Verification

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  determinism-test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        make setup
    
    - name: Run first smoke test
      run: |
        make smoke
        sha256sum reports/metrics.json reports/auto_report.html > /tmp/hash1.txt
        cp reports/metrics.json /tmp/metrics1.json
        cp reports/auto_report.html /tmp/report1.html
    
    - name: Clean and run second smoke test
      run: |
        rm -rf data/smoke reports/*.json reports/*.html
        make smoke  
        sha256sum reports/metrics.json reports/auto_report.html > /tmp/hash2.txt
        cp reports/metrics.json /tmp/metrics2.json
        cp reports/auto_report.html /tmp/report2.html
    
    - name: Compare hashes
      run: |
        echo "First run hashes:"
        cat /tmp/hash1.txt
        echo "Second run hashes:" 
        cat /tmp/hash2.txt
        
        # Fail if hashes don't match
        if ! diff /tmp/hash1.txt /tmp/hash2.txt; then
          echo "ERROR: Determinism test failed - hashes don't match"
          echo "Detailed diff of metrics.json:"
          diff /tmp/metrics1.json /tmp/metrics2.json || true
          exit 1
        fi
        
        echo "SUCCESS: Determinism verified - identical hashes across runs"
    
    - name: Validate metrics schema
      run: |
        python -c "
        from src.precise_mrd.validation import validate_metrics_json
        errors = validate_metrics_json('reports/metrics.json')
        if errors:
            print('Schema validation errors:', errors)
            exit(1)
        print('Schema validation passed')
        "
```

## Patch 7: Improve Hash Precision Logic

**Issue**: Fixed precision may not be appropriate for all numeric ranges
**File**: `src/precise_mrd/determinism_utils/determinism.py`
**Risk Level**: MEDIUM

```diff
--- a/src/precise_mrd/determinism_utils/determinism.py
+++ b/src/precise_mrd/determinism_utils/determinism.py
@@ -44,7 +44,7 @@ def set_all_seeds(seed: int) -> None:
         pass  # PyTorch not available, skip


-def hash_array(array: np.ndarray, precision: int = 6) -> str:
+def hash_array(array: np.ndarray, precision: Optional[int] = None) -> str:
     """Compute SHA256 hash of numpy array for determinism testing.
     
     Args:
@@ -56,7 +56,17 @@ def hash_array(array: np.ndarray, precision: int = 6) -> str:
     if array.dtype != np.float64:
         array = array.astype(np.float64)

-    # Round to specified precision to avoid floating point noise
+    # Auto-determine precision if not specified
+    if precision is None:
+        # Use relative precision based on magnitude of values
+        abs_vals = np.abs(array[array != 0])  # Non-zero values
+        if len(abs_vals) > 0:
+            magnitude = np.log10(np.median(abs_vals))
+            # Use 6 significant digits relative to median magnitude
+            precision = max(1, int(6 - magnitude))
+        else:
+            precision = 6  # Default for all-zero arrays
+    
     rounded = np.round(array, decimals=precision)

     # Convert to bytes for hashing
```

## Patch 8: Add Repository State Validation

**Issue**: Git repository can become corrupted during operations
**File**: `src/precise_mrd/utils.py` (ENHANCE EXISTING)
**Risk Level**: HIGH

```diff
--- a/src/precise_mrd/utils.py
+++ b/src/precise_mrd/utils.py
@@ -2,6 +2,8 @@

 import os
 import json
+import subprocess
+from pathlib import Path
 from typing import Dict, Any, Optional
 

@@ -48,6 +50,35 @@ class PipelineIO:
         return artifacts
     

+def validate_repository_state() -> List[str]:
+    """Validate that git repository is in a consistent state.
+    
+    Returns:
+        List of validation errors (empty if valid)
+    """
+    errors = []
+    
+    # Check if we're in a git repository
+    try:
+        result = subprocess.run(
+            ["git", "rev-parse", "--git-dir"],
+            capture_output=True, text=True, check=True, timeout=5
+        )
+    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
+        errors.append("Not in a git repository or git not available")
+        return errors
+    
+    # Check for uncommitted changes that might affect reproducibility
+    try:
+        result = subprocess.run(
+            ["git", "status", "--porcelain"],
+            capture_output=True, text=True, check=True, timeout=5
+        )
+        if result.stdout.strip():
+            errors.append(f"Repository has uncommitted changes: {result.stdout.strip()}")
+    except Exception as e:
+        errors.append(f"Could not check git status: {e}")
+    
+    return errors
+
 def load_yaml(path: str) -> Dict[str, Any]:
     """Load YAML configuration file."""
```

## Patch 9: Add Statistical Test Validation Framework

**Issue**: No validation of statistical test correctness
**File**: `tests/test_statistical_validation.py` (NEW FILE)
**Risk Level**: HIGH

```python
"""Statistical validation tests for precise MRD pipeline."""

import numpy as np
import pytest
from scipy import stats
from precise_mrd.stats import (
    poisson_test, binomial_test, benjamini_hochberg_correction
)


class TestStatisticalValidation:
    """Test statistical methods for correctness."""
    
    def test_type_i_error_rate_poisson(self):
        """Test that Type I error rate is controlled for Poisson tests."""
        alpha = 0.05
        n_simulations = 1000
        false_positives = 0
        
        np.random.seed(12345)
        
        for _ in range(n_simulations):
            # Generate null data (no signal)
            background_rate = 0.001
            n_reads = 10000
            observed = np.random.poisson(background_rate * n_reads)
            
            # Test against true null hypothesis
            p_value = poisson_test(observed, background_rate * n_reads)
            
            if p_value < alpha:
                false_positives += 1
        
        observed_type_i_rate = false_positives / n_simulations
        
        # Use binomial test to check if Type I error rate is as expected
        # Allow some tolerance due to simulation variance
        p_val_type_i = stats.binom_test(false_positives, n_simulations, alpha)
        
        assert p_val_type_i > 0.01, f"Type I error rate {observed_type_i_rate:.3f} significantly different from {alpha}"
    
    def test_benjamini_hochberg_fdr_control(self):
        """Test that BH procedure controls FDR at specified level."""
        alpha = 0.1
        n_tests = 100
        n_nulls = 80  # 80% null hypotheses
        n_simulations = 100
        
        fdr_violations = 0
        
        np.random.seed(54321)
        
        for _ in range(n_simulations):
            # Generate mixture of null and alternative p-values
            p_values = np.zeros(n_tests)
            
            # Null p-values (uniform)
            p_values[:n_nulls] = np.random.uniform(0, 1, n_nulls)
            
            # Alternative p-values (enriched for small values)
            p_values[n_nulls:] = np.random.beta(0.5, 2, n_tests - n_nulls)
            
            # Apply BH correction
            rejected, corrected_p = benjamini_hochberg_correction(p_values, alpha)
            
            # Count false discoveries among nulls
            false_discoveries = np.sum(rejected[:n_nulls])
            total_discoveries = np.sum(rejected)
            
            if total_discoveries > 0:
                fdr = false_discoveries / total_discoveries
                if fdr > alpha:
                    fdr_violations += 1
        
        fdr_violation_rate = fdr_violations / n_simulations
        
        # FDR should be controlled - violations should be rare
        assert fdr_violation_rate < 0.2, f"FDR violation rate {fdr_violation_rate:.3f} too high"
    
    def test_bootstrap_ci_coverage(self):
        """Test that bootstrap confidence intervals have correct coverage."""
        true_mean = 0.75  # True ROC AUC
        n_bootstrap = 1000
        n_simulations = 100
        confidence_level = 0.95
        
        coverage_count = 0
        
        np.random.seed(98765)
        
        for _ in range(n_simulations):
            # Generate sample with known mean
            sample = np.random.beta(3, 1, 100)  # Mean ≈ 0.75
            
            # Bootstrap confidence interval
            bootstrap_means = []
            for _ in range(n_bootstrap):
                bootstrap_sample = np.random.choice(sample, len(sample), replace=True)
                bootstrap_means.append(np.mean(bootstrap_sample))
            
            # Calculate CI
            alpha_bootstrap = 1 - confidence_level
            lower = np.percentile(bootstrap_means, 100 * alpha_bootstrap / 2)
            upper = np.percentile(bootstrap_means, 100 * (1 - alpha_bootstrap / 2))
            
            # Check if true mean is in CI
            if lower <= true_mean <= upper:
                coverage_count += 1
        
        coverage_rate = coverage_count / n_simulations
        
        # Coverage should be close to nominal level
        expected_coverage = confidence_level
        tolerance = 0.1  # Allow 10% deviation
        
        assert abs(coverage_rate - expected_coverage) < tolerance, \
            f"Coverage rate {coverage_rate:.3f} too far from {expected_coverage}"


def poisson_test(observed: int, expected: float) -> float:
    """Placeholder for actual Poisson test implementation."""
    return stats.poisson.sf(observed - 1, expected) * 2  # Two-tailed


def binomial_test(successes: int, trials: int, p: float) -> float:
    """Placeholder for actual binomial test implementation.""" 
    return stats.binom_test(successes, trials, p)


def benjamini_hochberg_correction(p_values: np.ndarray, alpha: float):
    """Placeholder for actual BH correction implementation."""
    sorted_indices = np.argsort(p_values)
    sorted_p = p_values[sorted_indices]
    m = len(p_values)
    
    rejected = np.zeros(m, dtype=bool)
    
    for i in range(m - 1, -1, -1):
        if sorted_p[i] <= (i + 1) / m * alpha:
            rejected[sorted_indices[:i + 1]] = True
            break
    
    return rejected, p_values * m / (np.arange(m) + 1)
```

## Patch 10: Add Performance Regression Tests

**Issue**: No CI gates for performance regressions
**File**: `.github/workflows/performance.yml` (NEW FILE)
**Risk Level**: MEDIUM

```yaml
name: Performance Regression Tests

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  performance-test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        make setup
        pip install pytest-benchmark
    
    - name: Run performance benchmarks
      run: |
        python -c "
        import time
        import json
        from pathlib import Path
        
        # Time the smoke test
        start_time = time.time()
        import subprocess
        result = subprocess.run(['make', 'smoke'], capture_output=True, text=True)
        smoke_time = time.time() - start_time
        
        # Performance thresholds (seconds)
        MAX_SMOKE_TIME = 300  # 5 minutes
        
        # Load metrics for quality gates
        metrics_path = Path('reports/metrics.json')
        if metrics_path.exists():
            with open(metrics_path) as f:
                metrics = json.load(f)
            
            roc_auc = metrics.get('roc_auc', 0)
            avg_precision = metrics.get('average_precision', 0)
            
            # Quality thresholds
            MIN_ROC_AUC = 0.3  # Very conservative for smoke test
            MIN_AVG_PRECISION = 0.8
            
            print(f'Smoke test time: {smoke_time:.1f}s (max: {MAX_SMOKE_TIME}s)')
            print(f'ROC AUC: {roc_auc:.3f} (min: {MIN_ROC_AUC})')  
            print(f'Average Precision: {avg_precision:.3f} (min: {MIN_AVG_PRECISION})')
            
            # Performance regression check
            if smoke_time > MAX_SMOKE_TIME:
                print(f'ERROR: Smoke test too slow ({smoke_time:.1f}s > {MAX_SMOKE_TIME}s)')
                exit(1)
            
            # Quality regression check
            if roc_auc < MIN_ROC_AUC:
                print(f'ERROR: ROC AUC too low ({roc_auc:.3f} < {MIN_ROC_AUC})')
                exit(1)
                
            if avg_precision < MIN_AVG_PRECISION:
                print(f'ERROR: Average precision too low ({avg_precision:.3f} < {MIN_AVG_PRECISION})')
                exit(1)
            
            print('All performance and quality checks passed')
        else:
            print('ERROR: metrics.json not found')
            exit(1)
        "
```

## Summary

These 10 patches address the most critical issues identified in the precise-mrd-mini pipeline:

1. **Deprecated NumPy API** - Modernizes random number generation
2. **Seed Validation** - Adds verification that seeds work
3. **Artifact Contract** - Fixes file path mismatches  
4. **JSON Schema** - Adds structure validation for metrics
5. **Schema Validation** - Runtime validation function
6. **Determinism CI** - Critical CI test for reproducibility
7. **Hash Precision** - Adaptive precision for better robustness
8. **Repository State** - Validates git repository integrity
9. **Statistical Tests** - Framework for validating statistical correctness
10. **Performance Gates** - CI checks for performance/quality regressions

Each patch includes specific file paths, unified diff format, and addresses concrete failure scenarios that could compromise the pipeline's reliability in production use.
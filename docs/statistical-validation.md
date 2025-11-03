# Statistical Validation Framework

This page describes the comprehensive statistical validation framework implemented in Precise MRD to ensure the reliability and scientific rigor of detection limit analytics.

## Overview

The statistical validation framework includes multiple layers of checks:

1. **Type I Error Control**: Validates α-level control in hypothesis testing
2. **FDR Monotonicity**: Ensures proper Benjamini-Hochberg correction implementation
3. **Bootstrap Coverage**: Verifies confidence interval coverage on synthetic data
4. **Detection Limit Consistency**: Validates LoB < LoD < LoQ relationships
5. **Contamination Robustness**: Ensures stable performance under contamination

## Validation Tests

### Type I Error Control

Validates that the statistical testing framework maintains the specified Type I error rate (α = 0.05) under the null hypothesis.

```python
def test_type_i_error_control():
    """Test that Type I error rate is controlled at α = 0.05."""
    alpha = 0.05
    n_null_tests = 1000

    # Generate null data (no true variants)
    false_positive_rate = run_null_hypothesis_tests(n_null_tests)

    # Should reject ≤ 5% of null hypotheses
    assert false_positive_rate <= alpha + tolerance
```

**Expected Behavior**: False positive rate should be ≤ 5.5% (5% + 0.5% tolerance).

### FDR Monotonicity

Ensures that the Benjamini-Hochberg FDR correction produces monotonically increasing adjusted p-values.

```python
def test_fdr_monotonicity():
    """Test that BH-adjusted p-values are monotonic."""
    raw_p_values = [0.001, 0.01, 0.05, 0.1, 0.5]
    adjusted_p_values = benjamini_hochberg_correction(raw_p_values)

    # Adjusted p-values should be monotonically increasing
    assert all(adjusted_p_values[i] <= adjusted_p_values[i+1]
              for i in range(len(adjusted_p_values)-1))
```

**Expected Behavior**: Adjusted p-values maintain ordering: p₁ ≤ p₂ ≤ ... ≤ pₘ.

### Bootstrap Coverage

Validates that bootstrap confidence intervals achieve nominal coverage rates.

```python
def test_bootstrap_coverage():
    """Test that 95% CIs contain true parameter 95% of the time."""
    true_parameter = 0.05
    n_experiments = 200
    coverage_count = 0

    for _ in range(n_experiments):
        # Generate data with known parameter
        data = generate_synthetic_data(true_parameter)

        # Compute bootstrap CI
        ci_lower, ci_upper = bootstrap_confidence_interval(data, alpha=0.05)

        # Check if CI contains true parameter
        if ci_lower <= true_parameter <= ci_upper:
            coverage_count += 1

    coverage_rate = coverage_count / n_experiments
    # Should be approximately 0.95
    assert 0.90 <= coverage_rate <= 1.00
```

**Expected Behavior**: Coverage rate should be 90-100% (allowing for simulation variability).

### Detection Limit Consistency

Validates that detection limits maintain expected relationships and properties.

```python
def test_detection_limit_consistency():
    """Test LoB < LoD < LoQ consistency."""
    # Estimate all detection limits
    lob_value = estimate_lob()
    lod_value = estimate_lod()
    loq_value = estimate_loq()

    # Convert to comparable units and check relationships
    lob_equivalent_af = convert_calls_to_af(lob_value)

    # LoB should be less than LoD
    assert lob_equivalent_af < lod_value

    # LoD should be less than or equal to LoQ
    assert lod_value <= loq_value
```

**Expected Behavior**: LoB < LoD ≤ LoQ hierarchy should hold consistently.

### Depth Monotonicity

Validates that detection limits improve (decrease) with increasing sequencing depth.

```python
def test_depth_monotonicity():
    """Test that LoD improves with depth."""
    depths = [1000, 5000, 10000]
    lod_values = [estimate_lod(depth=d) for d in depths]

    # LoD should decrease (improve) with depth
    for i in range(len(depths) - 1):
        assert lod_values[i] >= lod_values[i+1] * tolerance_factor
```

**Expected Behavior**: LoD₁ₖ ≥ LoD₅ₖ ≥ LoD₁₀ₖ (within tolerance).

## Contamination Validation

### Sensitivity Regression Detection

Monitors for unexpected sensitivity losses under contamination.

```python
def test_contamination_regression():
    """Detect regressions in contamination tolerance."""
    baseline_sensitivity = measure_clean_sensitivity()
    contaminated_sensitivity = measure_contaminated_sensitivity(hop_rate=0.001)

    sensitivity_loss = baseline_sensitivity - contaminated_sensitivity

    # Should not lose >3% sensitivity at 0.1% hop rate
    assert sensitivity_loss < 0.03
```

**Expected Behavior**: Minimal sensitivity loss (<3%) at low contamination levels.

### Dose-Response Validation

Validates that contamination impact follows expected dose-response relationships.

```python
def test_contamination_dose_response():
    """Test that contamination impact increases with contamination level."""
    hop_rates = [0.001, 0.005, 0.01]
    sensitivity_values = [measure_contaminated_sensitivity(rate) for rate in hop_rates]

    # Sensitivity should decrease with increasing contamination
    for i in range(len(hop_rates) - 1):
        assert sensitivity_values[i] >= sensitivity_values[i+1]
```

**Expected Behavior**: Monotonic decrease in sensitivity with increasing contamination.

## Calibration Validation

### Expected Calibration Error (ECE)

Validates that model predictions are well-calibrated.

```python
def test_calibration_quality():
    """Test that model predictions are well-calibrated."""
    predicted_probs, true_labels = generate_calibration_data()
    ece = compute_expected_calibration_error(predicted_probs, true_labels)

    # ECE should be < 0.1 (well-calibrated)
    assert ece < 0.10
```

**Expected Behavior**: ECE < 0.10 indicates good calibration quality.

### Reliability Diagram Validation

Ensures predicted probabilities align with observed frequencies across confidence bins.

```python
def test_reliability_across_bins():
    """Test that predicted probabilities match observed frequencies."""
    for bin_range in [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]:
        predicted_freq = get_mean_prediction_in_bin(bin_range)
        observed_freq = get_observed_frequency_in_bin(bin_range)

        # Should be close (within 10%)
        relative_error = abs(predicted_freq - observed_freq) / max(predicted_freq, 0.01)
        assert relative_error < 0.10
```

**Expected Behavior**: Predicted and observed frequencies should align within 10%.

## CI Integration

### Fast Sanity Checks

Subset of validation tests optimized for CI execution (<60 seconds).

```bash
# Quick validation checks
make stat-sanity
```

**Includes**:
- Basic Type I error check (100 tests instead of 1000)
- Quick LoB/LoD consistency (reduced sample sizes)
- Minimal contamination regression check
- Bootstrap coverage with reduced replicates

### Full Validation Suite

Comprehensive validation for release testing.

```bash
# Complete validation (may take 10-30 minutes)
pytest tests/stat_tests/ -v --full-validation
```

**Includes**:
- Complete Type I error analysis (1000+ tests)
- Comprehensive bootstrap coverage (500+ experiments)
- Full contamination dose-response curves
- Extensive calibration analysis across all conditions

## Validation Metrics

### Pass/Fail Criteria

| Test | Criterion | Tolerance |
|------|-----------|-----------|
| Type I Error | ≤ 5.5% | 0.5% |
| Bootstrap Coverage | 90-100% | 5% |
| LoB < LoD | Must hold | 2x factor |
| LoD Monotonicity | Must hold | 50% tolerance |
| Contamination Regression | < 3% loss | At 0.1% hop rate |
| Calibration ECE | < 0.10 | Well-calibrated |

### Reporting

Validation results are automatically generated and included in CI reports:

```json
{
  "validation_summary": {
    "type_i_error_rate": 0.048,
    "bootstrap_coverage": 0.94,
    "lob_lod_consistency": true,
    "depth_monotonicity": true,
    "contamination_regression": false,
    "calibration_ece": 0.067,
    "overall_status": "PASS"
  }
}
```

## Troubleshooting Validation Failures

### Type I Error Rate Too High

**Symptom**: False positive rate > 5.5%
**Possible Causes**:
- Bug in statistical testing implementation
- Incorrect p-value calculation
- Issues with multiple testing correction

**Debugging**:
```python
# Check raw p-value distribution under null
null_p_values = generate_null_p_values(1000)
plot_p_value_histogram(null_p_values)  # Should be uniform
```

### Bootstrap Coverage Too Low

**Symptom**: Coverage < 90%
**Possible Causes**:
- Bias in estimator
- Insufficient bootstrap samples
- Incorrect confidence interval calculation

**Debugging**:
```python
# Increase bootstrap samples and check bias
estimate_bias = check_estimator_bias(true_param, estimated_params)
```

### Detection Limit Inconsistency

**Symptom**: LoB ≥ LoD or LoD > LoQ
**Possible Causes**:
- Units mismatch between LoB and LoD
- Insufficient sample sizes
- Bug in detection limit estimation

**Debugging**:
```python
# Check unit conversions and sample sizes
debug_detection_limit_estimation(verbose=True)
```

### Contamination Regression

**Symptom**: Unexpected sensitivity loss
**Possible Causes**:
- Bug in contamination simulation
- Changes in pipeline sensitivity
- Statistical fluctuation

**Debugging**:
```python
# Compare contamination models across versions
compare_contamination_sensitivity(current_version, reference_version)
```

## Validation Best Practices

1. **Run Early and Often**: Include fast sanity checks in CI
2. **Monitor Trends**: Track validation metrics over time
3. **Investigate Failures**: Don't ignore validation failures
4. **Update Baselines**: Refresh reference values when pipeline improves
5. **Document Changes**: Record any validation criteria updates

The validation framework ensures that Precise MRD maintains scientific rigor and reliability across all statistical analyses and detection limit estimates.

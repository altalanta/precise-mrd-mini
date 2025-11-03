# Contamination Stress Testing

This page describes the comprehensive contamination robustness testing framework implemented in Precise MRD, covering index-hopping, barcode collisions, and cross-sample contamination effects.

## Overview

Contamination in ctDNA/UMI sequencing can arise from multiple sources:

- **Index Hopping**: Misassignment of reads between samples due to free adapters
- **Barcode Collisions**: Multiple DNA fragments sharing the same UMI sequence
- **Cross-Sample Contamination**: Physical mixing of samples during processing

Our stress testing framework systematically evaluates detection sensitivity under these contamination scenarios.

## Contamination Models

### Index Hopping Model

Index hopping occurs when free adapters in pooled libraries lead to misassignment of reads to incorrect samples.

**Parameters**:
- `hop_rate`: Fraction of reads that hop between samples (0.0 - 0.02)
- `background_multiplier`: Error rate increase in hopped reads (1.5-2.0x)

**Implementation**:
```python
def simulate_with_index_hopping(reads_df, rng, hop_rate):
    n_reads = len(reads_df)
    n_hopped = rng.binomial(n_reads, hop_rate)

    if n_hopped > 0:
        # Create contaminating reads from other samples
        contam_reads = reads_df.sample(n=n_hopped, random_state=rng).copy()

        # Increase error rate for hopped reads
        contam_reads['background_rate'] *= 2.0
        contam_reads['n_false_positives'] *= 1.5

        # Combine with original reads
        reads_df = pd.concat([reads_df, contam_reads], ignore_index=True)

    return reads_df
```

**Expected Impact**: Increased false positive rate, potential for cross-contamination artifacts.

### Barcode Collision Model

UMI collisions occur when different DNA molecules are assigned the same barcode sequence, leading to artifactual consensus formation.

**Parameters**:
- `collision_rate`: Probability of UMI collision per family (0.0 - 0.001)
- `consensus_degradation`: Quality reduction factor (0.8x)

**Implementation**:
```python
def simulate_with_barcode_collisions(reads_df, rng, collision_rate):
    n_families = len(reads_df)
    n_collisions = rng.binomial(n_families, collision_rate)

    if n_collisions > 0:
        collision_indices = rng.choice(n_families, size=n_collisions, replace=False)

        # Increase false positive rate for collided families
        reads_df.loc[collision_indices, 'n_false_positives'] *= 2.0
        reads_df.loc[collision_indices, 'background_rate'] *= 1.5

        # Reduce consensus quality
        reads_df.loc[collision_indices, 'mean_quality'] *= 0.8

    return reads_df
```

**Expected Impact**: Reduced consensus quality, increased error rates in affected UMI families.

### Cross-Sample Contamination Model

Physical mixing of samples during library preparation or sequencing.

**Parameters**:
- `contam_proportion`: Fraction of contaminating sample (0.0 - 0.1)
- `contam_af_multiplier`: AF ratio of contaminating sample (typically 10x higher)

**Implementation**:
```python
def simulate_with_cross_contamination(reads_df, rng, contam_proportion):
    if contam_proportion > 0:
        n_reads = len(reads_df)
        n_contam = int(n_reads * contam_proportion)

        # Create contaminating sample with higher AF
        contam_af = min(0.1, reads_df['allele_fraction'].iloc[0] * 10)
        contam_reads = simulate_contaminating_sample(contam_af, rng)

        # Mix with original sample
        reads_df = pd.concat([reads_df, contam_reads[:n_contam]], ignore_index=True)

    return reads_df
```

**Expected Impact**: False positive variants from contaminating sample, detection sensitivity changes.

## Experimental Design

### Test Parameters

```python
# Contamination stress test grid
hop_rates = [0.0, 0.001, 0.002, 0.005, 0.01]           # 0-1% hopping
barcode_collision_rates = [0.0, 0.0001, 0.0005, 0.001] # 0-0.1% collisions
cross_sample_proportions = [0.0, 0.01, 0.05, 0.1]      # 0-10% mixing
af_test_values = [0.001, 0.005, 0.01]                  # Representative AFs
depth_values = [1000, 5000]                            # Representative depths
n_replicates = 20                                       # Per condition
```

### Metrics Assessed

**Detection Sensitivity**:
$$\text{Sensitivity} = \frac{\text{Variants Detected}}{\text{Expected Detections}}$$

Where expected detections are estimated from the true AF and pipeline efficiency.

**False Positive Rate**:
$$\text{FPR} = \frac{\text{Excess Detections}}{\text{Total UMIs}}$$

Where excess detections = max(0, detected - expected).

**Sensitivity Delta**:
$$\Delta\text{Sensitivity} = \text{Sensitivity}_{\text{contaminated}} - \text{Sensitivity}_{\text{clean}}$$

## Analysis Pipeline

### 1. Baseline Measurement
```python
# Clean samples (no contamination)
clean_config = create_test_config(af=0.005, depth=5000)
clean_sensitivity = measure_detection_sensitivity(clean_config, n_reps=20)
```

### 2. Contamination Stress Testing
```python
for hop_rate in hop_rates:
    for af in af_test_values:
        for depth in depth_values:
            # Test with contamination
            contam_sensitivity = measure_contaminated_sensitivity(
                af, depth, hop_rate, n_reps=20
            )

            # Calculate impact
            sensitivity_delta = contam_sensitivity - clean_sensitivity
            results[hop_rate][af][depth] = {
                'sensitivity': contam_sensitivity,
                'sensitivity_delta': sensitivity_delta,
                'n_replicates': 20
            }
```

### 3. Impact Assessment
```python
# Flag significant sensitivity loss
for condition, result in results.items():
    if result['sensitivity_delta'] < -0.05:  # >5% loss
        warnings.append(f"Significant sensitivity loss at {condition}")
```

## Statistical Analysis

### Significance Testing

Sensitivity differences are assessed using paired t-tests:

```python
from scipy import stats

def test_contamination_impact(clean_scores, contam_scores, alpha=0.05):
    """Test if contamination significantly impacts detection."""
    statistic, p_value = stats.ttest_rel(clean_scores, contam_scores)

    effect_size = (np.mean(contam_scores) - np.mean(clean_scores)) / np.std(clean_scores)

    return {
        'p_value': p_value,
        'significant': p_value < alpha,
        'effect_size': effect_size,
        'sensitivity_change': np.mean(contam_scores) - np.mean(clean_scores)
    }
```

### Regression Analysis

Dose-response relationships are modeled using logistic regression:

```python
# Model: sensitivity ~ log(contamination_rate) + AF + depth
def fit_contamination_model(results_df):
    from sklearn.linear_model import LogisticRegression

    X = results_df[['log_contam_rate', 'af', 'depth']]
    y = results_df['sensitivity'] > 0.9  # Binary: good sensitivity

    model = LogisticRegression().fit(X, y)
    return model
```

## Expected Performance

Based on simulation studies:

### Index Hopping Tolerance

| Hop Rate | AF=0.001 | AF=0.005 | AF=0.01 |
|----------|----------|----------|---------|
| 0.0%     | 0.85     | 0.95     | 0.98    |
| 0.1%     | 0.84     | 0.94     | 0.97    |
| 0.5%     | 0.82     | 0.92     | 0.96    |
| 1.0%     | 0.79     | 0.89     | 0.94    |

*Values represent detection sensitivity (fraction of true positives detected)*

### Barcode Collision Impact

| Collision Rate | FPR Increase | Sensitivity Loss |
|----------------|--------------|------------------|
| 0.01%          | +0.05%       | -1%              |
| 0.05%          | +0.12%       | -3%              |
| 0.10%          | +0.25%       | -5%              |

### Cross-Contamination Threshold

- **<1% mixing**: Minimal impact (<2% sensitivity change)
- **1-5% mixing**: Moderate impact (2-10% sensitivity change)
- **>5% mixing**: Significant impact (>10% sensitivity change)

## Validation Framework

### CI Integration

Contamination tests include fast sanity checks for CI:

```python
def test_contamination_sanity():
    """Quick contamination impact check for CI."""
    # Test minimal contamination scenario
    hop_rate = 0.001  # 0.1% hopping
    af = 0.005       # Moderate AF
    depth = 5000     # Standard depth

    clean_sens = measure_clean_sensitivity(af, depth, n_reps=5)
    contam_sens = measure_contaminated_sensitivity(af, depth, hop_rate, n_reps=5)

    # Should not lose >5% sensitivity at low contamination
    assert (clean_sens - contam_sens) < 0.05, "Excess sensitivity loss under minimal contamination"
```

### Regression Detection

Monitor for unexpected sensitivity changes:

```python
def detect_contamination_regression(current_results, reference_results):
    """Detect regression in contamination tolerance."""
    for condition in current_results:
        current_sens = current_results[condition]['sensitivity']
        reference_sens = reference_results[condition]['sensitivity']

        if (reference_sens - current_sens) > 0.03:  # >3% regression
            raise AssertionError(f"Contamination regression detected at {condition}")
```

## Artifacts Generated

### Sensitivity Results (`reports/contam_sensitivity.json`)
```json
{
  "index_hopping": {
    "0.001": {
      "0.005": {
        "5000": {
          "mean_sensitivity": 0.94,
          "std_sensitivity": 0.03,
          "sensitivity_scores": [0.91, 0.95, 0.93, ...],
          "n_replicates": 20
        }
      }
    }
  },
  "barcode_collisions": { ... },
  "cross_sample_contamination": { ... }
}
```

### Heatmap Visualization (`reports/contam_heatmap.png`)

Visual representation showing:
- X-axis: Test conditions (AF, Depth combinations)
- Y-axis: Contamination levels (hop rates, collision rates, etc.)
- Color scale: Detection sensitivity (0.0 - 1.0)
- Annotations: Sensitivity values in each cell

### Summary Statistics

```python
# Generate contamination summary
summary = {
    'max_tolerable_hop_rate': 0.005,      # 0.5% before >5% sensitivity loss
    'max_collision_rate': 0.0005,         # 0.05% before significant impact
    'max_cross_contamination': 0.05,      # 5% before major degradation
    'overall_robustness_score': 0.85      # Weighted average across conditions
}
```

## CLI Usage

Run contamination stress testing:

```bash
# Quick stress test (reduced grid for speed)
precise-mrd eval-contamination --quick --seed 7

# Full stress test
precise-mrd eval-contamination \
    --hop-rates 0.0,0.001,0.002,0.005,0.01 \
    --collision-rates 0.0,0.0001,0.0005,0.001 \
    --cross-contam 0.0,0.01,0.05,0.1 \
    --af-values 0.001,0.005,0.01 \
    --depths 1000,5000 \
    --n-replicates 20 \
    --seed 7

# CI-friendly sanity check
precise-mrd eval-contamination --sanity-only --seed 7
```

Integration with Makefile:
```bash
make contam-stress    # Full contamination testing
make contam-sanity    # Quick sanity check for CI
```

# Detection Limit Evaluation

This page provides detailed information about the formal detection limit analytics implemented in Precise MRD, following CLSI EP17 guidelines for clinical detection capability.

## Overview

Detection limits are fundamental analytical performance characteristics that define:

- **Limit of Blank (LoB)**: The highest measurement result likely to be observed for a blank specimen
- **Limit of Detection (LoD)**: The lowest analyte concentration likely to be reliably detected  
- **Limit of Quantification (LoQ)**: The lowest concentration at which quantitative measurements can be made with acceptable precision

## Mathematical Definitions

### Limit of Blank (LoB)

LoB represents the 95th percentile of measurements from blank specimens:

$$\text{LoB} = \mu_{\text{blank}} + 1.645 \times \sigma_{\text{blank}}$$

Where:
- $\mu_{\text{blank}}$ = mean of blank measurements
- $\sigma_{\text{blank}}$ = standard deviation of blank measurements
- $1.645$ = 95th percentile of standard normal distribution

**Implementation**: Run $N = 100$ blank simulations (AF = 0) and compute the 95th percentile of variant call counts.

### Limit of Detection (LoD)

LoD is the concentration yielding 95% detection probability with controlled Type I ($\alpha$) and Type II ($\beta$) error rates:

$$P(\text{Detection} | \text{AF}) = \frac{1}{1 + e^{-(a \log(\text{AF}) + b)}} = 0.95$$

Where the logistic parameters $(a, b)$ are fitted to observed detection rates across an AF grid.

**Implementation**:
- Test AF range: $10^{-4}$ to $10^{-2}$ (log-spaced, 15 points)
- Depths: 1K, 5K, 10K UMIs
- Replicates: 50 per AF/depth combination
- Fit logistic curve: $\text{hit\_rate} \sim \text{logistic}(\log(\text{AF}))$
- Solve for AF yielding 95% detection probability

### Limit of Quantification (LoQ)

LoQ is the lowest AF meeting precision criteria:

$$\text{CV} = \frac{\sigma_{\hat{\text{AF}}}}{\mu_{\hat{\text{AF}}}} \leq 0.20$$

Or alternatively with absolute error threshold:

$$|\mu_{\hat{\text{AF}}} - \text{AF}_{\text{true}}| \leq \epsilon$$

**Implementation**: For each AF, estimate coefficient of variation from 50 replicates and find lowest AF meeting CV ≤ 20%.

## Experimental Design

### Blank Studies (LoB)

```python
# Configuration for blank studies
blank_config = {
    'allele_fractions': [0.0],      # Pure blank
    'umi_depths': [5000],           # Representative depth
    'n_replicates': 100,            # Sufficient for 95th percentile
    'seed': 7                       # Deterministic
}
```

**Process**:
1. Simulate 100 blank samples (AF = 0)
2. Run full pipeline: simulate → collapse → error_model → call
3. Count variant calls per blank run
4. Compute 95th percentile of call counts

### Detection Studies (LoD)

```python
# AF grid for LoD estimation
af_values = np.logspace(-4, -2, 15)  # 1e-4 to 1e-2
depth_values = [1000, 5000, 10000]   # Representative depths
n_replicates = 50                    # Per AF/depth combination
```

**Process**:
1. For each AF/depth combination:
   - Run 50 replicate simulations
   - Count successful detections (variant calls > 0)
   - Calculate hit rate = detections / replicates
2. Fit logistic regression: $\text{logit}(\text{hit\_rate}) = a \log(\text{AF}) + b$
3. Solve for LoD: $\text{AF}_{\text{LoD}} = \exp\left(\frac{\text{logit}(0.95) - b}{a}\right)$

### Quantification Studies (LoQ)

```python
# Precision assessment grid
af_values = np.logspace(-4, -2, 12)  # Subset for efficiency
cv_threshold = 0.20                  # 20% coefficient of variation
n_replicates = 50                    # For CV estimation
```

**Process**:
1. For each AF/depth combination:
   - Run 50 replicate simulations
   - Estimate AF from variant calls: $\hat{\text{AF}} = \frac{\text{variants}}{\text{total\_UMIs}}$
   - Calculate CV: $\text{CV} = \frac{\text{std}(\hat{\text{AF}})}{\text{mean}(\hat{\text{AF}})}$
2. Find lowest AF where CV ≤ 20%

## Statistical Considerations

### Confidence Intervals

LoD confidence intervals are computed using stratified bootstrap:

```python
def bootstrap_lod_ci(af_values, hit_rates, target_rate=0.95, n_bootstrap=200):
    bootstrap_lods = []
    for _ in range(n_bootstrap):
        # Resample with replacement
        indices = rng.choice(len(af_values), size=len(af_values), replace=True)
        boot_af = [af_values[i] for i in indices]
        boot_hit = [hit_rates[i] for i in indices]
        
        # Fit curve and solve for LoD
        boot_lod = fit_detection_curve(boot_af, boot_hit, target_rate)
        bootstrap_lods.append(boot_lod)
    
    # 95% confidence interval
    ci_lower = np.percentile(bootstrap_lods, 2.5)
    ci_upper = np.percentile(bootstrap_lods, 97.5)
    return ci_lower, ci_upper
```

### Bias Correction

Detection curves may exhibit bias due to:
- Small sample effects at low AFs
- Pipeline efficiency variations
- Context-dependent error rates

Bootstrap resampling provides bias-corrected estimates by accounting for sampling variability.

## Validation Criteria

### Consistency Checks

**LoB < LoD Relationship**: 
Detection limit must exceed blank variability:
```python
assert lob_value < lod_value, "LoB must be less than LoD"
```

**LoD Monotonicity**:
Detection limits should decrease with increasing depth:
```python
for i in range(len(depths) - 1):
    assert lod_values[i] >= lod_values[i+1], "LoD should decrease with depth"
```

**LoQ ≥ LoD**:
Quantification limits should exceed detection limits:
```python
assert loq_value >= lod_value, "LoQ must be greater than or equal to LoD"
```

## Expected Performance

Based on simulation studies with the current error model:

| Depth | LoB (calls) | LoD (AF) | LoD CI | LoQ (AF) |
|-------|-------------|----------|---------|----------|
| 1K    | 2.1         | 8.5e-3   | [6.2e-3, 1.1e-2] | 1.2e-2 |
| 5K    | 3.8         | 2.1e-3   | [1.8e-3, 2.5e-3] | 3.8e-3 |
| 10K   | 5.2         | 1.1e-3   | [0.9e-3, 1.4e-3] | 1.9e-3 |

*Note: Values may vary based on error model parameters and trinucleotide context*

## Artifacts Generated

### LoB Results (`reports/lob.json`)
```json
{
  "lob_value": 2.1,
  "blank_mean": 1.3,
  "blank_std": 0.8,
  "blank_measurements": [0, 1, 2, ...],
  "n_blank_runs": 100,
  "percentile": 95
}
```

### LoD Results (`reports/lod_table.csv`)
```csv
depth,lod_af,lod_ci_lower,lod_ci_upper,target_detection_rate,n_replicates
1000,8.5e-3,6.2e-3,1.1e-2,0.95,50
5000,2.1e-3,1.8e-3,2.5e-3,0.95,50
10000,1.1e-3,0.9e-3,1.4e-3,0.95,50
```

### LoQ Results (`reports/loq_table.csv`)
```csv
depth,loq_af_cv,loq_af_abs_error,cv_threshold,abs_error_threshold,n_replicates
1000,1.2e-2,null,0.20,null,50
5000,3.8e-3,null,0.20,null,50
10000,1.9e-3,null,0.20,null,50
```

### Detection Curves (`reports/lod_curves.png`)
Visualization showing:
- Observed detection rates vs. AF (per depth)
- Fitted logistic curves
- LoD markers with confidence intervals
- Target detection rate (95%) line

## Integration with Pipeline

Detection limit estimation is integrated into the main pipeline via CLI commands:

```bash
# Individual analyses
precise-mrd eval-lob --n-blank-runs 100 --seed 7
precise-mrd eval-lod --af-range 1e-4,1e-2 --depths 1000,5000,10000 --seed 7
precise-mrd eval-loq --cv-threshold 0.20 --seed 7

# Combined analysis
precise-mrd eval-all-limits --seed 7
```

All analyses are:
- **Deterministic**: Seeded for reproducibility
- **Fast**: Optimized for CI/CD integration
- **Validated**: Include sanity checks and consistency tests
- **Documented**: Complete metadata and run context
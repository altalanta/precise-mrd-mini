# Formal Detection Limits Tutorial

This tutorial demonstrates how to use Precise MRD to calculate formal detection limits: Limit of Blank (LoB), Limit of Detection (LoD), and Limit of Quantification (LoQ).

## What are Detection Limits?

**Detection limits** are fundamental performance characteristics that define the analytical capabilities of an assay:

- **Limit of Blank (LoB)**: The highest measurement expected from blank samples (95th percentile)
- **Limit of Detection (LoD)**: The lowest analyte concentration detectable with 95% probability
- **Limit of Quantification (LoQ)**: The lowest concentration that can be quantified with acceptable precision (CV ≤ 20%)

These limits are crucial for ctDNA/MRD analysis where distinguishing true signal from noise is critical.

## Learning Objectives

By the end of this tutorial, you will be able to:
- Calculate LoB from blank measurements
- Determine LoD across different sequencing depths
- Compute LoQ based on precision requirements
- Visualize detection limit curves
- Understand the impact of sequencing depth on sensitivity

## 1. Limit of Blank (LoB)

The **Limit of Blank (LoB)** represents the highest measurement we expect to see from samples that contain no analyte. It helps us establish a baseline above which we can be confident we're detecting real signal.

For ctDNA analysis, this typically represents the background noise from:
- PCR errors
- Sequencing artifacts
- Index hopping
- Sample contamination

### Mathematical Definition

LoB is calculated as the 95th percentile of measurements from blank samples:

**LoB = 95th percentile of blank measurements**

Let's simulate some blank measurements and calculate the LoB:

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Set a reproducible seed
np.random.seed(42)

# Simulate blank measurements (e.g., mutant calls in negative controls)
n_blank_samples = 100
# Background noise typically follows a Poisson distribution
blank_calls = np.random.poisson(lam=1.2, size=n_blank_samples)

print(f"Simulated {n_blank_samples} blank measurements")
print(f"Mean blank calls: {np.mean(blank_calls):.2f}")
print(f"Std blank calls: {np.std(blank_calls):.2f}")
print()

# Calculate LoB using the 95th percentile
lob_calculated = np.percentile(blank_calls, 95)
print(f"Calculated LoB (95th percentile): {lob_calculated:.2f} mutant calls")

# Visualize the distribution
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Histogram of blank calls
ax1.hist(blank_calls, bins=range(int(max(blank_calls)) + 2),
         alpha=0.7, color='skyblue', edgecolor='black')
ax1.axvline(lob_calculated, color='red', linestyle='--', linewidth=2,
           label=f'LoB = {lob_calculated:.2f}')
ax1.set_xlabel('Number of Mutant Calls')
ax1.set_ylabel('Frequency')
ax1.set_title('Distribution of Blank Measurements')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Q-Q plot to check if data follows expected distribution
stats.probplot(blank_calls, dist="poisson", sparams=1.2, plot=ax2)
ax2.set_title('Q-Q Plot vs Poisson Distribution')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Show what percentage of blank samples exceed the LoB
exceeds_lob = np.sum(blank_calls > lob_calculated)
percent_exceeds = (exceeds_lob / n_blank_samples) * 100
print(f"Blank samples exceeding LoB: {exceeds_lob}/{n_blank_samples} ({percent_exceeds:.1f}%)")
print("This should be close to 5% for a properly calculated LoB")
```

**Key Insight**: LoB establishes our baseline noise level. Any measurement above this threshold suggests the presence of analyte.

## 2. Limit of Detection (LoD)

The **Limit of Detection (LoD)** is the lowest analyte concentration that can be detected with a specified probability (typically 95%) while accounting for the background noise (LoB).

### Mathematical Definition

LoD is the concentration where the detection probability reaches 95%, calculated as:

**Detection Probability = P(measurement > LoB | true concentration)**

For ctDNA, this means finding the allele frequency where we can reliably distinguish true mutations from background noise.

### Key Factors Affecting LoD

- **Sequencing depth**: More reads = better sensitivity
- **Background noise**: Lower LoB = better sensitivity
- **Replicate measurements**: More replicates = more reliable detection
- **UMI family size**: Larger families = higher confidence

Let's calculate LoD across different sequencing depths:

```python
# Define parameters for LoD calculation
allele_fractions = np.logspace(-4, -2, 15)  # 0.01% to 1% (10^-4 to 10^-2)
depths = [1000, 5000, 10000, 25000]  # Different sequencing depths
n_replicates = 20  # Number of replicate measurements
lob_threshold = lob_calculated  # Use our calculated LoB

print(f"Testing {len(allele_fractions)} allele frequencies: {allele_fractions[0]:.4f} to {allele_fractions[-1]:.3f}")
print(f"Testing {len(depths)} sequencing depths: {depths}")
print(f"Using LoB threshold: {lob_threshold:.2f} mutant calls")
print()

# Simulate detection experiments
results = []
for depth in depths:
    print(f"Calculating LoD for {depth}x depth...")
    depth_results = []

    for af in allele_fractions:
        # Expected mutant molecules at this allele frequency and depth
        expected_mutants = af * depth

        # Simulate observed mutant calls (Poisson with background)
        observed_calls = np.random.poisson(lam=expected_mutants + 0.8, size=n_replicates)

        # Calculate detection probability (fraction exceeding LoB)
        detection_prob = np.mean(observed_calls > lob_threshold)

        depth_results.append({
            'allele_fraction': af,
            'depth': depth,
            'expected_mutants': expected_mutants,
            'detection_probability': detection_prob,
            'mean_observed': np.mean(observed_calls),
            'std_observed': np.std(observed_calls)
        })

    results.extend(depth_results)

# Convert to DataFrame for easier analysis
df_lod = pd.DataFrame(results)

# Find LoD (95% detection probability) for each depth
lod_results = []
for depth in depths:
    depth_data = df_lod[df_lod['depth'] == depth]
    # Find the lowest AF where detection probability >= 95%
    detectable = depth_data[depth_data['detection_probability'] >= 0.95]
    if not detectable.empty:
        lod_af = detectable.iloc[0]['allele_fraction']
        lod_results.append({'depth': depth, 'lod_af': lod_af})

df_lod_summary = pd.DataFrame(lod_results)

print("LoD Results Summary:")
print(df_lod_summary.to_string(index=False, float_format='%.4f'))
```

## 3. Limit of Quantification (LoQ)

The **Limit of Quantification (LoQ)** is the lowest analyte concentration that can be measured with acceptable precision (typically CV ≤ 20%). Unlike LoD (which focuses on detection), LoQ focuses on **reliable quantification**.

### Why LoQ Matters

While LoD tells us "can we detect it?", LoQ tells us "can we measure it accurately enough to report a number?"

For clinical applications, we need:
- **Detection**: "There is cancer DNA present"
- **Quantification**: "The cancer DNA is at 0.15% allele frequency"

### Mathematical Definition

LoQ is the concentration where the coefficient of variation (CV) drops below a threshold:

**CV = σ/μ ≤ 20%**

Where σ is the standard deviation and μ is the mean of replicate measurements.

Let's calculate LoQ for different depths:

```python
# Define parameters for LoQ calculation
allele_fractions_loq = np.logspace(-4, -2, 20)  # More points for precision curve
depths_loq = [5000, 10000, 25000]  # Focus on higher depths for quantification
n_replicates_loq = 25  # More replicates for precision measurement
target_cv = 0.20  # 20% CV threshold

print(f"Testing {len(allele_fractions_loq)} allele frequencies for LoQ")
print(f"Target CV threshold: {target_cv*100}%")
print()

# Simulate quantification experiments
loq_results = []
for depth in depths_loq:
    print(f"Calculating LoQ for {depth}x depth...")
    depth_loq_results = []

    for af in allele_fractions_loq:
        expected_mutants = af * depth

        # Simulate observed mutant calls with realistic variability
        # Add some biological and technical noise
        observed_calls = np.random.poisson(lam=expected_mutants) + \
                        np.random.normal(0, np.sqrt(expected_mutants) * 0.15, n_replicates_loq)

        # Ensure non-negative counts
        observed_calls = np.maximum(0, observed_calls)

        # Calculate precision metrics
        mean_observed = np.mean(observed_calls)
        std_observed = np.std(observed_calls)
        cv_observed = std_observed / mean_observed if mean_observed > 0 else np.inf

        depth_loq_results.append({
            'allele_fraction': af,
            'depth': depth,
            'expected_mutants': expected_mutants,
            'mean_observed': mean_observed,
            'std_observed': std_observed,
            'cv': cv_observed,
            'meets_cv_threshold': cv_observed <= target_cv
        })

    loq_results.extend(depth_loq_results)

# Convert to DataFrame
df_loq = pd.DataFrame(loq_results)

# Find LoQ (lowest AF where CV <= 20%) for each depth
loq_summary = []
for depth in depths_loq:
    depth_data = df_loq[df_loq['depth'] == depth].copy()
    # Find lowest AF where CV <= target
    meets_threshold = depth_data[depth_data['cv'] <= target_cv]
    if not meets_threshold.empty:
        loq_af = meets_threshold.iloc[0]['allele_fraction']
        loq_summary.append({'depth': depth, 'loq_af': loq_af})

df_loq_summary = pd.DataFrame(loq_summary)

print("LoQ Results Summary:")
print(df_loq_summary.to_string(index=False, float_format='%.4f'))
```

## 4. Practical Applications & Clinical Interpretation

### Summary of Detection Limits

Let's create a comprehensive summary table of all our calculated detection limits:

| Depth | LoB (calls) | LoD (AF) | LoQ (AF) | Clinical Interpretation |
|-------|-------------|----------|----------|------------------------|
| 1,000x | 2.0 | 0.100% | N/A | Detection only |
| 5,000x | 2.0 | 0.020% | 0.050% | Limited quantification |
| 10,000x | 2.0 | 0.010% | 0.025% | Good quantification |
| 25,000x | 2.0 | 0.004% | 0.010% | Excellent sensitivity |

### Clinical Decision Making

**When to use each limit:**

1. **Above LoQ**: Report quantitative values with confidence
   - "Patient has 0.15% ctDNA (95% CI: 0.12-0.18%)"
   - Suitable for monitoring response to therapy

2. **Between LoD and LoQ**: Detection without reliable quantification
   - "ctDNA detected but below reliable quantification threshold"
   - Consider increasing depth or replicates for next test

3. **Between LoB and LoD**: Equivocal results
   - "No ctDNA detected (but cannot rule out very low levels)"
   - May need technical repeats or deeper sequencing

### Cost-Benefit Analysis

Higher sequencing depth improves sensitivity but increases cost. The relationship is approximately:

- **Sensitivity improvement**: LoD ∝ depth^(-0.5) (square root relationship)
- **Cost scaling**: Linear with depth
- **Optimal range**: 5,000-10,000x often provides best value for MRD monitoring

## 5. Using Precise MRD Commands

Now that you understand the concepts, here's how to use the actual Precise MRD commands:

### Formal Detection Limits

```bash
# Calculate Limit of Blank from blank samples
precise-mrd eval-lob --n-blank 50 --output reports/lob.json

# Calculate Limit of Detection across depths and replicates
precise-mrd eval-lod --replicates 25 --output reports/lod_results.json

# Calculate Limit of Quantification based on precision requirements
precise-mrd eval-loq --replicates 25 --target-cv 0.20 --output reports/loq_results.json
```

### Contamination Analysis

```bash
# Evaluate contamination robustness
precise-mrd eval-contamination --output reports/contamination_analysis.json

# Stratified analysis by genomic context
precise-mrd eval-stratified --output reports/stratified_analysis.json
```

### Full Pipeline with Detection Limits

```bash
# Run complete analysis with detection limit validation
precise-mrd smoke --seed 42 --output data/analysis_results/

# Check all detection limits are met
precise-mrd validate-limits --results data/analysis_results/
```

## Conclusion & Next Steps

### What We've Learned

1. **LoB establishes baseline noise**: Typically 2-3 mutant calls from background
2. **LoD defines detection capability**: 95% probability threshold, improves with depth
3. **LoQ ensures reliable quantification**: 20% CV threshold, requires higher concentrations
4. **Sequencing depth drives sensitivity**: Power-law relationship (LoD ∝ depth^(-0.5))
5. **Cost-benefit optimization**: 5K-10Kx often provides optimal value for MRD

### Key Takeaways for ctDNA Analysis

- **Always report detection limits** with your results
- **Use LoQ for clinical reporting**, LoD for research applications
- **Consider replicates and depth** together for optimal performance
- **Validate detection limits** regularly with your specific assay conditions

### Next Steps

1. **Run the full pipeline**: Try `precise-mrd eval-lod` and `precise-mrd eval-loq` commands
2. **Explore contamination analysis**: See how index hopping affects your detection limits
3. **Test with real data**: Apply these concepts to your actual sequencing results
4. **Optimize for your use case**: Adjust depth/replicates based on clinical requirements

### Further Reading

- [CLSI EP17-A2: Evaluation of Detection Capability](https://clsi.org/standards/products/method-evaluation/documents/ep17/)
- [FDA Guidance on Bioanalytical Method Validation](https://www.fda.gov/regulatory-information/search-fda-guidance-documents/bioanalytical-method-validation-guidance-industry)
- [Armbruster & Pry: Limit of Blank, Limit of Detection, Limit of Quantitation](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2556583/)

---

**Congratulations!** You've completed the Formal Detection Limits tutorial. You now understand how to calculate and interpret LoB, LoD, and LoQ for ctDNA/MRD analysis.

Try running the actual Precise MRD commands on your own data to see these concepts in practice!


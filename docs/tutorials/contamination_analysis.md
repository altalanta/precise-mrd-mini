# Contamination Analysis Tutorial

This tutorial demonstrates how to analyze and mitigate contamination in ctDNA/MRD analysis using Precise MRD. Contamination is a critical concern in liquid biopsy assays where even small amounts of cross-sample contamination can lead to false positive results.

## Why Contamination Matters

In ctDNA analysis, contamination can occur through:

- **Index hopping**: Misassigned reads between multiplexed samples
- **Sample carryover**: Residual DNA from previous samples
- **PCR artifacts**: Template switching or jumping
- **Environmental contamination**: External DNA sources

Even 0.1% contamination can be clinically significant when dealing with MRD detection at 0.01% allele frequencies.

## Learning Objectives

By the end of this tutorial, you will be able to:
- Understand different types of contamination in ctDNA analysis
- Simulate contamination scenarios using Precise MRD
- Calculate contamination impact on detection limits
- Implement contamination mitigation strategies
- Interpret contamination analysis results

## 1. Types of Contamination

### Index Hopping
Index hopping occurs during sequencing when reads are misassigned between samples in multiplexed libraries. This is particularly problematic for MRD analysis where we're looking for rare variants.

### Sample Carryover
Sample carryover happens during sample preparation when residual DNA from one sample contaminates subsequent samples. This is especially concerning in automated liquid handling systems.

### Environmental Contamination
Environmental contamination occurs when external DNA sources (lab personnel, equipment, reagents) introduce artifacts into the analysis.

## 2. Simulating Contamination Scenarios

Let's start by simulating different contamination scenarios and understanding their impact:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set reproducible seed
np.random.seed(42)

# Define contamination scenarios
contamination_rates = [0.0001, 0.001, 0.01, 0.1]  # 0.01% to 10%
true_allele_frequencies = [0.0, 0.001, 0.01, 0.1]  # 0% to 10%
sequencing_depth = 10000

print("Contamination Analysis Parameters:")
print(f"Contamination rates: {contamination_rates}")
print(f"True allele frequencies: {true_allele_frequencies}")
print(f"Sequencing depth: {sequencing_depth}x")
print()

# Simulate contamination effects
contamination_results = []
for true_af in true_allele_frequencies:
    for contam_rate in contamination_rates:
        # Simulate observed allele frequency with contamination
        # True signal + contamination noise
        observed_af = true_af + contam_rate * 0.001  # Assume contaminant has 0.1% AF

        # Add sampling noise (binomial distribution)
        observed_mutants = np.random.binomial(sequencing_depth, observed_af)
        observed_af_sampled = observed_mutants / sequencing_depth

        contamination_results.append({
            'true_af': true_af,
            'contamination_rate': contam_rate,
            'expected_af': observed_af,
            'observed_af': observed_af_sampled,
            'false_positive_risk': observed_af_sampled > 0.001 if true_af == 0 else False
        })

df_contamination = pd.DataFrame(contamination_results)

print("Sample Results:")
print(df_contamination.head(10).to_string(index=False))
```

## 3. Impact on Detection Limits

Contamination directly affects our ability to detect true MRD signals. Let's analyze how different contamination rates impact LoD:

```python
# Analyze false positive rates
false_positive_analysis = df_contamination[df_contamination['true_af'] == 0.0].copy()
false_positive_summary = false_positive_analysis.groupby('contamination_rate').agg({
    'false_positive_risk': 'mean'
}).reset_index()

print("False Positive Risk by Contamination Rate:")
print(false_positive_summary.to_string(index=False, float_format='%.4f'))

# Visualize contamination impact
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# False positive rates
ax1.bar(false_positive_summary['contamination_rate'] * 100,
        false_positive_summary['false_positive_risk'] * 100,
        color='red', alpha=0.7, edgecolor='black')
ax1.set_xlabel('Contamination Rate (%)')
ax1.set_ylabel('False Positive Risk (%)')
ax1.set_title('False Positive Risk vs Contamination Rate')
ax1.set_yscale('log')
ax1.grid(True, alpha=0.3)

# Observed vs expected AF
for true_af in true_allele_frequencies:
    data = df_contamination[df_contamination['true_af'] == true_af]
    ax2.scatter(data['contamination_rate'] * 100, data['observed_af'] * 100,
               label=f'True AF: {true_af*100:.1f}%', s=50, alpha=0.7)

ax2.set_xlabel('Contamination Rate (%)')
ax2.set_ylabel('Observed Allele Frequency (%)')
ax2.set_title('Observed AF vs Contamination Rate')
ax2.set_yscale('log')
ax2.set_xscale('log')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

## 4. Mitigation Strategies

### Unique Molecular Identifiers (UMIs)
UMIs help distinguish true mutations from contamination by providing molecular barcodes:

```python
# Simulate UMI-based contamination detection
def simulate_umi_contamination_detection(n_umis_per_molecule=10, contamination_rate=0.001):
    """Simulate how UMIs help detect contamination"""

    # True sample: 0.01% AF with 5 UMI families
    true_molecules = 100000  # Total molecules sequenced
    true_mutant_molecules = int(true_molecules * 0.0001)  # 0.01% AF

    # Generate UMI families for true mutants
    true_umi_families = np.random.poisson(3, true_mutant_molecules)  # Average 3 reads per family

    # Contamination: 0.1% AF with different UMI pattern
    contaminant_molecules = int(true_molecules * contamination_rate)
    contaminant_umi_families = np.random.poisson(1, contaminant_molecules)  # Average 1 read per family

    # Analysis
    total_observed_mutants = np.sum(true_umi_families) + np.sum(contaminant_umi_families)
    total_umi_families = len(true_umi_families) + len(contaminant_umi_families)

    # UMI-based AF estimate (more accurate)
    umi_based_af = total_umi_families / true_molecules

    # Read-based AF estimate (contaminated)
    read_based_af = total_observed_mutants / (true_molecules * n_umis_per_molecule)

    return {
        'true_af': 0.0001,
        'umi_based_af': umi_based_af,
        'read_based_af': read_based_af,
        'contamination_detected': umi_based_af < read_based_af * 0.5
    }

# Test UMI effectiveness
umi_results = []
for contam_rate in [0.0001, 0.001, 0.01]:
    result = simulate_umi_contamination_detection(contamination_rate=contam_rate)
    umi_results.append(result)

df_umi = pd.DataFrame(umi_results)
print("UMI-based Contamination Detection:")
print(df_umi.to_string(index=False, float_format='%.6f'))
```

## 5. Using Precise MRD Contamination Commands

Now let's use the actual Precise MRD commands for comprehensive contamination analysis:

### Basic Contamination Evaluation

```bash
# Evaluate contamination robustness across different scenarios
precise-mrd eval-contamination --output reports/contamination_analysis.json

# Generate contamination sensitivity heatmap
precise-mrd eval-contamination --heatmap --output reports/contamination_heatmap.png

# Test specific contamination rates
precise-mrd eval-contamination --rates 0.001 0.01 0.1 --output reports/specific_contamination.json
```

### Index Hopping Analysis

```bash
# Simulate and analyze index hopping effects
precise-mrd eval-contamination --index-hopping --hop-rate 0.001 --output reports/index_hopping.json

# Multi-sample contamination analysis
precise-mrd eval-contamination --multiplex 96 --output reports/multiplex_contamination.json
```

### Mitigation Strategy Testing

```bash
# Test UMI-based contamination detection
precise-mrd eval-contamination --umi-families --min-family-size 3 --output reports/umi_contamination.json

# Evaluate decontamination strategies
precise-mrd eval-contamination --decontamination --strategy consensus --output reports/decontaminated.json
```

## 6. Interpreting Contamination Results

### Key Metrics to Monitor

1. **False Positive Rate**: Percentage of negative samples incorrectly called positive
2. **Sensitivity Loss**: Reduction in true positive detection due to contamination
3. **Quantification Accuracy**: How well we measure true allele frequencies
4. **UMI Family Distribution**: Pattern of read distribution across molecular families

### Acceptable Contamination Levels

- **Research applications**: < 0.1% contamination acceptable
- **Clinical MRD**: < 0.01% contamination required
- **Ultra-sensitive MRD**: < 0.001% contamination target

### Quality Control Thresholds

```python
# Example QC thresholds
QC_THRESHOLDS = {
    'max_contamination_rate': 0.001,  # 0.1%
    'min_umi_family_size': 3,
    'max_index_hopping_rate': 0.0005,  # 0.05%
    'min_detection_efficiency': 0.95   # 95%
}

def validate_sample_quality(contamination_results, qc_thresholds):
    """Validate if sample meets quality criteria"""

    validation_results = {}

    # Check contamination rate
    validation_results['contamination_ok'] = contamination_results['rate'] <= qc_thresholds['max_contamination_rate']

    # Check UMI quality
    validation_results['umi_ok'] = contamination_results['mean_family_size'] >= qc_thresholds['min_umi_family_size']

    # Check detection efficiency
    validation_results['efficiency_ok'] = contamination_results['detection_efficiency'] >= qc_thresholds['min_detection_efficiency']

    # Overall quality score
    validation_results['quality_score'] = sum(validation_results.values()) / len(validation_results)

    return validation_results

# Example validation
sample_results = {
    'rate': 0.0005,
    'mean_family_size': 4.2,
    'detection_efficiency': 0.97
}

validation = validate_sample_quality(sample_results, QC_THRESHOLDS)
print("Sample Quality Validation:")
for metric, result in validation.items():
    print(f"{metric}: {result}")
```

## 7. Best Practices for Contamination Control

### Laboratory Practices

1. **Sample Preparation**
   - Use dedicated workspaces for pre- and post-PCR steps
   - Implement strict cleaning protocols between samples
   - Use UMI-based library preparation methods

2. **Sequencing Considerations**
   - Use unique dual indexing to minimize index hopping
   - Implement proper library pooling strategies
   - Monitor sequencing metrics for contamination indicators

3. **Data Analysis**
   - Always calculate and report contamination estimates
   - Use UMI-based variant calling for better specificity
   - Implement duplicate read removal and proper alignment

### Computational Mitigation

1. **Background Subtraction**
   - Subtract estimated contamination from observed frequencies
   - Use control samples to estimate background rates

2. **Statistical Filtering**
   - Apply stricter significance thresholds for low-frequency variants
   - Use contamination-aware statistical models

3. **Quality-Based Filtering**
   - Filter variants based on UMI family support
   - Remove variants with suspicious read distribution patterns

## Conclusion

Contamination analysis is essential for reliable ctDNA/MRD detection. Key takeaways:

1. **Monitor contamination continuously**: Even small amounts can affect MRD detection
2. **Use UMI-based methods**: They provide better contamination resistance
3. **Set appropriate QC thresholds**: Based on your clinical requirements
4. **Validate your assay**: Regular contamination testing ensures reliable results

### Next Steps

1. **Run contamination analysis**: Use `precise-mrd eval-contamination` on your data
2. **Implement QC workflows**: Set up automated contamination monitoring
3. **Optimize your protocol**: Use these insights to improve your lab processes
4. **Validate with controls**: Always include positive and negative controls

---

This tutorial covered the essential aspects of contamination analysis in ctDNA/MRD. For more advanced topics, explore the [Precise MRD documentation](../index.md) or run the contamination evaluation commands on your own datasets.

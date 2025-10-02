# Assumptions & Limitations

Understanding the assumptions and limitations of Precise MRD is crucial for proper interpretation of results.

## Biological Assumptions

### UMI Family Structure

!!! info "Assumption: UMI Independence"
    **Reads sharing the same UMI originate from the same DNA molecule**
    
    - UMIs uniquely tag individual DNA molecules
    - PCR amplification preserves UMI-molecule relationships
    - Cross-contamination between UMI families is negligible

**Implications:**
- UMI family size reflects PCR amplification efficiency
- Consensus calling assumes errors are independent across reads within a family
- Family size thresholds filter low-confidence molecules

**Validation:**
```python
# Check UMI family size distribution
df = pd.read_parquet("collapse.parquet")
family_sizes = df.groupby("umi").size()
assert family_sizes.min() >= config.umi.min_family_size
```

### Error Model

!!! info "Assumption: Context-Dependent Errors"
    **Sequencing errors vary by trinucleotide context**
    
    - Error rates depend on local sequence context (XYZ → XWZ)
    - Errors are independent across genomic positions
    - Error rates are consistent within experimental batches

**Implications:**
- Error model requires sufficient negative control data
- Context bins with few observations have wide confidence intervals
- Batch effects can confound error rate estimation

**Validation:**
```python
# Check error model coverage
error_df = pd.read_parquet("error_model.parquet")
contexts = error_df["context"].value_counts()
assert contexts.min() >= 10, "Insufficient observations for some contexts"
```

### Contamination Model

!!! info "Assumption: Barcode Swapping"
    **Cross-sample contamination follows index hopping patterns**
    
    - Index hopping occurs during sequencing
    - Contamination rates are symmetric between samples
    - Contamination affects all loci equally

**Implications:**
- Contamination model is simplified (real contamination is more complex)
- Does not model DNA carryover between extractions
- Assumes contamination is detectable through negative controls

## Technical Limitations

### Synthetic Data

!!! warning "Limitation: Simulated Reads"
    **Uses synthetic data, not real FASTQ files**
    
    **Current approach:**
    - Generates synthetic read counts directly
    - Models UMI families with configurable size distributions
    - Simulates errors based on trinucleotide context
    
    **What's missing:**
    - Real sequencing quality scores
    - Complex alignment artifacts
    - Batch-specific systematic errors
    - Amplicon-specific biases

**Mitigation:**
- Synthetic parameters calibrated from real data
- Error models include empirical context effects
- Quality thresholds mimic real QC filters

### Statistical Methods

!!! warning "Limitation: Independence Assumptions"
    **Multiple testing correction assumes independence**
    
    **FDR Control:**
    - Benjamini-Hochberg procedure assumes weak dependence
    - May be conservative with strong linkage disequilibrium
    - Does not account for spatial correlation
    
    **Hypothesis Tests:**
    - Poisson test assumes constant error rate per context
    - Binomial test assumes fixed number of trials
    - Does not model overdispersion

**Mitigation:**
- Use permutation-based FDR for dependent tests
- Empirical p-value calibration validation
- Bootstrap confidence intervals for robust inference

### Performance Constraints

!!! warning "Limitation: Python Implementation"
    **Core algorithms implemented in Python**
    
    **Performance impact:**
    - Slower than optimized C/C++ implementations
    - Memory usage scales with family count
    - Bootstrap iterations are computationally expensive
    
    **Memory scaling:**
    - Small config: ~200MB
    - Default config: ~1GB
    - Large config: ~4GB

**Mitigation:**
- Optional Rust extensions for 2x speedup
- Efficient pandas/numpy operations
- Configurable batch sizes for large datasets

## Statistical Considerations

### Bootstrap Validity

!!! info "Assumption: Bootstrap Conditions"
    **Bootstrap confidence intervals require sufficient replicates**
    
    **Requirements:**
    - `n_bootstrap >= 100` for stable percentile intervals
    - `n_replicates >= 10` for meaningful sampling distribution
    - Approximately normal sampling distribution for symmetric intervals

**Implications:**
- Small configurations may have wide confidence intervals
- Extreme quantiles (LoD95) require more bootstrap samples
- Non-parametric bootstrap assumes exchangeability

### P-value Calibration

!!! info "Assumption: Null Distribution"
    **Test statistics follow expected null distributions**
    
    **Requirements:**
    - Error rates accurately estimated from negative controls
    - Sufficient negative control observations per context
    - No systematic biases in error model

**Validation:**
```python
# Check p-value calibration
from scipy.stats import kstest
pvalues = calls["pvalue"][calls["true_status"] == "negative"]
ks_stat, ks_pvalue = kstest(pvalues, "uniform")
assert ks_pvalue > 0.05, f"P-values not uniform (KS p-value: {ks_pvalue})"
```

### LoD Definition

!!! info "Assumption: Detection Threshold"
    **LoD95 represents 95% detection probability**
    
    **Definition:**
    - LoD95 = allele fraction with 95% detection probability
    - Assumes consistent sensitivity across genomic positions
    - Based on specific statistical test and threshold

**Implications:**
- LoD varies by genomic context and depth
- 95% threshold is somewhat arbitrary (could be 90% or 99%)
- Detection probability depends on assay-specific factors

## Clinical Interpretation

### Sensitivity Limits

!!! warning "Clinical Limitation: Analytical vs Clinical Sensitivity"
    **LoD estimates reflect analytical not clinical performance**
    
    **Analytical LoD:**
    - Based on synthetic variants with known truth
    - Controlled error conditions
    - Idealized UMI family structure
    
    **Clinical performance depends on:**
    - Pre-analytical variables (collection, storage, extraction)
    - Tumor heterogeneity and clonal evolution
    - Circulating tumor DNA shedding patterns
    - Patient-specific factors

### False Discovery Rates

!!! warning "Clinical Limitation: FDR in Practice"
    **FDR control assumes correct error model**
    
    **Laboratory conditions:**
    - Batch effects may violate error model assumptions
    - Contamination rates may exceed model predictions
    - Technical replicates may show unexpected correlation
    
    **Clinical conditions:**
    - Clonal hematopoiesis contributes background mutations
    - Germline variants may confound somatic calling
    - Sample quality affects error rates

## Validation Requirements

### Before Clinical Use

!!! warning "Required Validation"
    **Extensive validation required before clinical application**
    
    **Analytical validation:**
    - Accuracy studies with reference materials
    - Precision studies with technical replicates
    - Linearity across clinically relevant range
    - Interference studies with common substances
    
    **Clinical validation:**
    - Sensitivity/specificity in patient samples
    - Comparison with established methods
    - Clinical cut-off determination
    - Performance across patient populations

### Quality Control

!!! info "Ongoing QC Requirements"
    **Continuous monitoring essential for maintained performance**
    
    **Daily QC:**
    - Negative control contamination rates
    - Positive control recovery
    - Family size distribution metrics
    
    **Batch QC:**
    - Error rate stability across batches
    - LoD consistency over time
    - P-value calibration validation

## Usage Recommendations

### Appropriate Use Cases

✅ **Recommended for:**
- Method development and optimization
- Assay validation studies  
- Algorithm benchmarking
- Educational demonstrations
- Research applications with synthetic data

❌ **Not recommended for:**
- Direct clinical decision making
- Patient diagnosis without validation
- Regulatory submissions without additional validation
- High-stakes research without method validation

### Best Practices

1. **Validation Studies:**
   ```bash
   # Run multiple seeds to assess variability
   for seed in {1..10}; do
     precise-mrd smoke --seed $seed --out results/seed_$seed
   done
   ```

2. **Parameter Sensitivity:**
   ```bash
   # Test key parameter ranges
   for threshold in 0.6 0.7 0.8; do
     precise-mrd init-config --template default \
       | sed "s/consensus_threshold: 0.7/consensus_threshold: $threshold/" \
       > configs/threshold_$threshold.yaml
     precise-mrd smoke --config configs/threshold_$threshold.yaml
   done
   ```

3. **Performance Monitoring:**
   ```bash
   # Regular benchmarking
   precise-mrd benchmark --config production.yaml --n-runs 5
   ```

## Future Improvements

### Planned Enhancements

- **Real FASTQ support**: Direct analysis of sequencing files
- **Advanced error models**: Machine learning-based error prediction
- **Spatial correlation**: Account for genomic position dependencies
- **Batch effect correction**: Robust methods for multi-batch studies

### Research Directions

- **Clinical validation**: Correlation with patient outcomes
- **Tumor biology**: Integration with clonal evolution models
- **Multi-omic integration**: Combine with other liquid biopsy markers
- **Real-time analysis**: Streaming analysis for rapid turnaround

## Getting Help

For questions about assumptions and limitations:

- 📖 Read the [Methods documentation](methods/index.md)
- 💬 [Start a discussion](https://github.com/precise-mrd/precise-mrd-mini/discussions)
- 📧 Contact the development team

Remember: **Always validate assumptions in your specific use case before drawing conclusions.**
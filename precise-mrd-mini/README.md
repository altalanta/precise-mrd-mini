# Precise MRD

[![CI](https://github.com/user/precise-mrd-mini/workflows/CI/badge.svg)](https://github.com/user/precise-mrd-mini/actions)
[![Coverage](https://img.shields.io/badge/coverage-85%25-brightgreen.svg)](coverage.svg)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A ctDNA/UMI toy MRD pipeline with UMI-aware error modeling, LoB/LoD estimation, contamination controls, and comprehensive reporting.

## Features

🧬 **UMI-Aware Error Modeling**
- Consensus UMI calling with family-size thresholds
- Trinucleotide context-specific error rates
- Quality-weighted consensus with configurable parameters

📊 **Statistical Analysis**
- Poisson/binomial hypothesis testing
- Benjamini-Hochberg FDR correction
- P-value calibration and validation

🎯 **LoB/LoD Estimation**
- Bootstrap-based LoD95 estimation
- Detection probability curves across AF×depth grids
- Limit of Blank calculation from negative controls

🔬 **Quality Control**
- Strand bias detection with Fisher's exact test
- End-repair artifact filtering
- Clinical guardrail validation

📈 **Contamination Modeling**
- Cross-sample contamination simulation
- Barcode swap rate estimation
- Contamination impact assessment

📋 **Comprehensive Reporting**
- Interactive HTML reports with embedded plots
- LoD heatmaps and detection curves
- QC metrics and performance summaries

## Quick Start

### Installation

```bash
git clone https://github.com/user/precise-mrd-mini.git
cd precise-mrd-mini
make setup
```

### Basic Usage

```bash
# Run simulation with default parameters
make simulate

# Generate HTML report
make report

# Run quick simulation for CI/testing
make ci-small

# Performance benchmark
make benchmark
```

### Command Line Interface

```bash
# Initialize custom configuration
precise-mrd init-config --output configs/my_analysis.yaml

# Run simulation
precise-mrd simulate --config configs/my_analysis.yaml --run-id my_analysis

# Generate report
precise-mrd report --results results/latest

# Validate results
precise-mrd validate --results results/latest

# Performance benchmark
precise-mrd benchmark
```

## Configuration

### Default Configuration (`configs/default.yaml`)

```yaml
run_id: "default_run"
seed: 42

simulation:
  allele_fractions: [0.05, 0.01, 0.001, 0.0005, 0.0001]
  umi_depths: [5000, 10000, 20000, 50000]
  n_replicates: 1000
  n_bootstrap: 1000

umi:
  min_family_size: 3
  max_family_size: 1000
  quality_threshold: 20
  consensus_threshold: 0.6

stats:
  test_type: "poisson"  # or "binomial"
  alpha: 0.05
  fdr_method: "benjamini_hochberg"

lod:
  detection_threshold: 0.95
  confidence_level: 0.95
```

### Key Parameters

- **allele_fractions**: Range of allele fractions to test (0.01% to 5%)
- **umi_depths**: UMI family depths for simulation grid
- **min_family_size**: Minimum UMI family size for consensus calling
- **test_type**: Statistical test (`poisson` or `binomial`)
- **detection_threshold**: LoD95 detection probability threshold

## Pipeline Overview

### 1. UMI Consensus Calling

```python
from precise_mrd import UMIProcessor

processor = UMIProcessor(min_family_size=3, consensus_threshold=0.6)
families = processor.process_reads(reads)
consensus_data = processor.get_consensus_counts(families)
```

- Groups reads by UMI and genomic position
- Calls consensus with quality weighting
- Filters families by size thresholds
- Handles outlier family sizes

### 2. Error Modeling

```python
from precise_mrd import ErrorModel, ContextAnalyzer

error_model = ErrorModel()
context_analyzer = ContextAnalyzer()

# Estimate context-specific error rates
error_rates = error_model.estimate_background_errors(negative_control_data)
```

- Trinucleotide context normalization
- Background error estimation from negative controls
- Contamination rate modeling

### 3. Statistical Testing

```python
from precise_mrd import StatisticalTester

tester = StatisticalTester(test_type="poisson", alpha=0.05)
results = tester.test_multiple_variants(variant_data, error_rates)
```

- Per-locus hypothesis testing
- Multiple testing correction (FDR)
- Effect size calculation

### 4. LoD/LoB Estimation

```python
from precise_mrd import LODEstimator

estimator = LODEstimator(detection_threshold=0.95)
lod_result = estimator.bootstrap_lod_estimation(simulation_results, depth=10000)
```

- Bootstrap confidence intervals
- Detection probability curves
- Clinical sensitivity metrics

## Assumptions and Limitations

### Biological Assumptions

1. **UMI Families**: Reads sharing UMIs originate from the same DNA molecule
2. **Error Independence**: Sequencing errors are independent across reads
3. **Context Effects**: Error rates vary by trinucleotide context
4. **Contamination Model**: Cross-sample contamination follows barcode swap patterns

### Technical Limitations

1. **Synthetic Data**: Uses simulated reads, not real FASTQ files
2. **Simplified Contamination**: Basic contamination model
3. **Context Dependencies**: Limited trinucleotide contexts implemented
4. **Performance**: Python implementation (Rust extension optional)

### Statistical Considerations

1. **Multiple Testing**: FDR correction assumes independence
2. **Bootstrap Validity**: Requires sufficient replicates for stable estimates
3. **P-value Calibration**: Depends on accurate error model
4. **LoD Definition**: Based on 95% detection probability threshold

## Performance

### Benchmarks (N=1e6 reads)

| Operation | Pure Python | Rust Extension | Speedup |
|-----------|-------------|----------------|---------|
| UMI Grouping | 2.3s | 1.1s | 2.1× |
| Consensus Calling | 1.8s | 0.9s | 2.0× |
| Statistical Testing | 0.5s | 0.5s | 1.0× |

*Benchmarks on MacBook Pro M1, 16GB RAM*

### Memory Usage

- **Small simulation** (CI): ~200MB peak
- **Default simulation**: ~1.2GB peak  
- **Large simulation** (1M families): ~4.8GB peak

## Clinical Guardrails

### Sample Validity Criteria

```python
# Minimum requirements for valid sample
qc_thresholds = {
    'min_total_umi_families': 1000,
    'max_contamination_rate': 0.05,
    'min_depth_per_locus': 100,
    'min_consensus_rate': 0.8
}
```

### Invalid Sample Criteria

- **Insufficient depth**: <1000 total UMI families
- **High contamination**: >5% estimated contamination rate
- **Poor consensus**: <80% consensus calling rate
- **Technical failure**: Multiple QC metrics outside acceptable ranges

## Results Interpretation

### LoD95 Values

- **Excellent** (≤0.01%): Suitable for MRD monitoring
- **Good** (0.01-0.1%): Adequate for most clinical applications  
- **Acceptable** (0.1-1%): Limited sensitivity, use with caution
- **Poor** (>1%): Insufficient for MRD detection

### LoB Interpretation

- **LoB <1%**: Low false positive rate, high specificity
- **LoB 1-5%**: Moderate false positive rate, acceptable for screening
- **LoB >5%**: High false positive rate, requires optimization

### QC Metrics

- **Strand bias p-value <0.01**: Potential technical artifact
- **End-repair enrichment**: Filter G>T/C>A near fragment ends
- **Family size distribution**: Log-normal expected, extreme outliers flagged

## Output Files

### Results Directory Structure

```
results/runs/<run_id>/
├── detection_matrix.csv      # AF×depth detection probabilities
├── statistical_results.csv   # Per-variant test results
├── filter_results.csv        # Quality filter outcomes
├── qc_metrics.json          # Quality control metrics
├── runtime_metrics.json     # Performance measurements
└── lockfile.json           # Reproducibility metadata
```

### HTML Report Sections

1. **Summary**: Key metrics and performance indicators
2. **LoD Analysis**: Detection curves and LoD95 estimates  
3. **LoB Analysis**: False positive rates and thresholds
4. **Context Analysis**: Error rates by trinucleotide context
5. **QC Metrics**: Sample quality and filter performance
6. **Configuration**: Run parameters and reproducibility info

## Development

### Running Tests

```bash
# Unit tests with coverage
make coverage

# Specific test modules
pytest tests/test_umi.py -v
pytest tests/test_stats.py -v
pytest tests/test_lod.py -v

# Performance tests
python -m precise_mrd.profiling
```

### Code Quality

```bash
# Linting and formatting
make lint
make format

# Type checking
mypy src/precise_mrd/

# Security scan
bandit -r src/
```

### Building Documentation

```bash
# Generate API documentation
sphinx-build -b html docs/ docs/_build/

# Update validation report
make report
# -> Updates docs/validation.md automatically
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt
pip install -e .[dev]

# Setup pre-commit hooks
pre-commit install

# Run full test suite
make test
```

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{precise_mrd,
  title = {Precise MRD: ctDNA/UMI MRD Pipeline},
  author = {Precise MRD Team},
  url = {https://github.com/user/precise-mrd-mini},
  version = {0.1.0},
  year = {2024}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

- 📖 [Documentation](https://user.github.io/precise-mrd-mini/)
- 🐛 [Issue Tracker](https://github.com/user/precise-mrd-mini/issues)
- 💬 [Discussions](https://github.com/user/precise-mrd-mini/discussions)

## Acknowledgments

- Inspired by best practices in clinical NGS pipelines
- Statistical methods adapted from cancer genomics literature
- UMI consensus algorithms based on published protocols
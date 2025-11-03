# Precise MRD - Detection Limit Analytics

[![CI](https://github.com/altalanta/precise-mrd-mini/workflows/CI/badge.svg)](https://github.com/altalanta/precise-mrd-mini/actions)
[![Python 3.11](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-mkdocs-blue)](https://altalanta.github.io/precise-mrd-mini/)

A ctDNA/UMI toy MRD pipeline with **formal detection limit analytics**, statistical validation, and comprehensive contamination robustness testing.

## 🎯 Key Features

### 🧬 **Deterministic UMI Processing**
- Modern NumPy RNG API (`np.random.default_rng`)
- Reproducible seed management across all components
- Hash-verified artifact consistency

### 📊 **Formal Detection Limits**
- **LoB (Limit of Blank)**: 95th percentile of blank measurements
- **LoD (Limit of Detection)**: AF yielding 95% detection probability with bias-corrected CIs
- **LoQ (Limit of Quantification)**: Lowest AF meeting precision criteria (CV ≤ 20%)

### 🔬 **Contamination Robustness**
- Index-hopping stress testing with configurable hop rates
- Barcode collision modeling and impact assessment
- Cross-sample contamination sensitivity analysis

### 📈 **Stratified Analysis**
- Power analysis by trinucleotide context and depth
- Calibration assessment across AF/depth strata
- Context-specific error modeling

### 🎯 **Artifact Contract**
- Schema-validated JSON outputs
- Guaranteed artifact paths and structure
- Complete run context metadata

## 🚀 Quick Start

**3-command demo** (deterministic, <5 minutes):

```bash
make setup     # Install dependencies and package
make smoke     # Run fast end-to-end pipeline
make determinism  # Verify identical hashes across runs
```

## 📋 Detection Limit Analytics

### Limit of Blank (LoB)
Simulate pure-blank runs and estimate the 95th percentile of background measurements:

```bash
# Run LoB estimation
precise-mrd eval-lob --n-blank-runs 100 --seed 7
```

**Output**: `reports/lob.json` with background statistics

### Limit of Detection (LoD)
Estimate AF yielding 95% detection across depth grid with bias-corrected confidence intervals:

```bash
# Run LoD estimation across AF range
precise-mrd eval-lod --af-range 1e-4,1e-2 --depths 1000,5000,10000 --seed 7
```

**Outputs**:
- `reports/lod_table.csv` - LoD values per depth with CIs
- `reports/lod_curves.png` - Detection curves visualization

### Limit of Quantification (LoQ)
Find lowest AF meeting precision criteria (CV ≤ 20%):

```bash
# Run LoQ estimation
precise-mrd eval-loq --cv-threshold 0.20 --seed 7
```

**Output**: `reports/loq_table.csv` - LoQ values per depth

## 🧪 Contamination Stress Testing

Test detection robustness under various contamination scenarios:

```bash
# Run contamination stress tests
precise-mrd eval-contamination --hop-rates 0.0,0.001,0.005,0.01 --seed 7
```

**Outputs**:
- `reports/contam_sensitivity.json` - Sensitivity under contamination
- `reports/contam_heatmap.png` - Impact visualization

## 📊 Stratified Analysis

Analyze power and calibration by context and depth:

```bash
# Run stratified analysis
precise-mrd eval-stratified --contexts CpG,CHG,CHH,NpN --seed 7
```

**Outputs**:
- `reports/power_by_stratum.json` - Context-specific power
- `reports/calibration_by_bin.csv` - Binned calibration metrics

## 🔄 Reproducibility

All analyses are **deterministically reproducible**:

```bash
# Verify determinism
make determinism
# Should show identical hashes across runs

# Full statistical validation
make stat-sanity
# Validates Type I error control, FDR, bootstrap coverage
```

## 📈 Performance Metrics

### Expected Detection Limits
- **LoB**: ~0.5-2 false positives per 10K UMIs
- **LoD**: ~1e-3 to 1e-4 AF (depth-dependent)
- **LoQ**: ~5e-3 to 1e-3 AF (20% CV threshold)

### Contamination Tolerance
- **Index hopping**: <1% sensitivity loss at 0.5% hop rate
- **Barcode collisions**: <5% false positive increase at 0.1% collision rate
- **Cross-contamination**: Robust up to 5% cross-sample mixing

## 🎯 Validation Framework

### Statistical Sanity Tests
- **Type I Error Control**: α-level validation in hypothesis testing
- **FDR Monotonicity**: Benjamini-Hochberg correction verification
- **Bootstrap Coverage**: CI coverage on synthetic data
- **LoB/LoD Consistency**: LoB < LoD monotonicity checks

### CI/CD Integration
All tests run in <60s for CI efficiency with fail-closed behavior.

## 📁 Artifact Structure

```
reports/
├── lob.json                    # Limit of Blank results
├── lod_table.csv              # Limit of Detection per depth
├── lod_curves.png             # LoD visualization
├── loq_table.csv              # Limit of Quantification
├── power_by_stratum.json      # Stratified power analysis
├── calibration_by_bin.csv     # Binned calibration metrics
├── contam_sensitivity.json    # Contamination impact
├── contam_heatmap.png         # Contamination visualization
├── metrics.json               # Performance metrics
├── auto_report.html           # Interactive HTML report
├── run_context.json           # Reproducibility metadata
└── hash_manifest.txt          # SHA256 verification
```

## 🔬 Scientific Rigor

### Detection Limit Standards
Following **CLSI EP17** guidelines for detection capability:
- LoB: Highest blank measurement (95th percentile)
- LoD: 95% detection probability with Type I/II error control
- LoQ: Acceptable precision (CV ≤ 20% or absolute error threshold)

### Statistical Methodology
- **Stratified bootstrap**: Bias-corrected confidence intervals
- **Logistic regression**: Detection curve fitting with robust estimation
- **Calibration assessment**: Expected Calibration Error (ECE) and reliability diagrams

## 🛠 Development

### Running Tests
```bash
make test          # All tests
make coverage      # Test coverage
make stat-sanity   # Statistical validation only
make determinism   # Determinism verification
```

### Code Quality
```bash
make lint      # Flake8 + mypy
make format    # Black + isort
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

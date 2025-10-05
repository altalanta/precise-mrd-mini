# Precise MRD

[![CI](https://github.com/altalanta/precise-mrd-mini/workflows/CI/badge.svg)](https://github.com/altalanta/precise-mrd-mini/actions)
[![Documentation](https://github.com/altalanta/precise-mrd-mini/workflows/Documentation/badge.svg)](https://github.com/altalanta/precise-mrd-mini/actions)
[![Docs](https://img.shields.io/badge/docs-mkdocs-blue)](https://altalanta.github.io/precise-mrd-mini/)
[![Python 3.11](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A ctDNA/UMI toy MRD pipeline with **formal detection limit analytics**, deterministic error modeling, statistical validation, and comprehensive CI/CD hardening.

## Features

🧬 **Deterministic UMI Processing**
- Modern NumPy RNG API (`np.random.default_rng`)
- Reproducible seed management across all components
- Hash-verified artifact consistency

📊 **Formal Detection Limits** ⭐
- **LoB (Limit of Blank)**: 95th percentile of blank measurements
- **LoD (Limit of Detection)**: AF yielding 95% detection probability with bias-corrected CIs
- **LoQ (Limit of Quantification)**: Lowest AF meeting precision criteria (CV ≤ 20%)

🔬 **Contamination Robustness** ⭐
- Index-hopping stress testing with configurable hop rates
- Barcode collision modeling and impact assessment
- Cross-sample contamination sensitivity analysis

📈 **Stratified Analysis** ⭐
- Power analysis by trinucleotide context and depth
- Calibration assessment across AF/depth strata
- Context-specific error modeling

📊 **Statistical Rigor**
- Type I error control validation
- Benjamini-Hochberg FDR correction with monotonicity tests
- Bootstrap confidence intervals with coverage validation

🎯 **Artifact Contract**
- Schema-validated JSON outputs
- Guaranteed artifact paths and structure
- Complete run context metadata

🔍 **CI/CD Hardening**
- Determinism verification (hash comparison)
- Statistical sanity tests (< 60s runtime)
- Fail-closed behavior on regressions

📚 **Documentation** ⭐
- Complete MkDocs site with GitHub Pages deployment
- Comprehensive evaluation methodology documentation
- Interactive examples and tutorials

## Quick Start

**3-command demo** (deterministic, <5 minutes):

```bash
make setup     # Install dependencies and package
make smoke     # Run fast end-to-end pipeline  
make determinism  # Verify identical hashes across runs
```

## Determinism & Reproducibility

**Guaranteed deterministic execution**:
- All random operations use seeded `np.random.Generator` instances
- No global random state dependencies
- SHA256-verified artifact consistency
- Complete environment fingerprinting

### Verification Commands

```bash
# Verify determinism (should show identical hashes)
make determinism

# Run statistical validation tests
make stat-sanity

# Generate hash manifest
make smoke
# Creates reports/hash_manifest.txt with SHA256 hashes
```

## Detection Limit Analytics

### New Evaluation Commands ⭐

```bash
# Formal detection limits
make eval-lob          # Limit of Blank estimation  
make eval-lod          # Limit of Detection with CIs
make eval-loq          # Limit of Quantification (CV ≤ 20%)

# Contamination robustness
make eval-contamination # Index hopping, barcode collisions, cross-contamination

# Stratified analysis
make eval-stratified   # Context-specific power and calibration

# Run all evaluations
make eval-all
```

### New Artifacts ⭐

**Detection Limits**:
- `reports/lob.json` - Limit of Blank results (95th percentile of blank measurements)
- `reports/lod_table.csv` - Limit of Detection per depth with confidence intervals
- `reports/lod_curves.png` - Detection curves visualization
- `reports/loq_table.csv` - Limit of Quantification results

**Contamination Analysis**:
- `reports/contam_sensitivity.json` - Impact assessment across contamination scenarios
- `reports/contam_heatmap.png` - Contamination impact heatmap

**Stratified Analysis**:
- `reports/power_by_stratum.json` - Context-specific detection power
- `reports/calibration_by_bin.csv` - Binned calibration metrics

### Expected Performance

| Depth | LoB (calls) | LoD (AF) | LoQ (AF) |
|-------|-------------|----------|----------|
| 1K    | ~2.1        | ~8.5e-3  | ~1.2e-2  |
| 5K    | ~3.8        | ~2.1e-3  | ~3.8e-3  |
| 10K   | ~5.2        | ~1.1e-3  | ~1.9e-3  |

## Artifact Contract

The pipeline **guarantees** these outputs:

**Core Pipeline**:
- `reports/metrics.json` - Performance metrics with bootstrap CIs
- `reports/auto_report.html` - Interactive HTML report
- `reports/run_context.json` - Complete reproducibility metadata
- `reports/hash_manifest.txt` - SHA256 verification manifest

**Detection Limits** (new):
- `reports/lob.json`, `reports/lod_table.csv`, `reports/loq_table.csv`
- `reports/lod_curves.png` - LoD visualization

**Contamination** (new):
- `reports/contam_sensitivity.json`, `reports/contam_heatmap.png`

**Stratified** (new):
- `reports/power_by_stratum.json`, `reports/calibration_by_bin.csv`

All artifacts validate against JSON schemas in `schemas/`.

### Example Run Context

```json
{
  "seed": 7,
  "timestamp": "2024-10-03T14:30:00.000Z",
  "config_hash": "a1b2c3d4e5f6",
  "git_sha": "7fd5373abc...",
  "python_version": "3.11.5",
  "numpy_version": "1.24.4",
  "cli_args": {"command": "smoke", "seed": 7}
}
```

## Pipeline Overview

### 1. Simulation (`simulate_reads`)
- Generates synthetic UMI families with configurable AF/depth grid
- Trinucleotide context-aware error modeling
- Deterministic family size distributions

### 2. UMI Collapse (`collapse_umis`) 
- Quality-weighted consensus calling
- Configurable family size thresholds
- Consensus agreement filtering

### 3. Error Modeling (`fit_error_model`)
- Context-specific background error estimation
- Bootstrap confidence intervals
- Negative control validation

### 4. Statistical Testing (`call_mrd`)
- Poisson/binomial hypothesis testing
- Benjamini-Hochberg FDR correction
- Effect size calculation

### 5. Performance Metrics (`calculate_metrics`)
- ROC/PR AUC with bootstrap CIs
- Calibration analysis
- Detection rate statistics

## Configuration

```yaml
# configs/smoke.yaml
run_id: "smoke_test"
seed: 7

simulation:
  allele_fractions: [0.01, 0.001, 0.0001]
  umi_depths: [1000, 5000]
  n_replicates: 10
  n_bootstrap: 100

umi:
  min_family_size: 3
  consensus_threshold: 0.6

stats:
  test_type: "poisson"
  alpha: 0.05
  fdr_method: "benjamini_hochberg"
```

## Command Line Interface

```bash
# Fast smoke test
precise-mrd smoke --seed 7 --out data/smoke

# Determinism verification
precise-mrd determinism-check --seed 7
```

## Development

### Running Tests

```bash
# All tests
make test

# Test coverage
make coverage

# Statistical validation only
make stat-sanity

# Determinism check
make determinism
```

### Code Quality

```bash
make lint      # Flake8 + mypy
make format    # Black + isort
```

### CI/CD Pipeline

The CI includes **fail-closed** jobs:

1. **Unit Tests** - Full test suite with coverage
2. **Determinism** - Hash comparison across identical runs
3. **Stats Sanity** - Type I error, FDR, bootstrap validation

**Any failure blocks the PR merge.**

## Statistical Validation

The pipeline includes lightweight statistical tests:

- **Type I Error Control**: Validates α-level control in hypothesis testing
- **FDR Monotonicity**: Ensures BH correction is properly implemented  
- **Bootstrap Coverage**: Verifies CI coverage on synthetic data

All tests run in <60s for CI efficiency.

## Repository Safety

⚠️ **Important**: Avoid `git clean -xfd` as it removes untracked work.

Use `make clean-safe` instead to clean only generated artifacts.

## Performance

- **Smoke test**: <5 minutes on standard CI
- **Statistical tests**: <60 seconds  
- **Memory usage**: ~200MB peak for smoke test

## Limitations

1. **Synthetic Data**: Uses simulated reads, not real FASTQ
2. **Simplified Models**: Basic error and contamination models
3. **Limited Contexts**: Subset of trinucleotide contexts implemented

## Citation

```bibtex
@software{precise_mrd,
  title = {Precise MRD: Deterministic ctDNA/UMI Pipeline},
  author = {Precise MRD Team},
  url = {https://github.com/user/precise-mrd-mini},
  version = {0.1.0},
  year = {2024}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
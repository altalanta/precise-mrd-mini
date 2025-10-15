# Getting Started

This guide will help you get up and running with Precise MRD detection limit analytics.

## Installation

### Prerequisites

- Python 3.9 or higher
- Git

### Setup

1. **Clone the repository**:
```bash
git clone https://github.com/altalanta/precise-mrd-mini.git
cd precise-mrd-mini
```

2. **Install the package**:
```bash
make setup
# This runs: python -m pip install -e .[dev]
```

3. **Verify installation**:
```bash
precise-mrd --help
```

## Quick Start

### 3-Command Demo

The fastest way to see Precise MRD in action:

```bash
make setup     # Install dependencies and package
make smoke     # Run fast end-to-end pipeline  
make determinism  # Verify identical hashes across runs
```

This will:
- Generate synthetic ctDNA data
- Run the complete analysis pipeline
- Produce evaluation artifacts
- Verify deterministic reproducibility

### Understanding the Output

After running `make smoke`, check the `reports/` directory:

```bash
ls reports/
# Expected files:
# metrics.json          - Performance metrics with bootstrap CIs
# auto_report.html      - Interactive HTML report  
# run_context.json      - Complete reproducibility metadata
# hash_manifest.txt     - SHA256 verification manifest
```

## Basic Usage

### Command Line Interface

Precise MRD provides several CLI commands:

```bash
# Basic pipeline
precise-mrd smoke --seed 7 --out data/smoke

# Detection limit analysis
precise-mrd eval-lob --n-blank-runs 100 --seed 7
precise-mrd eval-lod --af-range 1e-4,1e-2 --depths 1000,5000,10000 --seed 7  
precise-mrd eval-loq --cv-threshold 0.20 --seed 7

# Contamination testing
precise-mrd eval-contamination --hop-rates 0.0,0.001,0.005,0.01 --seed 7

# Stratified analysis
precise-mrd eval-stratified --contexts CpG,CHG,CHH,NpN --seed 7
```

### Configuration

Analyses are configured via YAML files in the `configs/` directory:

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
  max_family_size: 1000
  quality_threshold: 20
  consensus_threshold: 0.6

stats:
  test_type: "poisson"
  alpha: 0.05
  fdr_method: "benjamini_hochberg"

lod:
  detection_threshold: 0.95
  confidence_level: 0.95
```

## Analysis Workflows

### Detection Limit Analysis

1. **Limit of Blank (LoB)**:
```bash
precise-mrd eval-lob --n-blank-runs 100 --seed 7
# Output: reports/lob.json
```

2. **Limit of Detection (LoD)**:
```bash
precise-mrd eval-lod --af-range 1e-4,1e-2 --depths 1000,5000,10000 --seed 7
# Outputs: reports/lod_table.csv, reports/lod_curves.png
```

3. **Limit of Quantification (LoQ)**:
```bash
precise-mrd eval-loq --cv-threshold 0.20 --seed 7
# Output: reports/loq_table.csv
```

### Contamination Analysis

```bash
precise-mrd eval-contamination \
    --hop-rates 0.0,0.001,0.002,0.005,0.01 \
    --collision-rates 0.0,0.0001,0.0005,0.001 \
    --cross-contam 0.0,0.01,0.05,0.1 \
    --seed 7
# Outputs: reports/contam_sensitivity.json, reports/contam_heatmap.png
```

### Stratified Analysis

```bash
precise-mrd eval-stratified \
    --contexts CpG,CHG,CHH,NpN \
    --af-values 0.001,0.005,0.01,0.05 \
    --depths 1000,5000,10000 \
    --seed 7
# Outputs: reports/power_by_stratum.json, reports/calibration_by_bin.csv
```

## Makefile Targets

Common operations are available as Makefile targets:

```bash
# Setup and testing
make setup          # Install dependencies
make test           # Run unit tests
make coverage       # Test coverage report
make lint           # Code quality checks
make format         # Auto-format code

# Pipeline execution
make smoke          # Fast end-to-end test
make determinism    # Verify reproducibility
make stat-sanity    # Statistical validation tests

# Cleaning
make clean          # Remove build artifacts
make clean-safe     # Clean generated artifacts safely
```

## Python API

You can also use Precise MRD programmatically:

```python
import numpy as np
from precise_mrd.config import load_config
from precise_mrd.eval import estimate_lob, estimate_lod, estimate_loq
from precise_mrd.sim import run_contamination_stress_test

# Load configuration
config = load_config("configs/smoke.yaml")
rng = np.random.default_rng(config.seed)

# Run detection limit analysis
lob_results = estimate_lob(config, rng, n_blank_runs=100)
lod_results = estimate_lod(config, rng, af_range=(1e-4, 1e-2))  
loq_results = estimate_loq(config, rng, cv_threshold=0.20)

# Run contamination testing
contam_results = run_contamination_stress_test(config, rng)

print(f"LoB: {lob_results['lob_value']:.3f}")
print(f"LoD (5K depth): {lod_results['depth_results'][5000]['lod_af']:.2e}")
print(f"LoQ (5K depth): {loq_results['depth_results'][5000]['loq_af_cv']:.2e}")
```

## Reproducibility

All analyses are deterministically reproducible:

### Seed Management
- All random operations use seeded `np.random.Generator` instances
- No global random state dependencies
- Configuration includes deterministic hash

### Verification
```bash
# Run analysis twice and compare
make determinism
# Should show: ✅ Determinism verified - identical hashes
```

### Hash Verification
```bash
# Check artifact hashes
cat reports/hash_manifest.txt
# Example output:
# a1b2c3d4... reports/metrics.json
# e5f6g7h8... reports/auto_report.html
```

## Next Steps

- **[Evaluation](evaluation.md)**: Deep dive into detection limit analytics
- **[Contamination](contamination.md)**: Contamination stress testing details
- **[Statistical Validation](statistical-validation.md)**: Understanding the validation framework
- **[Reproducibility](reproducibility.md)**: Complete reproducibility guidelines

## Troubleshooting

### Common Issues

**ImportError**: Module not found
```bash
# Ensure package is installed in development mode
make setup
```

**Permission errors**:
```bash
# Ensure write permissions for data/ and reports/ directories
chmod -R 755 data/ reports/
```

**Determinism failures**:
```bash
# Check for any non-deterministic dependencies
make stat-sanity
```

**Slow execution**:
```bash
# Use reduced parameters for testing
precise-mrd eval-lod --af-range 1e-3,1e-2 --depths 1000 --n-replicates 10
```

For additional help, see the [API Reference](api-reference.md) or open an issue on GitHub.
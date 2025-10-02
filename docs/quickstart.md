# Quickstart Guide

Get up and running with Precise MRD in under 5 minutes.

## Installation

=== "pip (Recommended)"

    ```bash
    pip install precise-mrd
    ```

=== "Docker"

    ```bash
    # Pull the latest image
    docker pull ghcr.io/precise-mrd/precise-mrd:latest
    
    # Run with volume mount for results
    docker run --rm \
      -v $PWD/reports:/app/reports \
      ghcr.io/precise-mrd/precise-mrd:latest
    ```

=== "Development Install"

    ```bash
    git clone https://github.com/precise-mrd/precise-mrd-mini.git
    cd precise-mrd-mini
    make setup
    ```

## 60-Second Demo

Run the complete MRD pipeline with synthetic data:

```bash
# Fast deterministic end-to-end pipeline
precise-mrd smoke --seed 7
```

This produces:

```
data/smoke/smoke/
├── metrics.json         # ROC AUC, PR AUC, LoD estimates
├── auto_report.html     # Interactive HTML report  
├── run_context.json     # Reproducibility metadata
├── smoke_scores.npy     # Golden test validation
└── *.parquet           # Intermediate results
```

!!! tip "Performance"
    The smoke test runs in **<60 seconds** and validates deterministic execution.

## Step-by-Step Pipeline

For more control, run each stage individually:

### 1. Initialize Configuration

```bash
# Create a custom configuration
precise-mrd init-config --output configs/my_analysis.yaml --template default
```

This creates a YAML configuration with sensible defaults:

```yaml
run_id: "my_analysis"
simulation:
  allele_fractions: [0.02, 0.01, 0.005, 0.002, 0.001]
  umi_depths: [2000, 5000, 10000]
  n_replicates: 12
  n_bootstrap: 200
umi:
  min_family_size: 5
  consensus_threshold: 0.7
stats:
  alpha: 0.05
  test_type: "poisson"
lod:
  detection_threshold: 0.95
```

### 2. Run Simulation

```bash
# Simulate ctDNA reads with UMI families
precise-mrd simulate \
  --config configs/my_analysis.yaml \
  --seed 42 \
  --out artifacts
```

### 3. UMI Consensus Calling

```bash
# Collapse UMI families to consensus sequences
precise-mrd collapse \
  --config configs/my_analysis.yaml \
  --out artifacts
```

### 4. Error Modeling

```bash
# Fit trinucleotide context error model
precise-mrd error-model \
  --config configs/my_analysis.yaml \
  --out artifacts
```

### 5. Variant Calling

```bash
# Statistical calling with LoD estimation
precise-mrd call \
  --config configs/my_analysis.yaml \
  --seed 42 \
  --out artifacts
```

### 6. Generate Report

```bash
# Create interactive HTML report
precise-mrd report \
  --config configs/my_analysis.yaml \
  --out artifacts
```

## Configuration Templates

Choose the right template for your use case:

=== "Small (Testing)"

    ```bash
    precise-mrd init-config --template small
    ```
    
    - **Runtime**: <30 seconds
    - **Replicates**: 10
    - **Bootstrap**: 50
    - **Use case**: CI/CD, development

=== "Default (Research)"

    ```bash
    precise-mrd init-config --template default
    ```
    
    - **Runtime**: ~5 minutes
    - **Replicates**: 12
    - **Bootstrap**: 200
    - **Use case**: Standard analysis

=== "Large (Publication)"

    ```bash
    precise-mrd init-config --template large
    ```
    
    - **Runtime**: ~30 minutes
    - **Replicates**: 1000
    - **Bootstrap**: 1000
    - **Use case**: Rigorous analysis

## Output Files

Each run produces a structured set of outputs:

```
artifacts/my_analysis/
├── config.json              # Final configuration
├── simulate.parquet          # Simulated reads
├── collapse.parquet          # UMI consensus data
├── error_model.parquet       # Context error rates
├── call.parquet             # Variant calls
├── lod_grid.parquet         # Detection probability grid
├── metrics.json             # Summary statistics
├── auto_report.html         # Interactive report
├── lineage.jsonl            # Execution trace
└── run_context.json         # Reproducibility info
```

### Key Metrics

The `metrics.json` contains clinical-grade metrics:

```json
{
  "roc_auc": 0.987,
  "roc_auc_ci": [0.982, 0.991],
  "pr_auc": 0.924,
  "pr_auc_ci": [0.915, 0.932],
  "lod95_estimate": 0.0031,
  "lod95_ci": [0.0028, 0.0035],
  "sensitivity_at_95": 0.93,
  "specificity": 0.994,
  "n_true_positives": 847,
  "n_false_positives": 23,
  "n_families_processed": 156432
}
```

## Validation & QC

### Validate Results

```bash
# Check pipeline outputs and configuration
precise-mrd validate \
  --config configs/my_analysis.yaml \
  --results artifacts/my_analysis
```

### Performance Benchmark

```bash
# Measure pipeline performance
precise-mrd benchmark \
  --config configs/my_analysis.yaml \
  --n-runs 3
```

### Determinism Check

The smoke test includes golden hash validation:

```bash
# Run smoke test and check determinism
precise-mrd smoke --seed 7

# Validate scores match reference (if available)
python -c "
import numpy as np
scores = np.load('data/smoke/smoke/smoke_scores.npy')
print(f'Hash: {hash(scores.tobytes())}')
"
```

## Machine-Readable Output

Add `--json` to any command for machine-readable output:

```bash
precise-mrd smoke --seed 7 --json > results.json
```

```json
{
  "stage": "smoke",
  "output_dir": "data/smoke/smoke",
  "artifacts": {
    "metrics": "data/smoke/smoke/metrics.json",
    "html_report": "data/smoke/smoke/auto_report.html"
  },
  "metrics": {
    "roc_auc": 0.987,
    "lod95_estimate": 0.0031
  }
}
```

## Docker Usage

The Docker image preserves the same interface:

```bash
# Create a configs directory
mkdir -p configs reports

# Generate config
docker run --rm \
  -v $PWD/configs:/app/configs \
  ghcr.io/precise-mrd/precise-mrd:latest \
  precise-mrd init-config --output configs/docker.yaml

# Run analysis
docker run --rm \
  -v $PWD/configs:/app/configs \
  -v $PWD/reports:/app/reports \
  ghcr.io/precise-mrd/precise-mrd:latest \
  precise-mrd smoke --config configs/docker.yaml --out reports
```

## Common Patterns

### Batch Processing

```bash
# Generate multiple configurations
for size in small default large; do
  precise-mrd init-config \
    --template $size \
    --output configs/batch_${size}.yaml
done

# Run batch analysis
for config in configs/batch_*.yaml; do
  run_id=$(basename $config .yaml)
  precise-mrd smoke \
    --config $config \
    --out results/$run_id
done
```

### Parameter Sweeps

```bash
# Test different consensus thresholds
for threshold in 0.6 0.7 0.8; do
  # Modify config programmatically
  python -c "
import yaml
with open('configs/base.yaml') as f:
    config = yaml.safe_load(f)
config['umi']['consensus_threshold'] = $threshold
config['run_id'] = f'threshold_{$threshold}'
with open('configs/threshold_${threshold}.yaml', 'w') as f:
    yaml.dump(config, f)
"
  
  # Run analysis
  precise-mrd smoke --config configs/threshold_${threshold}.yaml
done
```

## Troubleshooting

### Common Issues

!!! warning "Permission Denied"
    ```bash
    # Fix Docker permissions
    sudo chown -R $USER:$USER reports/
    ```

!!! warning "Out of Memory"
    ```bash
    # Use smaller configuration
    precise-mrd init-config --template small
    ```

!!! warning "Slow Performance"
    ```bash
    # Check if you're using the large template
    precise-mrd benchmark --config configs/my_analysis.yaml
    ```

### Getting Help

```bash
# Show version and determinism info
precise-mrd --version
precise-mrd --deterministic

# Command-specific help
precise-mrd simulate --help
precise-mrd validate --help
```

## Next Steps

- **[CLI Guide](cli/index.md)**: Complete command reference
- **[Methods](methods/index.md)**: Statistical algorithms explained
- **[API Reference](reference/)**: Python API documentation
- **[Contributing](contributing.md)**: How to contribute

Ready to dive deeper? Check out the [CLI Guide](cli/index.md) for complete command documentation.
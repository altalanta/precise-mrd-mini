# Reproducibility Guidelines

This page provides comprehensive guidelines for ensuring full reproducibility of detection limit analytics and evaluation results in Precise MRD.

## Reproducibility Principles

Precise MRD is designed with **deterministic reproducibility** as a core principle:

1. **Seeded Randomness**: All random operations use explicit seeds
2. **Artifact Hashing**: SHA256 verification of all outputs
3. **Environment Capture**: Complete computational environment fingerprinting
4. **Version Control**: Git commit tracking for exact code state

## Deterministic Execution

### Random Number Generation

All analyses use the modern NumPy random API with explicit seeding:

```python
# Correct approach - seeded generator
config = load_config("configs/smoke.yaml")
rng = np.random.default_rng(config.seed)  # Deterministic seed

# Generate reproducible random numbers
random_values = rng.normal(0, 1, size=1000)
```

**Avoid**:
```python
# Incorrect - global random state
np.random.seed(42)  # Global state can be modified
values = np.random.normal(0, 1, size=1000)  # Non-deterministic
```

### Seed Management

Seeds are managed hierarchically to ensure independence while maintaining reproducibility:

```python
def run_replicates(base_seed: int, n_replicates: int):
    """Run multiple replicates with independent but reproducible seeds."""
    results = []
    for rep in range(n_replicates):
        # Derive unique seed for each replicate
        rep_seed = base_seed + rep * 1000
        rep_rng = np.random.default_rng(rep_seed)

        result = run_single_replicate(rep_rng)
        results.append(result)

    return results
```

### Configuration Hashing

All configurations are hashed to detect changes:

```python
def config_hash(config: PipelineConfig) -> str:
    """Compute deterministic hash of configuration."""
    config_str = json.dumps(config.to_dict(), sort_keys=True)
    return hashlib.sha256(config_str.encode()).hexdigest()[:16]
```

## Verification Commands

### Basic Determinism Check

```bash
# Verify identical results across runs
precise-mrd determinism --out-dir data/determinism
```

This runs the same analysis twice and compares SHA256 hashes:
```
✅ Determinism verified - identical hashes
Hash comparison:
a1b2c3d4e5f6... reports/metrics.json
e7f8g9h0i1j2... reports/auto_report.html
```

### Hash Manifest Verification

```bash
# Generate hash manifest
precise-mrd smoke --out-dir data/contracts
cat reports/hash_manifest.txt

# Example output:
# a1b2c3d4e5f6g7h8i9j0  reports/metrics.json
# k1l2m3n4o5p6q7r8s9t0  reports/auto_report.html
# u1v2w3x4y5z6a7b8c9d0  reports/run_context.json
```

Programmatic validation is available via:

```python
from pathlib import Path
from precise_mrd.validation import validate_artifacts

validate_artifacts(Path("reports"))
```

### Complete Reproduction

```bash
# Reproduce exact results from configuration
git checkout <commit_sha>
uv sync --extra dev --extra docs
precise-mrd smoke --seed 7 --out-dir data/smoke
sha256sum -c reports/hash_manifest.txt  # Should all pass
```

## Environment Fingerprinting

### Run Context Metadata

Every analysis generates complete run context:

```json
{
  "schema_version": "1.0.0",
  "seed": 7,
  "timestamp": "2024-10-05T14:30:00.000Z",
  "config_hash": "a1b2c3d4e5f6",
  "git_sha": "7fd5373abc123def456",
  "git_branch": "main",
  "git_dirty": false,
  "python_version": "3.11.5",
  "numpy_version": "1.24.4",
  "scipy_version": "1.10.1",
  "pandas_version": "2.0.3",
  "platform": "Linux-5.4.0-x86_64",
  "hostname": "ci-runner-123",
  "user": "runner",
  "cli_args": {
    "command": "smoke",
    "seed": 7,
    "config_run_id": "smoke_test"
  },
  "execution_time_seconds": 127.3
}
```

### Package Versions

Critical package versions are captured automatically:

```python
import precise_mrd
import numpy as np
import scipy
import pandas as pd

version_info = {
    "precise_mrd": precise_mrd.__version__,
    "numpy": np.__version__,
    "scipy": scipy.__version__,
    "pandas": pd.__version__
}
```

## Reproduction Workflows

### Release Reproduction

For versioned releases, use the complete reproduction workflow:

```bash
# 1. Clone at specific release
git clone --branch v1.0.0 https://github.com/altalanta/precise-mrd-mini.git
cd precise-mrd-mini

# 2. Install exact dependencies
make setup

# 3. Reproduce release artifacts
export RELEASE_SEED=42
make smoke SEED=$RELEASE_SEED
make eval-all SEED=$RELEASE_SEED

# 4. Verify hashes against release manifest
sha256sum -c reports/release_hash_manifest.txt
```

### Development Reproduction

For development work, use commit-specific reproduction:

```bash
# 1. Checkout specific commit
git checkout 7fd5373abc123def456

# 2. Reproduce with same configuration
make smoke SEED=7

# 3. Compare against reference
diff reports/metrics.json reference/metrics.json
```

### Cross-Platform Reproduction

Platform differences can affect reproducibility. Use containers for exact reproduction:

```dockerfile
# Dockerfile for reproducible environment
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
RUN pip install -e .

# Fixed environment variables
ENV PYTHONHASHSEED=0
ENV OMP_NUM_THREADS=1
ENV NUMBA_NUM_THREADS=1

CMD ["make", "smoke"]
```

### Docker Reproduction

```bash
# Build reproducible container
docker build -t precise-mrd-repro .

# Run with fixed seed
docker run --rm -v $(pwd)/reports:/app/reports precise-mrd-repro
```

## Debugging Non-Determinism

### Common Sources of Non-Determinism

1. **Global Random State**: Using `np.random` without seeding
2. **Hash Order**: Dictionary iteration order (use `sort_keys=True`)
3. **Floating Point**: Platform-specific precision differences
4. **Threading**: Non-deterministic thread execution order
5. **System Time**: Using timestamps in analysis

### Debugging Tools

```python
def debug_determinism(func, *args, **kwargs):
    """Debug function determinism."""
    # Run function twice
    result1 = func(*args, **kwargs)
    result2 = func(*args, **kwargs)

    # Compare results
    if isinstance(result1, dict):
        for key in result1:
            if not np.allclose(result1[key], result2[key], rtol=1e-10):
                print(f"Non-determinism detected in key: {key}")
                return False

    return True
```

### Detection Techniques

```bash
# Run multiple times and check for differences
for i in {1..5}; do
    make smoke SEED=7
    cp reports/metrics.json reports/run_${i}.json
done

# Compare all runs
for i in {2..5}; do
    diff reports/run_1.json reports/run_${i}.json || echo "Difference in run $i"
done
```

## Best Practices

### For Users

1. **Always Use Seeds**: Specify explicit seeds for all analyses
2. **Document Environment**: Capture package versions and platform info
3. **Verify Hashes**: Check artifact hashes after important runs
4. **Archive Configurations**: Save exact configuration files used

### For Developers

1. **Avoid Global State**: Never use global random state
2. **Test Determinism**: Include determinism tests in CI
3. **Hash Everything**: Generate hashes for all significant outputs
4. **Capture Context**: Always save complete run context metadata

### For CI/CD

1. **Fixed Seeds**: Use consistent seeds across CI runs
2. **Hash Comparison**: Compare hashes to detect regressions
3. **Environment Pinning**: Pin exact package versions
4. **Parallel Safety**: Ensure analyses are thread-safe

## Troubleshooting

### Hash Mismatch

**Symptom**: Different hashes for same input
```bash
❌ Determinism FAILED - hashes differ
Hash comparison:
< a1b2c3d4e5f6... reports/metrics.json
> x9y8z7w6v5u4... reports/metrics.json
```

**Debug Steps**:
1. Check for floating point differences: `diff -u run1.json run2.json`
2. Verify seed usage: `grep -r "random\|seed" src/`
3. Check configuration consistency: `diff config1.yaml config2.yaml`

### Platform Differences

**Symptom**: Different results on different platforms

**Solutions**:
- Use Docker for consistent environment
- Pin NumPy/SciPy to exact versions
- Set environment variables: `PYTHONHASHSEED=0`

### Version Drift

**Symptom**: Results change across package updates

**Solutions**:
- Pin all dependencies to exact versions
- Maintain reference results for regression testing
- Document breaking changes in package updates

## Validation

### Automated Checks

```python
def test_reproducibility():
    """Test that analysis is fully reproducible."""
    config = load_config("configs/test.yaml")

    # Run analysis twice with same seed
    rng1 = np.random.default_rng(42)
    result1 = run_analysis(config, rng1)

    rng2 = np.random.default_rng(42)  # Same seed
    result2 = run_analysis(config, rng2)

    # Results should be identical
    assert np.allclose(result1, result2, rtol=1e-12)
```

### Release Validation

Before each release:

1. **Full Reproduction**: Reproduce all artifacts from scratch
2. **Hash Verification**: Verify all hashes match expected values
3. **Cross-Platform**: Test on multiple platforms/Python versions
4. **Documentation**: Update reproduction instructions

## Archive Structure

For long-term reproducibility, maintain this archive structure:

```
precise-mrd-archive/
├── v1.0.0/
│   ├── source/                    # Complete source code
│   ├── environment/
│   │   ├── requirements.txt       # Exact package versions
│   │   ├── python_version.txt     # Python version used
│   │   └── platform_info.txt      # Platform details
│   ├── configs/                   # Exact configurations used
│   ├── artifacts/                 # All generated artifacts
│   ├── hashes/
│   │   ├── source_hash.txt        # Source code hash
│   │   └── artifact_hashes.txt    # All artifact hashes
│   └── reproduction/
│       ├── reproduce.sh           # Complete reproduction script
│       └── verification.log       # Verification results
```

This comprehensive approach ensures that Precise MRD analyses remain fully reproducible across time, platforms, and computational environments.

## Data Version Control (DVC) Integration ⭐

Precise MRD integrates with **DVC (Data Version Control)** to provide comprehensive experiment tracking, data versioning, and pipeline orchestration.

### DVC Setup

DVC is already initialized in this repository. To start using it:

```bash
# Initialize DVC (already done)
dvc init --no-scm

# Set up data versioning
make dvc-setup

# Or manually
dvc add data
dvc add configs
dvc add reports
```

### DVC Pipeline Stages

The pipeline is organized into DVC stages for systematic execution:

```bash
# Run complete pipeline
dvc repro

# Run specific stages
dvc repro smoke          # Run smoke test
dvc repro eval-lod       # Run LoD calculation
dvc repro docs           # Build documentation

# Check pipeline status
dvc status
dvc diff
```

### Enhanced Makefile Commands

New DVC-enabled commands are available:

```bash
# DVC-enhanced pipeline execution
make smoke-dvc           # Smoke test with DVC tracking
make determinism-dvc     # Determinism check with DVC
make eval-dvc           # All evaluations with DVC
make docs-dvc           # Documentation with DVC

# DVC management
make dvc-status         # Show DVC status
make dvc-push          # Push data to remote storage
make dvc-pull          # Pull data from remote storage
make dvc-experiment name=my_experiment  # Run experiment
```

### Experiment Tracking

DVC enables systematic experiment tracking:

```bash
# Run experiments with different parameters
make dvc-experiment name="high_depth_test"
make dvc-experiment name="contamination_sensitivity"
make dvc-experiment name="umi_optimization"

# Compare experiments
dvc exp show
dvc exp diff main high_depth_test

# List all experiments
dvc exp list
```

### Data Versioning

Key directories are automatically versioned:

- **`data/`**: All input and output data
- **`configs/`**: Configuration files and parameters
- **`reports/`**: Generated reports and artifacts

```bash
# Track changes to data
dvc add data

# See what changed
dvc diff

# Push large data to remote storage
dvc push

# Pull data from remote storage
dvc pull
```

### Pipeline Parameters

Pipeline parameters are managed through `params.yaml`:

```yaml
base:
  seed: 7
  out_dir: data/results

simulation:
  allele_fractions: [0.01, 0.001, 0.0001]
  umi_depths: [1000, 5000]
  n_replicates: 10

evaluation:
  n_blank: 50
  replicates: 25
  target_cv: 0.20

contamination:
  rates: [0.0001, 0.001, 0.01, 0.1]
```

Modify parameters and re-run pipeline:

```bash
# Edit parameters
vim params.yaml

# Reproduce with new parameters
dvc repro eval-lod

# Check what changed
dvc diff
```

### Remote Storage Integration

For large datasets and collaboration, configure remote storage:

```bash
# Add remote storage (S3, GCS, Azure, etc.)
dvc remote add myremote s3://my-bucket/precise-mrd-data

# Configure remote
dvc remote modify myremote access_key_id $AWS_ACCESS_KEY_ID
dvc remote modify myremote secret_access_key $AWS_SECRET_ACCESS_KEY

# Push data to remote
dvc push

# Pull from remote
dvc pull
```

### DVC Workflows

Use the provided workflow script for common operations:

```bash
# Setup DVC (one-time)
python dvc-workflows.py setup

# Run specific workflows
python dvc-workflows.py smoke
python dvc-workflows.py evaluation
python dvc-workflows.py validation
python dvc-workflows.py docs

# Experiment tracking
python dvc-workflows.py experiment --name "parameter_study"

# Data management
python dvc-workflows.py push    # Push to remote
python dvc-workflows.py pull    # Pull from remote
```

### Integration Benefits

DVC integration provides:

1. **Automatic Dependency Tracking**: Pipeline stages know what depends on what
2. **Parameter Management**: Centralized parameter configuration
3. **Data Versioning**: Track changes to large datasets efficiently
4. **Experiment Tracking**: Systematic comparison of different runs
5. **Remote Storage**: Share large data across teams
6. **Reproducible Workflows**: Execute exact same pipeline steps

### Best Practices for DVC

1. **Commit DVC Files**: Always commit `.dvc` files and `dvc.yaml` to Git
2. **Use Remotes**: Configure remote storage for large datasets
3. **Parameterize Everything**: Use `params.yaml` for all configurable values
4. **Document Experiments**: Name experiments descriptively
5. **Validate Pipeline**: Use `dvc repro` to ensure pipeline integrity

### Troubleshooting DVC

**Common Issues:**

```bash
# Pipeline not running
dvc status                    # Check what's outdated
dvc repro --force            # Force re-run

# Missing data
dvc pull                     # Pull from remote
dvc import <remote> <path>   # Import specific data

# Parameter issues
dvc params diff              # See parameter changes
dvc repro <stage>           # Re-run specific stage
```

**Performance Tips:**

- Use `.dvcignore` to exclude unnecessary files
- Configure remote storage for large datasets
- Use `dvc exp run` for parameter sweeps instead of manual loops

This DVC integration enhances Precise MRD's already excellent reproducibility with systematic data versioning and experiment tracking capabilities.

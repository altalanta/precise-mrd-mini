# API Reference

This page provides comprehensive documentation for the Precise MRD Python API, including all modules for detection limit analytics, contamination testing, and statistical validation.

## Core Modules

### Configuration Management

::: precise_mrd.config.PipelineConfig
    options:
      members:
        - to_dict
        - config_hash

::: precise_mrd.config.load_config

### Detection Limit Analytics

#### LoB/LoD/LoQ Analysis

::: precise_mrd.eval.lod.LODAnalyzer
    options:
      members:
        - estimate_lob
        - estimate_lod
        - estimate_loq
        - generate_reports

::: precise_mrd.eval.lod.estimate_lob

::: precise_mrd.eval.lod.estimate_lod

::: precise_mrd.eval.lod.estimate_loq

#### Stratified Analysis

::: precise_mrd.eval.stratified.StratifiedAnalyzer
    options:
      members:
        - analyze_stratified_power
        - analyze_calibration_by_bins
        - generate_stratified_reports

::: precise_mrd.eval.stratified.run_stratified_analysis

### Contamination Testing

::: precise_mrd.sim.contamination.ContaminationSimulator
    options:
      members:
        - simulate_contamination_effects
        - generate_contamination_reports

::: precise_mrd.sim.contamination.run_contamination_stress_test

### Core Pipeline Components

#### Simulation

::: precise_mrd.simulate.simulate_reads

#### UMI Collapse

::: precise_mrd.collapse.collapse_umis

#### Error Modeling

::: precise_mrd.error_model.fit_error_model

#### Statistical Testing

::: precise_mrd.call.call_mrd

#### Performance Metrics

::: precise_mrd.metrics.calculate_metrics

## Usage Examples

### Basic Detection Limit Analysis

```python
import numpy as np
from precise_mrd.config import load_config
from precise_mrd.eval.lod import LODAnalyzer

# Load configuration
config = load_config("configs/smoke.yaml")
rng = np.random.default_rng(config.seed)

# Create analyzer
analyzer = LODAnalyzer(config, rng)

# Estimate Limit of Blank
lob_results = analyzer.estimate_lob(n_blank_runs=100)
print(f"LoB: {lob_results['lob_value']:.3f}")

# Estimate Limit of Detection
lod_results = analyzer.estimate_lod(
    af_range=(1e-4, 1e-2),
    depth_values=[1000, 5000, 10000],
    n_replicates=50
)

# Print LoD for each depth
for depth, results in lod_results['depth_results'].items():
    lod_af = results['lod_af']
    ci_lower = results['lod_ci_lower']
    ci_upper = results['lod_ci_upper']
    print(f"LoD at {depth} depth: {lod_af:.2e} AF [{ci_lower:.2e}, {ci_upper:.2e}]")

# Generate reports
analyzer.generate_reports("reports/")
```

### Contamination Stress Testing

```python
from precise_mrd.sim.contamination import ContaminationSimulator

# Create contamination simulator
simulator = ContaminationSimulator(config, rng)

# Run contamination effects simulation
results = simulator.simulate_contamination_effects(
    hop_rates=[0.0, 0.001, 0.005, 0.01],
    barcode_collision_rates=[0.0, 0.0001, 0.001],
    cross_sample_proportions=[0.0, 0.01, 0.05, 0.1],
    af_test_values=[0.001, 0.005, 0.01],
    depth_values=[1000, 5000],
    n_replicates=20
)

# Access results by contamination type
hop_results = results['index_hopping']
for hop_rate, af_data in hop_results.items():
    for af, depth_data in af_data.items():
        for depth, metrics in depth_data.items():
            sensitivity = metrics['mean_sensitivity']
            print(f"Hop rate {hop_rate}, AF {af}, depth {depth}: {sensitivity:.3f}")

# Generate reports
simulator.generate_contamination_reports("reports/")
```

### Stratified Power Analysis

```python
from precise_mrd.eval.stratified import StratifiedAnalyzer

# Create stratified analyzer
analyzer = StratifiedAnalyzer(config, rng)

# Run stratified power analysis
power_results = analyzer.analyze_stratified_power(
    af_values=[0.001, 0.005, 0.01, 0.05],
    depth_values=[1000, 5000, 10000],
    contexts=['CpG', 'CHG', 'CHH', 'NpN'],
    n_replicates=50
)

# Access power results by context
for context, depth_data in power_results['stratified_results'].items():
    print(f"\nContext: {context}")
    for depth, af_data in depth_data.items():
        for af, results in af_data.items():
            power = results['mean_detection_rate']
            std = results['std_detection_rate']
            print(f"  Depth {depth}, AF {af:.0e}: {power:.3f} ± {std:.3f}")

# Run calibration analysis
calib_results = analyzer.analyze_calibration_by_bins(
    af_values=[0.001, 0.005, 0.01, 0.05],
    depth_values=[1000, 5000, 10000],
    n_bins=10,
    n_replicates=100
)

# Print calibration summary
for data_point in calib_results['calibration_data']:
    depth = data_point['depth']
    af = data_point['af']
    ece = data_point['ece']
    print(f"Depth {depth}, AF {af:.0e}: ECE = {ece:.3f}")
```

### Complete Pipeline Execution

```python
from precise_mrd.simulate import simulate_reads
from precise_mrd.collapse import collapse_umis
from precise_mrd.error_model import fit_error_model
from precise_mrd.call import call_mrd
from precise_mrd.metrics import calculate_metrics

# Run complete pipeline
def run_complete_pipeline(config, rng):
    """Run the complete Precise MRD pipeline."""
    
    # 1. Simulate synthetic reads
    print("Simulating reads...")
    reads_df = simulate_reads(config, rng)
    print(f"Generated {len(reads_df)} read families")
    
    # 2. Collapse UMI families
    print("Collapsing UMI families...")
    collapsed_df = collapse_umis(reads_df, config, rng)
    print(f"Collapsed to {len(collapsed_df)} consensus reads")
    
    # 3. Fit error model
    print("Fitting error model...")
    error_model = fit_error_model(collapsed_df, config, rng)
    print(f"Error model fitted with {len(error_model)} parameters")
    
    # 4. Call variants
    print("Calling variants...")
    calls_df = call_mrd(collapsed_df, error_model, config, rng)
    n_variants = len(calls_df[calls_df['variant_call'] == True])
    print(f"Called {n_variants} variants from {len(calls_df)} tests")
    
    # 5. Calculate performance metrics
    print("Calculating metrics...")
    metrics = calculate_metrics(calls_df, config, rng)
    print(f"Calculated {len(metrics)} performance metrics")
    
    return {
        'reads': reads_df,
        'collapsed': collapsed_df,
        'error_model': error_model,
        'calls': calls_df,
        'metrics': metrics
    }

# Execute pipeline
results = run_complete_pipeline(config, rng)
```

### Convenience Functions

For quick analyses, use the convenience functions:

```python
from precise_mrd.eval.lod import estimate_lob, estimate_lod, estimate_loq
from precise_mrd.eval.stratified import run_stratified_analysis
from precise_mrd.sim.contamination import run_contamination_stress_test

# Quick detection limit estimation
lob_results = estimate_lob(config, rng, n_blank_runs=50)
lod_results = estimate_lod(config, rng, af_range=(1e-4, 1e-2))
loq_results = estimate_loq(config, rng, cv_threshold=0.20)

# Quick stratified analysis
power_results, calib_results = run_stratified_analysis(config, rng)

# Quick contamination testing
contam_results = run_contamination_stress_test(config, rng)
```

## Configuration Objects

### PipelineConfig

Main configuration object containing all analysis parameters:

```python
@dataclass
class PipelineConfig:
    run_id: str
    seed: int
    simulation: SimulationConfig
    umi: UMIConfig
    stats: StatsConfig
    lod: LODConfig
```

### SimulationConfig

Parameters for synthetic data generation:

```python
@dataclass
class SimulationConfig:
    allele_fractions: List[float]
    umi_depths: List[int]
    n_replicates: int
    n_bootstrap: int
```

### UMIConfig

UMI processing parameters:

```python
@dataclass
class UMIConfig:
    min_family_size: int
    max_family_size: int
    quality_threshold: int
    consensus_threshold: float
```

### StatsConfig

Statistical testing parameters:

```python
@dataclass
class StatsConfig:
    test_type: str
    alpha: float
    fdr_method: str
```

### LODConfig

Detection limit estimation parameters:

```python
@dataclass
class LODConfig:
    detection_threshold: float
    confidence_level: float
```

## Error Handling

The API uses custom exceptions for specific error conditions:

```python
from precise_mrd.config import ConfigurationError
from precise_mrd.eval.lod import DetectionLimitError

try:
    config = load_config("invalid_config.yaml")
except ConfigurationError as e:
    print(f"Configuration error: {e}")

try:
    lod_results = estimate_lod(config, rng, af_range=(1e-2, 1e-4))  # Invalid range
except DetectionLimitError as e:
    print(f"Detection limit error: {e}")
```

## Logging

The API includes structured logging for debugging and monitoring:

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('precise_mrd')

# Run analysis with detailed logging
analyzer = LODAnalyzer(config, rng)
lob_results = analyzer.estimate_lob(n_blank_runs=100)
```

## Thread Safety

All analysis functions are thread-safe when using independent random number generators:

```python
import concurrent.futures

def run_parallel_analysis(configs_and_seeds):
    """Run multiple analyses in parallel."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        
        for config, seed in configs_and_seeds:
            rng = np.random.default_rng(seed)  # Independent RNG per thread
            future = executor.submit(estimate_lod, config, rng)
            futures.append(future)
        
        results = [future.result() for future in futures]
    
    return results
```

## Performance Considerations

### Memory Usage

- **LoB estimation**: ~10MB for 100 blank runs
- **LoD estimation**: ~50MB for full AF grid (15 points × 3 depths × 50 reps)
- **Contamination testing**: ~100MB for complete stress test
- **Stratified analysis**: ~80MB for 4 contexts × 3 depths × 4 AFs

### Runtime Estimates

On a standard CPU (Intel i5):

- **Quick LoB** (20 runs): ~5 seconds
- **Quick LoD** (reduced grid): ~15 seconds  
- **Full LoD** (complete grid): ~2 minutes
- **Contamination stress test**: ~3 minutes
- **Stratified analysis**: ~90 seconds

### Optimization Tips

1. **Reduce Replicates**: For development, use `n_replicates=10-15`
2. **Limit AF Range**: Test with narrower AF ranges first
3. **Parallel Execution**: Use multiple processes for independent analyses
4. **Caching**: Cache intermediate results for repeated analyses

## Version Compatibility

The API maintains backward compatibility within major versions:

- **v1.x**: Stable API, backward compatible updates
- **Breaking changes**: Only in major version updates (v2.0, etc.)
- **Deprecation policy**: 2 minor versions warning before removal

Check version compatibility:

```python
import precise_mrd
print(f"Precise MRD version: {precise_mrd.__version__}")

# Check API compatibility
if precise_mrd.__version__.startswith('1.'):
    print("Compatible with v1.x API")
```
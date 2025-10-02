# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2024-10-02

### Added

#### Package & Infrastructure
- Initial production-ready package structure with proper `pyproject.toml`
- Python 3.10-3.12 support with full type annotations
- Comprehensive CI/CD pipeline with GitHub Actions
- PyPI and GitHub Container Registry publishing
- Pre-commit hooks with ruff, black, mypy, and bandit

#### Core Functionality
- **UMI-aware ctDNA simulation**: Realistic read simulation with configurable error models
- **Deterministic error modeling**: Trinucleotide context-specific error rates
- **UMI consensus calling**: Family-size thresholds with quality weighting
- **Statistical testing**: Poisson/binomial hypothesis tests with Benjamini-Hochberg FDR
- **LoB/LoD estimation**: Bootstrap-based detection limit calculation
- **Contamination modeling**: Cross-sample contamination simulation

#### CLI Interface
- **Typer-based CLI** with consistent flag naming across subcommands
- `precise-mrd simulate` - Run ctDNA simulation with UMI error modeling
- `precise-mrd collapse` - UMI consensus calling and family collapsing  
- `precise-mrd error-model` - Fit trinucleotide context error models
- `precise-mrd call` - Statistical variant calling with LoD estimation
- `precise-mrd report` - Generate interactive HTML reports
- `precise-mrd smoke` - Fast deterministic end-to-end pipeline (<60s)
- Machine-readable JSON output with `--json` flag
- Deterministic execution with `--seed` parameter

#### Quality Assurance
- **Golden tests**: SHA256 validation of first 10 variant scores for determinism
- **Comprehensive test suite**: 90%+ coverage with pytest
- **Property-based testing**: Hypothesis.py for robust statistical validation
- **Performance benchmarks**: Runtime budget enforcement (<60s smoke test)
- **Security scanning**: Bandit integration in CI

#### Documentation
- **Interactive documentation**: MkDocs with Material theme
- **API reference**: Auto-generated from docstrings with mkdocstrings
- **Quickstart guide**: Copy-paste examples for immediate use
- **CLI guide**: Complete subcommand documentation with examples
- **Mathematical methods**: LoD/LoB algorithms and statistical assumptions

#### Determinism & Reproducibility
- Centralized `set_all_seeds()` for Python, NumPy, and PyTorch determinism
- Git SHA tracking in run context for full reproducibility
- `reports/run_context.json` with complete execution metadata
- Deterministic defaults matching README examples

#### Docker Support
- CPU-only Docker image with optimized build layers
- GitHub Container Registry publishing with semantic versioning
- Preserved 3-command demo (`make setup && make smoke`)

### Technical Details

#### Statistical Methods
- Poisson test for low-frequency variants (AF < 1%)
- Binomial test for higher-frequency variants
- Bootstrap confidence intervals for LoD95 estimation
- Fisher's exact test for strand bias detection

#### Performance Optimizations
- Vectorized NumPy operations for statistical calculations
- Optional Rust extensions for UMI processing (2x speedup)
- Efficient pandas operations for large variant tables
- Memory-mapped file I/O for large datasets

#### Quality Control Features
- End-repair artifact filtering (G>T/C>A at fragment ends)
- Family size distribution validation (log-normal expected)
- Contamination rate estimation and clinical guardrails
- P-value calibration validation

### Dependencies
- **Core**: numpy, pandas, scipy, statsmodels, scikit-learn
- **CLI**: typer, tqdm, pyyaml
- **Plotting**: matplotlib, seaborn
- **Development**: pytest, mypy, ruff, pre-commit
- **Documentation**: mkdocs, mkdocs-material

### Backwards Compatibility
- N/A (initial release)

### Security
- Bandit security scanning in CI
- Pinned dependency versions
- No hardcoded secrets or credentials
- Secure OIDC publishing to PyPI

[0.1.0]: https://github.com/precise-mrd/precise-mrd-mini/releases/tag/v0.1.0
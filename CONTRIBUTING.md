# Contributing to Precise MRD

We welcome contributions to Precise MRD! This document provides guidelines for contributing to the project.

## Quick Start

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/yourusername/precise-mrd-mini.git
   cd precise-mrd-mini
   ```
3. **Set up development environment**:
   ```bash
   make setup
   ```
4. **Run tests** to ensure everything works:
   ```bash
   make test
   ```

## Development Workflow

### Setting Up Your Environment

```bash
# Install dependencies and package in editable mode
make setup

# Verify installation with smoke test
make smoke

# Run full test suite
make test

# Check code quality
make lint
```

### Making Changes

1. **Create a feature branch**:
   ```bash
   git checkout -b feature/my-awesome-feature
   ```

2. **Make your changes** following our coding standards (see below)

3. **Add tests** for new functionality:
   ```bash
   # Add tests to tests/test_*.py
   pytest tests/test_my_feature.py -v
   ```

4. **Run the full test suite**:
   ```bash
   make test
   make coverage  # Ensure >90% coverage
   ```

5. **Check code quality**:
   ```bash
   make lint
   make format
   ```

6. **Update documentation** if needed (see Documentation section)

### Submitting Changes

1. **Commit your changes** with a descriptive message:
   ```bash
   git add .
   git commit -m "feat: add new UMI consensus algorithm"
   ```

2. **Push to your fork**:
   ```bash
   git push origin feature/my-awesome-feature
   ```

3. **Open a Pull Request** on GitHub with:
   - Clear description of changes
   - Reference to any related issues
   - Test results and coverage information

## Coding Standards

### Python Style

We follow PEP 8 with some modifications. Code formatting is handled automatically:

```bash
# Format code
make format

# Check formatting and style
make lint
```

**Key guidelines:**
- Line length: 100 characters
- Use type hints for all functions
- Docstrings for all public functions (Google style)
- Import sorting with isort (black-compatible profile)

### Code Organization

```
src/precise_mrd/
├── __init__.py          # Public API exports
├── cli.py               # Command-line interface
├── config.py            # Configuration handling
├── simulate.py          # Core simulation logic
├── collapse.py          # UMI consensus calling
├── call.py              # Variant calling
├── error_model.py       # Error modeling
├── lod.py               # LoD/LoB estimation
├── reporting.py         # Report generation
├── utils.py             # Utility functions
├── determinism_utils/   # Reproducibility tools
├── schemas/             # Configuration schemas
└── templates/           # Report templates
```

### Type Hints

All functions must have complete type annotations:

```python
from __future__ import annotations

def process_umis(
    reads: pd.DataFrame, 
    min_family_size: int = 3,
    consensus_threshold: float = 0.6
) -> pd.DataFrame:
    """Process UMI families into consensus sequences.
    
    Args:
        reads: Input read data
        min_family_size: Minimum UMI family size
        consensus_threshold: Consensus calling threshold
        
    Returns:
        Consensus sequence data
    """
```

### Docstring Format

Use Google-style docstrings:

```python
def calculate_lod(
    detection_probs: np.ndarray,
    allele_fractions: np.ndarray,
    threshold: float = 0.95
) -> dict[str, float]:
    """Calculate LoD95 from detection probability curves.
    
    This function estimates the limit of detection at 95% detection
    probability using bootstrap confidence intervals.
    
    Args:
        detection_probs: Detection probabilities for each AF/depth combination
        allele_fractions: Tested allele fraction values
        threshold: Detection probability threshold (default: 0.95)
        
    Returns:
        Dictionary containing:
            - lod95: LoD95 estimate
            - ci_lower: 95% CI lower bound  
            - ci_upper: 95% CI upper bound
            
    Raises:
        ValueError: If detection_probs and allele_fractions have different lengths
        
    Example:
        >>> probs = np.array([0.1, 0.5, 0.9, 0.99])
        >>> afs = np.array([0.001, 0.005, 0.01, 0.05])
        >>> lod = calculate_lod(probs, afs)
        >>> print(f"LoD95: {lod['lod95']:.4f}")
    """
```

## Testing Guidelines

### Test Structure

```
tests/
├── conftest.py              # Shared fixtures
├── test_smoke_contract.py   # Smoke test validation
├── test_determinism.py      # Reproducibility tests
├── test_umi.py              # UMI processing tests
├── test_stats.py            # Statistical method tests
├── test_lod.py              # LoD/LoB calculation tests
├── test_config_validator.py # Configuration validation
├── test_io_schemas.py       # I/O and schema tests
├── test_numerical_stability.py # Numerical edge cases
└── test_exceptions.py       # Error handling tests
```

### Writing Tests

```python
import pytest
import numpy as np
from precise_mrd import simulate_reads, PipelineConfig

def test_simulation_determinism():
    """Test that simulation produces identical results with same seed."""
    config = PipelineConfig.default()
    
    # Run simulation twice with same seed
    result1 = simulate_reads(config, seed=42)
    result2 = simulate_reads(config, seed=42)
    
    # Results should be identical
    assert result1.reads.equals(result2.reads)
    
def test_umi_family_size_thresholds():
    """Test UMI family size filtering."""
    # Create test data
    reads = create_test_reads(n_families=100)
    
    # Test different thresholds
    for min_size in [1, 3, 5, 10]:
        collapsed = collapse_umis(reads, min_family_size=min_size)
        assert collapsed["family_size"].min() >= min_size
```

### Test Categories

1. **Unit tests**: Test individual functions
2. **Integration tests**: Test component interactions  
3. **Property-based tests**: Use Hypothesis for edge cases
4. **Golden tests**: Validate deterministic outputs
5. **Performance tests**: Ensure runtime budgets

### Running Tests

```bash
# Run all tests
make test

# Run specific test file
pytest tests/test_umi.py -v

# Run with coverage
make coverage

# Run performance benchmarks
pytest tests/test_performance.py --benchmark-only
```

## Documentation

### Building Documentation

```bash
# Install documentation dependencies
pip install -e .[docs]

# Build documentation locally
mkdocs serve

# Documentation will be available at http://localhost:8000
```

### Documentation Structure

```
docs/
├── index.md                # Homepage
├── quickstart.md           # Getting started guide
├── cli/                    # CLI documentation
│   ├── overview.md
│   ├── simulate.md
│   ├── call.md
│   └── report.md
├── api/                    # API reference (auto-generated)
├── methods/                # Statistical methods
│   ├── umi-consensus.md
│   ├── error-modeling.md
│   ├── statistical-tests.md
│   └── lod-estimation.md
├── assumptions.md          # Model assumptions and limitations
└── contributing.md         # This file
```

### Writing Documentation

- Use clear, concise language
- Include code examples for all APIs
- Add mathematical notation where appropriate
- Link related concepts
- Update docstrings when changing code

## Issue Reporting

### Bug Reports

Use the bug report template and include:

- Precise MRD version (`precise-mrd --version`)
- Python version and platform
- Minimal reproducing example
- Expected vs actual behavior
- Full error traceback

### Feature Requests

Use the feature request template and include:

- Clear use case description
- Proposed API (if applicable)
- Alternative approaches considered
- Backwards compatibility considerations

## Release Process

### Version Numbering

We follow [Semantic Versioning](https://semver.org/):

- **MAJOR**: Incompatible API changes
- **MINOR**: New functionality (backwards compatible)
- **PATCH**: Bug fixes (backwards compatible)

### Release Checklist

1. **Update version** in `pyproject.toml` and `src/precise_mrd/__init__.py`
2. **Update CHANGELOG.md** with release notes
3. **Run full test suite**: `make test`
4. **Check documentation**: `mkdocs serve`
5. **Tag release**: `git tag v0.1.0`
6. **Push tag**: `git push origin v0.1.0`
7. **CI/CD handles** PyPI and GitHub Container Registry publishing

## Code of Conduct

### Our Standards

- Use welcoming and inclusive language
- Be respectful of differing viewpoints
- Accept constructive criticism gracefully
- Focus on what's best for the community
- Show empathy towards other contributors

### Enforcement

Unacceptable behavior may be reported to the project maintainers. All complaints will be reviewed and investigated fairly.

## Development Tools

### Pre-commit Hooks

```bash
# Install pre-commit hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

Hooks include:
- `ruff`: Fast Python linter
- `black`: Code formatting
- `mypy`: Type checking
- `bandit`: Security scanning

### Continuous Integration

Our CI pipeline runs on GitHub Actions:

- **Tests**: Python 3.10, 3.11, 3.12 on Ubuntu
- **Linting**: ruff, mypy, bandit
- **Coverage**: Minimum 90% required
- **Documentation**: Build and deploy to GitHub Pages
- **Publishing**: PyPI and GHCR on tagged releases

### Performance Monitoring

```bash
# Profile smoke test
python -m cProfile -o smoke.prof -m precise_mrd.cli smoke

# Analyze with snakeviz
pip install snakeviz
snakeviz smoke.prof
```

## Questions?

- 📖 Check the [documentation](https://precise-mrd.github.io/precise-mrd-mini/)
- 🐛 Open an [issue](https://github.com/precise-mrd/precise-mrd-mini/issues)
- 💬 Start a [discussion](https://github.com/precise-mrd/precise-mrd-mini/discussions)

Thank you for contributing to Precise MRD! 🧬
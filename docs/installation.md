# Installation

Multiple installation options are available to suit different use cases.

## PyPI Installation (Recommended)

Install from PyPI for the latest stable release:

```bash
pip install precise-mrd
```

### With Optional Dependencies

```bash
# Development tools
pip install precise-mrd[dev]

# Documentation tools  
pip install precise-mrd[docs]

# Rust extensions (2x performance boost)
pip install precise-mrd[rust]

# All optional dependencies
pip install precise-mrd[dev,docs,rust]
```

## Docker Installation

Pre-built Docker images are available from GitHub Container Registry:

```bash
# Pull latest stable release
docker pull ghcr.io/precise-mrd/precise-mrd:latest

# Pull specific version
docker pull ghcr.io/precise-mrd/precise-mrd:v0.1.0
```

### Basic Usage

```bash
# Run with volume mounts
docker run --rm \
  -v $PWD/configs:/app/configs \
  -v $PWD/reports:/app/reports \
  ghcr.io/precise-mrd/precise-mrd:latest \
  precise-mrd smoke --out reports
```

### Docker Compose

Create a `docker-compose.yml`:

```yaml
version: '3.8'
services:
  precise-mrd:
    image: ghcr.io/precise-mrd/precise-mrd:latest
    volumes:
      - ./configs:/app/configs
      - ./reports:/app/reports
      - ./data:/app/data
    command: precise-mrd smoke --out reports
```

Run with:

```bash
docker-compose up
```

## Development Installation

For development and contributing:

```bash
# Clone repository
git clone https://github.com/precise-mrd/precise-mrd-mini.git
cd precise-mrd-mini

# Set up development environment
make setup

# Verify installation
make smoke
```

### Development Requirements

- **Python**: 3.10, 3.11, or 3.12
- **Git**: For version control and reproducibility tracking
- **Make**: For build automation (optional but recommended)

### Manual Development Setup

If `make` is not available:

```bash
# Install dependencies
pip install -r requirements.txt

# Install in editable mode
pip install -e .

# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/
```

## Conda Installation

Create a conda environment with precise-mrd:

```bash
# Create environment from file
conda env create -f environment.yml

# Or create manually
conda create -n precise-mrd python=3.11
conda activate precise-mrd
pip install precise-mrd
```

The `environment.yml` file:

```yaml
name: precise-mrd
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.11
  - numpy>=1.24.0
  - pandas>=2.0.0
  - scipy>=1.10.0
  - matplotlib>=3.7.0
  - pip
  - pip:
    - precise-mrd
```

## Verification

Verify your installation:

```bash
# Check version
precise-mrd --version

# Check determinism configuration
precise-mrd --deterministic

# Run smoke test (should complete in <60s)
precise-mrd smoke --seed 7

# Check outputs
ls data/smoke/smoke/
```

Expected output:

```json
{
  "version": "0.1.0",
  "git_sha": "abc123...",
  "python_version": "3.11.x",
  "platform": "...",
  "deterministic_flags": {
    "PYTHONHASHSEED": "0 (recommended)",
    "numpy_random_state": "controlled via --seed",
    "pytorch_deterministic": "set if PyTorch available"
  }
}
```

## System Requirements

### Minimum Requirements

- **CPU**: 2 cores
- **Memory**: 2GB RAM
- **Storage**: 500MB free space
- **OS**: Linux, macOS, or Windows

### Recommended Requirements

- **CPU**: 4+ cores
- **Memory**: 8GB+ RAM (for large configurations)
- **Storage**: 2GB+ free space
- **OS**: Linux or macOS (for best performance)

### Performance Notes

- **Small config**: ~200MB memory, <30s runtime
- **Default config**: ~1GB memory, ~5min runtime  
- **Large config**: ~4GB memory, ~30min runtime

## Platform-Specific Notes

### macOS

```bash
# Install via Homebrew (if available)
brew install precise-mrd

# Or use pip
pip install precise-mrd
```

### Windows

Use pip or conda installation. For best performance, consider using WSL2:

```bash
# In WSL2
pip install precise-mrd
```

### Linux

Most distributions work out-of-the-box:

```bash
# Ubuntu/Debian
sudo apt update
pip install precise-mrd

# CentOS/RHEL
sudo yum update
pip install precise-mrd
```

## Performance Optimization

### Rust Extensions

For 2x performance improvement in UMI processing:

```bash
# Install Rust toolchain
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source ~/.cargo/env

# Install with Rust extensions
pip install precise-mrd[rust]

# Verify Rust extensions
python -c "
import precise_mrd.rust_ext
print('Rust extensions available')
"
```

### Multiprocessing

Use multiple cores for large analyses:

```bash
# Set number of threads (default: 1)
precise-mrd simulate --threads 4 --config configs/large.yaml
```

### Memory Management

For large configurations:

```bash
# Monitor memory usage
precise-mrd benchmark --config configs/large.yaml

# Use smaller batch sizes if needed
python -c "
import yaml
with open('configs/large.yaml') as f:
    config = yaml.safe_load(f)
config['simulation']['n_replicates'] = 100  # Reduce from 1000
with open('configs/large_reduced.yaml', 'w') as f:
    yaml.dump(config, f)
"
```

## Troubleshooting

### Common Installation Issues

!!! warning "ImportError: No module named 'precise_mrd'"
    ```bash
    # Check installation
    pip list | grep precise-mrd
    
    # Reinstall if necessary
    pip uninstall precise-mrd
    pip install precise-mrd
    ```

!!! warning "Permission denied"
    ```bash
    # Use user install
    pip install --user precise-mrd
    
    # Or virtual environment
    python -m venv venv
    source venv/bin/activate  # Linux/macOS
    # venv\Scripts\activate   # Windows
    pip install precise-mrd
    ```

!!! warning "Rust compilation failed"
    ```bash
    # Install without Rust extensions
    pip install precise-mrd
    
    # Or install Rust toolchain first
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
    ```

### Version Conflicts

If you encounter dependency conflicts:

```bash
# Create fresh environment
python -m venv fresh_env
source fresh_env/bin/activate
pip install precise-mrd

# Or use pipx for isolated installation
pipx install precise-mrd
```

### Performance Issues

```bash
# Check system resources
precise-mrd benchmark --n-runs 1

# Use smaller configuration
precise-mrd init-config --template small

# Monitor resource usage
top -p $(pgrep -f precise-mrd)
```

## Updating

### Update from PyPI

```bash
pip install --upgrade precise-mrd
```

### Update Docker Image

```bash
docker pull ghcr.io/precise-mrd/precise-mrd:latest
```

### Development Updates

```bash
git pull origin main
pip install -e .
```

## Uninstallation

### Remove pip installation

```bash
pip uninstall precise-mrd
```

### Remove Docker images

```bash
docker rmi ghcr.io/precise-mrd/precise-mrd:latest
```

### Remove development installation

```bash
pip uninstall precise-mrd
rm -rf precise-mrd-mini/  # Remove cloned repository
```

## Support

For installation issues:

- 📖 Check the [troubleshooting section](#troubleshooting)
- 🐛 [Open an issue](https://github.com/precise-mrd/precise-mrd-mini/issues)
- 💬 [Start a discussion](https://github.com/precise-mrd/precise-mrd-mini/discussions)

## Next Steps

After installation:

- **[Quickstart](quickstart.md)**: Run your first analysis
- **[CLI Guide](cli/index.md)**: Learn all commands
- **[Methods](methods/index.md)**: Understand the algorithms
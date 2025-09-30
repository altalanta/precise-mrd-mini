.PHONY: setup clean test coverage lint format simulate report ci-small smoke help

PYTHON := python3
PIP := pip3
PYTEST := pytest
COVERAGE := coverage
BLACK := black
ISORT := isort
FLAKE8 := flake8

# Default target
help:
	@echo "Available targets:"
	@echo "  setup         - Install dependencies and setup development environment"
	@echo "  clean         - Clean build artifacts and cache files"
	@echo "  test          - Run unit tests"
	@echo "  coverage      - Run tests with coverage reporting"
	@echo "  lint          - Run code linting (flake8)"
	@echo "  format        - Format code with black and isort"
	@echo "  simulate      - Run full simulation grid (configs/default.yaml)"
	@echo "  ci-small      - Run quick simulation for CI (configs/simulate_small.yaml)"
	@echo "  report        - Generate HTML report from latest simulation"
	@echo "  profile       - Run performance profiling"

# Setup development environment
setup:
	$(PIP) install -r requirements.txt
	$(PIP) install -e .
	pre-commit install
	@echo "Checking for Rust toolchain..."
	@if command -v cargo >/dev/null 2>&1; then \
		echo "Building Rust extensions..."; \
		cd rust_ext && cargo build --release || echo "Rust build failed, falling back to Python"; \
	else \
		echo "Rust not found, using pure Python implementation"; \
	fi

# Clean build artifacts
clean:
	rm -rf build/ dist/ *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	$(COVERAGE) erase

# Run tests
test:
	$(PYTEST) tests/ -v

# Run tests with coverage
coverage:
	$(COVERAGE) run -m pytest tests/
	$(COVERAGE) report -m
	$(COVERAGE) html
	@echo "Generating coverage badge..."
	coverage-badge -o coverage.svg -f
	@echo "Coverage report: htmlcov/index.html"

# Lint code
lint:
	$(FLAKE8) src/ tests/
	mypy src/precise_mrd/

# Format code
format:
	$(BLACK) src/ tests/
	$(ISORT) src/ tests/

# Run full simulation
simulate:
	$(PYTHON) -m precise_mrd.simulate --config configs/default.yaml

# Run quick simulation for CI
ci-small:
	$(PYTHON) -m precise_mrd.simulate --config configs/simulate_small.yaml

# Run smoke test - fast end-to-end pipeline
smoke:
	@echo "Running smoke test..."
	@mkdir -p data/smoke reports
	$(PYTHON) -m precise_mrd.cli smoke --seed 7 --out data/smoke --config configs/smoke.yaml
	@if [ -f data/smoke/smoke/metrics.json ]; then \
		cp data/smoke/smoke/metrics.json reports/; \
		cp data/smoke/smoke/*.html reports/ 2>/dev/null || true; \
		cp data/smoke/smoke/*.png reports/ 2>/dev/null || true; \
	fi
	@echo "Smoke test completed successfully!"

# Generate report
report:
	$(PYTHON) -m precise_mrd.report

# Run performance profiling
profile:
	$(PYTHON) -m precise_mrd.profiling

# Install package in development mode
install:
	$(PIP) install -e .
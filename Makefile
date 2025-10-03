.PHONY: setup clean test coverage lint format smoke determinism stat-sanity clean-safe help

PYTHON := python3

# Default target
help:
	@echo "Available targets:"
	@echo "  setup         - Install dependencies and setup development environment"
	@echo "  clean         - Clean build artifacts and cache files"
	@echo "  test          - Run unit tests"
	@echo "  coverage      - Run tests with coverage reporting"
	@echo "  lint          - Run code linting"
	@echo "  format        - Format code with black and isort"
	@echo "  smoke         - Run fast end-to-end pipeline"
	@echo "  determinism   - Run determinism verification"
	@echo "  stat-sanity   - Run statistical sanity tests"
	@echo "  clean-safe    - Clean generated artifacts safely"

# Setup development environment
setup:
	$(PYTHON) -m pip install -e .[dev]
	@echo "Setup complete!"

# Clean build artifacts
clean:
	rm -rf build/ dist/ *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Run tests
test:
	$(PYTHON) -m pytest tests/ -x --tb=short

# Run tests with coverage
coverage:
	$(PYTHON) -m pytest tests/ --cov=precise_mrd --cov-report=term-missing --cov-report=html

# Lint code
lint:
	$(PYTHON) -m flake8 src/ tests/ --max-line-length=100
	$(PYTHON) -m mypy src/precise_mrd/ --ignore-missing-imports

# Format code
format:
	$(PYTHON) -m black src/ tests/
	$(PYTHON) -m isort src/ tests/

# Run smoke test - fast end-to-end pipeline
smoke:
	@echo "Running smoke test..."
	@mkdir -p data/smoke reports
	$(PYTHON) -m precise_mrd.cli smoke --seed 7 --out data/smoke --config configs/smoke.yaml
	@echo "Smoke test completed successfully!"

# Run determinism check
determinism:
	@echo "Running determinism verification..."
	$(PYTHON) -m precise_mrd.cli smoke --seed 7 --out data/det_a --config configs/smoke.yaml
	@cp reports/metrics.json /tmp/metrics_run1.json
	@cp reports/auto_report.html /tmp/report_run1.html 2>/dev/null || true
	@sha256sum reports/metrics.json reports/auto_report.html > /tmp/hash1.txt 2>/dev/null || sha256sum reports/metrics.json > /tmp/hash1.txt
	@rm -f reports/metrics.json reports/auto_report.html reports/run_context.json
	$(PYTHON) -m precise_mrd.cli smoke --seed 7 --out data/det_b --config configs/smoke.yaml
	@cp reports/metrics.json /tmp/metrics_run2.json
	@cp reports/auto_report.html /tmp/report_run2.html 2>/dev/null || true
	@sha256sum reports/metrics.json reports/auto_report.html > /tmp/hash2.txt 2>/dev/null || sha256sum reports/metrics.json > /tmp/hash2.txt
	@echo "Comparing hashes..."
	@if diff /tmp/hash1.txt /tmp/hash2.txt >/dev/null 2>&1; then \
		echo "✅ Determinism verified - identical hashes"; \
	else \
		echo "❌ Determinism FAILED - hashes differ"; \
		echo "Hash comparison:"; \
		diff /tmp/hash1.txt /tmp/hash2.txt || true; \
		echo "Metrics diff:"; \
		diff /tmp/metrics_run1.json /tmp/metrics_run2.json || true; \
		exit 1; \
	fi
	@$(PYTHON) -c "from precise_mrd.determinism_utils import write_manifest; write_manifest(['reports/metrics.json', 'reports/auto_report.html'])" 2>/dev/null || true
	@echo "Hash manifest written to reports/hash_manifest.txt"

# Run statistical sanity tests
stat-sanity:
	@echo "Running statistical sanity tests..."
	$(PYTHON) -m pytest tests/stat_tests/ -v --tb=short

# Clean generated artifacts safely (keep source)
clean-safe:
	rm -rf data/ reports/ .snakemake/ __pycache__/
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	@echo "Generated artifacts cleaned safely"
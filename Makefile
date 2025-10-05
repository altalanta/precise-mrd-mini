.PHONY: setup clean test coverage lint format smoke determinism stat-sanity clean-safe help
.PHONY: eval-lob eval-lod eval-loq eval-contamination eval-stratified eval-all docs serve-docs

PYTHON := python3
SEED := 7

# Default target
help:
	@echo "Available targets:"
	@echo "  setup         - Install dependencies and setup development environment"
	@echo "  clean         - Clean build artifacts and cache files"
	@echo "  test          - Run unit tests"
	@echo "  coverage      - Run tests with coverage reporting"
	@echo "  lint          - Run code linting (ruff + mypy)"
	@echo "  format        - Format code with black and isort"
	@echo "  smoke         - Run fast end-to-end pipeline"
	@echo "  determinism   - Run determinism verification"
	@echo "  stat-sanity   - Run statistical sanity tests"
	@echo "  clean-safe    - Clean generated artifacts safely"
	@echo ""
	@echo "Detection Limit Analytics:"
	@echo "  eval-lob      - Estimate Limit of Blank"
	@echo "  eval-lod      - Estimate Limit of Detection"
	@echo "  eval-loq      - Estimate Limit of Quantification"
	@echo "  eval-contamination - Run contamination stress testing"
	@echo "  eval-stratified    - Run stratified power analysis"
	@echo "  eval-all      - Run all detection limit evaluations"
	@echo ""
	@echo "Documentation:"
	@echo "  docs          - Build documentation site"
	@echo "  serve-docs    - Serve documentation locally"

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
	$(PYTHON) -m ruff check src/ tests/
	$(PYTHON) -m mypy src/precise_mrd/

# Format code
format:
	$(PYTHON) -m black src/ tests/
	$(PYTHON) -m isort src/ tests/

# Run smoke test - fast end-to-end pipeline with basic detection limits
smoke:
	@echo "Running smoke test..."
	@mkdir -p data/smoke reports
	$(PYTHON) -m precise_mrd.cli smoke --seed 7 --out data/smoke --config configs/smoke.yaml
	@echo "Running basic detection limit checks..."
	$(PYTHON) -c "
	from precise_mrd.config import load_config
	from precise_mrd.eval.lod import estimate_lob, estimate_lod
	import numpy as np
	config = load_config('configs/smoke.yaml')
	rng = np.random.default_rng(7)
	# Quick LoB estimation
	lob_results = estimate_lob(config, rng, n_blank_runs=20)
	print(f'  LoB (quick): {lob_results[\"lob_value\"]:.3f}')
	# Quick LoD estimation
	lod_results = estimate_lod(config, rng, af_range=(1e-3, 1e-2), depth_values=[5000], n_replicates=10)
	lod_af = lod_results['depth_results'][5000]['lod_af']
	print(f'  LoD (quick): {lod_af:.2e} AF')
	"
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

# Detection Limit Analytics
eval-lob:
	@echo "Estimating Limit of Blank (LoB)..."
	@mkdir -p reports
	$(PYTHON) -c "
	from precise_mrd.config import load_config
	from precise_mrd.eval.lod import estimate_lob
	import numpy as np
	config = load_config('configs/smoke.yaml')
	rng = np.random.default_rng($(SEED))
	results = estimate_lob(config, rng, n_blank_runs=50)
	print(f'LoB estimated: {results[\"lob_value\"]:.3f}')
	"

eval-lod:
	@echo "Estimating Limit of Detection (LoD)..."
	@mkdir -p reports
	$(PYTHON) -c "
	from precise_mrd.config import load_config
	from precise_mrd.eval.lod import estimate_lod
	import numpy as np
	config = load_config('configs/smoke.yaml')
	rng = np.random.default_rng($(SEED))
	results = estimate_lod(config, rng, af_range=(1e-4, 1e-2), depth_values=[1000, 5000], n_replicates=25)
	for depth, data in results['depth_results'].items():
		print(f'LoD at {depth} depth: {data[\"lod_af\"]:.2e} AF')
	"

eval-loq:
	@echo "Estimating Limit of Quantification (LoQ)..."
	@mkdir -p reports
	$(PYTHON) -c "
	from precise_mrd.config import load_config
	from precise_mrd.eval.lod import estimate_loq
	import numpy as np
	config = load_config('configs/smoke.yaml')
	rng = np.random.default_rng($(SEED))
	results = estimate_loq(config, rng, af_range=(1e-4, 1e-2), depth_values=[1000, 5000], n_replicates=25)
	for depth, data in results['depth_results'].items():
		loq = data['loq_af_cv']
		print(f'LoQ at {depth} depth: {loq:.2e} AF' if loq else f'LoQ at {depth} depth: Not found')
	"

eval-contamination:
	@echo "Running contamination stress testing..."
	@mkdir -p reports
	$(PYTHON) -c "
	from precise_mrd.config import load_config
	from precise_mrd.sim.contamination import run_contamination_stress_test
	import numpy as np
	config = load_config('configs/smoke.yaml')
	rng = np.random.default_rng($(SEED))
	results = run_contamination_stress_test(config, rng)
	print('Contamination stress testing completed')
	"

eval-stratified:
	@echo "Running stratified power and calibration analysis..."
	@mkdir -p reports
	$(PYTHON) -c "
	from precise_mrd.config import load_config
	from precise_mrd.eval.stratified import run_stratified_analysis
	import numpy as np
	config = load_config('configs/smoke.yaml')
	rng = np.random.default_rng($(SEED))
	power_results, calib_results = run_stratified_analysis(config, rng)
	print('Stratified analysis completed')
	"

eval-all: eval-lob eval-lod eval-loq eval-contamination eval-stratified
	@echo "All detection limit evaluations completed!"
	@echo "Check reports/ directory for artifacts"

# Documentation
docs:
	@echo "Building documentation..."
	$(PYTHON) -m pip install -e .[docs] || (echo "Installing docs dependencies..." && pip install mkdocs-material mkdocs-gen-files 'mkdocstrings[python]')
	mkdocs build

serve-docs:
	@echo "Serving documentation at http://localhost:8000"
	$(PYTHON) -m pip install -e .[docs] || (echo "Installing docs dependencies..." && pip install mkdocs-material mkdocs-gen-files 'mkdocstrings[python]')
	mkdocs serve
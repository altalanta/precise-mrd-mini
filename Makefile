SHELL := /bin/bash

PYTHON ?= python3
UV ?= uv
REPORTS_DIR := reports

.PHONY: setup fmt lint type test smoke determinism eval-lob eval-lod eval-loq \
        eval-contamination eval-stratified validate-model performance docs clean build release \
        dvc-setup dvc-status dvc-diff dvc-push dvc-pull dvc-repro dvc-experiment

setup:
	$(PYTHON) -m pip install --upgrade pip $(UV)
	$(UV) sync --frozen --extra dev --extra docs || $(UV) sync --extra dev --extra docs

fmt:
	$(UV) run ruff format src tests

lint:
	$(UV) run ruff check src tests

type:
	$(UV) run mypy src/precise_mrd

test:
	$(UV) run pytest -q

test-integration:
	$(UV) run pytest -m integration --tb=short

test-contamination:
	$(UV) run pytest -m contamination --tb=short

test-stratified:
	$(UV) run pytest -m stratified --tb=short

test-determinism:
	$(UV) run pytest -m determinism --tb=short

test-all:
	$(UV) run pytest --tb=short

# Integration Tests
test-integration-quick:
	$(UV) run python tests/integration/run_integration_tests.py quick

test-integration-contamination:
	$(UV) run python tests/integration/run_integration_tests.py contamination

test-integration-stratified:
	$(UV) run python tests/integration/run_integration_tests.py stratified

test-integration-deterministic:
	$(UV) run python tests/integration/run_integration_tests.py deterministic

test-integration-performance:
	$(UV) run python tests/integration/run_integration_tests.py performance

test-integration-full:
	$(UV) run python tests/integration/run_integration_tests.py full

# List available integration tests
test-integration-list:
	$(UV) run python tests/integration/run_integration_tests.py --list

smoke:
	$(UV) run precise-mrd smoke --out-dir data/smoke

determinism:
	$(UV) run precise-mrd determinism --out-dir data/determinism

eval-lob:
	$(UV) run precise-mrd eval-lob

eval-lod:
	$(UV) run precise-mrd eval-lod

eval-loq:
	$(UV) run precise-mrd eval-loq

eval-contamination:
	$(UV) run precise-mrd eval-contamination

eval-stratified:
	$(UV) run precise-mrd eval-stratified

performance:
	$(UV) run precise-mrd performance

validate-model:
	$(UV) run precise-mrd validate-model

docs:
	$(UV) run mkdocs build --strict

build:
	$(UV) run python -m build

release: build
	$(UV) run python -m twine check dist/*

clean:
	rm -rf build dist *.egg-info $(REPORTS_DIR) data __pycache__
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +

# DVC Commands
dvc-setup:
	$(UV) run python dvc-workflows.py setup

dvc-status:
	dvc status

dvc-diff:
	dvc diff

dvc-push:
	dvc push

dvc-pull:
	dvc pull

dvc-repro:
	dvc repro

dvc-experiment:
	@echo "Usage: make dvc-experiment name=<experiment_name>"
	$(UV) run python dvc-workflows.py experiment --name $(name)

# Enhanced pipeline commands with DVC
smoke-dvc: dvc-setup
	$(UV) run python dvc-workflows.py smoke

determinism-dvc: dvc-setup
	$(UV) run python dvc-workflows.py determinism

eval-dvc: dvc-setup
	$(UV) run python dvc-workflows.py evaluation

validate-dvc: dvc-setup
	$(UV) run python dvc-workflows.py validation

docs-dvc: dvc-setup
	$(UV) run python dvc-workflows.py docs

all-dvc: dvc-setup
	$(UV) run python dvc-workflows.py all

.PHONY: profile
profile:
	$(UV) run python profile_pipeline.py

.PHONY: mlflow-ui
mlflow-ui:
	@echo "Starting MLflow UI..."
	@echo "Access it at http://1227.0.0.1:5000"
	@$(UV) run mlflow ui

.PHONY: help
help:
	@echo "Available commands:"

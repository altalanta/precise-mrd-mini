SHELL := /bin/bash

PYTHON ?= python3
UV ?= uv
REPORTS_DIR := reports

.PHONY: setup fmt lint type test smoke determinism eval-lob eval-lod eval-loq \
        eval-contamination eval-stratified docs clean build release

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

docs:
	$(UV) run mkdocs build --strict

build:
	$(UV) run python -m build

release: build
	$(UV) run python -m twine check dist/*

clean:
	rm -rf build dist *.egg-info $(REPORTS_DIR) data __pycache__
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +

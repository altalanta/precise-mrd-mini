# precise-mrd-mini

Compact, deterministic MRD simulation and calling pipeline for bioinformatics prototyping.

## Overview
- Simulates ctDNA reads with configurable VAFs, UMI families, and sequencing error.
- Collapses UMIs, fits beta-binomial background error, and computes per-variant/global MRD statistics.
- Renders lightweight Markdown/HTML reports with QC metrics, ROC/PR curves, and LoD grids.
- Ships with docs, notebooks, and tests to mirror end-to-end assay development responsibilities.

## Quickstart
1. `conda env create -f environment.yml && conda activate precise-mrd-mini`
2. `pre-commit install`
3. `snakemake -j 4 --directory workflow`
4. Find the rendered report at `reports/auto_report.md` / `reports/auto_report.html` and figures under `reports/figures/`.

## Repo Map
- `src/mrd/`: Typer CLIs for simulation, UMI collapsing, error modelling, MRD calling, reporting.
- `workflow/Snakefile`: orchestrates the five-stage pipeline (simulate → collapse → error_model → mrd_call → report).
- `docs/`: assay notes, validation outline, manuscript draft, abstract, reading log template.
- `notebooks/`: exploratory and sweep notebooks producing LoD grids.
- `tests/`: pytest suite covering simulation determinism, UMI collapse logic, error model stats, and MRD calling.
- `reports/`: Jinja2 template, generated figures, slides stub.
- `data/`: default synthetic configuration and ground-truth placeholder.

## JD Mapping (responsibility → repo artifact)
- Simulation design & data generation → `src/mrd/simulate.py`, `data/synthetic_config.yaml`.
- Read collapsing / consensus → `src/mrd/umi.py`, tests in `tests/test_umi.py`.
- Statistical modelling & MRD calling → `src/mrd/error_model.py`, `src/mrd/mrd_call.py`, tests in `tests/test_error_model.py`, `tests/test_mrd_call.py`.
- Reporting & stakeholder communication → `src/mrd/report.py`, `reports/template.md.jinja`, notebooks, docs package.
- Engineering quality (tooling, CI, reproducibility) → `environment.yml`, `pyproject.toml`, `.pre-commit-config.yaml`, `.github/workflows/ci.yml`, `Makefile`, pytest suite.

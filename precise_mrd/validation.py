"""Artifact validation utilities for deterministic contract enforcement."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping

import jsonschema
import pandas as pd
import pandera as pa
from importlib import resources

BASE_REQUIRED_ARTIFACTS = [
    "metrics.json",
    "run_context.json",
    "hash_manifest.txt",
]

JSON_SCHEMA_FILES: Mapping[str, str] = {
    "metrics.json": "metrics.schema.json",
    "run_context.json": "run_context.schema.json",
    "lob.json": "lob.schema.json",
    "contam_sensitivity.json": "contamination.schema.json",
    "power_by_stratum.json": "power.schema.json",
    "calibration_by_bin.json": "calibration.schema.json",
}

PANDERA_SCHEMAS: Mapping[str, pa.DataFrameSchema] = {
    "lod_table.csv": pa.DataFrameSchema(
        {
            "depth": pa.Column(pa.Int64, checks=pa.Check.ge(0), nullable=False),
            "lod_af": pa.Column(float, checks=pa.Check.gt(0)),
            "lod_ci_lower": pa.Column(float, checks=pa.Check.gt(0)),
            "lod_ci_upper": pa.Column(float, checks=pa.Check.gt(0)),
            "target_detection_rate": pa.Column(float, checks=pa.Check.in_range(min_value=0, max_value=1)),
            "n_replicates": pa.Column(pa.Int64, checks=pa.Check.ge(1)),
        },
        coerce=True,
        strict=False,
    ),
    "loq_table.csv": pa.DataFrameSchema(
        {
            "depth": pa.Column(pa.Int64, checks=pa.Check.ge(0), nullable=False),
            "loq_af_cv": pa.Column(float, nullable=True),
            "loq_af_abs_error": pa.Column(float, nullable=True),
            "cv_threshold": pa.Column(float, checks=pa.Check.ge(0)),
            "abs_error_threshold": pa.Column(float, nullable=True),
            "n_replicates": pa.Column(pa.Int64, checks=pa.Check.ge(1)),
        },
        coerce=True,
        strict=False,
    ),
    "calibration_by_bin.csv": pa.DataFrameSchema(
        {
            "depth": pa.Column(pa.Int64, checks=pa.Check.ge(0)),
            "af": pa.Column(float, checks=pa.Check.in_range(0, 1)),
            "ece": pa.Column(float, checks=pa.Check.in_range(0, 1)),
            "max_ce": pa.Column(float, checks=pa.Check.in_range(0, 1)),
            "n_samples": pa.Column(pa.Int64, checks=pa.Check.ge(0)),
        },
        coerce=True,
        strict=False,
    ),
}


def _load_json_schema(name: str) -> Dict[str, Any]:
    with resources.as_file(resources.files("precise_mrd.assets.schemas") / name) as schema_path:
        with open(schema_path, "r", encoding="utf-8") as fh:
            return json.load(fh)


def _validate_json(path: Path, schema_name: str) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    schema = _load_json_schema(schema_name)
    jsonschema.validate(payload, schema)


def _validate_csv(path: Path, schema: pa.DataFrameSchema) -> None:
    frame = pd.read_csv(path)
    schema.validate(frame, lazy=True)


def validate_artifacts(run_dir: Path) -> None:
    """Validate artifacts inside the reports directory."""
    run_dir = Path(run_dir)
    reports_dir = run_dir if run_dir.name == "reports" else run_dir / "reports"

    if not reports_dir.exists():
        raise AssertionError(f"Reports directory not found: {reports_dir}")

    for required in BASE_REQUIRED_ARTIFACTS:
        candidate = reports_dir / required
        if not candidate.exists():
            raise AssertionError(f"Required artifact missing: {candidate}")
        if candidate.suffix == ".txt" and not candidate.read_text(encoding="utf-8").strip():
            raise AssertionError(f"Artifact is empty: {candidate}")

    for filename, schema_name in JSON_SCHEMA_FILES.items():
        candidate = reports_dir / filename
        if candidate.exists():
            _validate_json(candidate, schema_name)

    for filename, schema in PANDERA_SCHEMAS.items():
        candidate = reports_dir / filename
        if candidate.exists():
            _validate_csv(candidate, schema)

    heatmap = reports_dir / "contam_heatmap.png"
    if heatmap.exists() and heatmap.stat().st_size == 0:
        raise AssertionError(f"Heatmap artifact is empty: {heatmap}")


def _read_manifest(path: Path) -> Dict[str, str]:
    entries: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split("  ", 1)
            if len(parts) != 2:
                continue
            hash_value, file_path = parts
            entries[file_path] = hash_value
    return entries


def assert_hashes_stable(previous_manifest: Path, current_manifest: Path) -> None:
    """Ensure two manifest files contain identical hashes."""
    prev_entries = _read_manifest(Path(previous_manifest))
    curr_entries = _read_manifest(Path(current_manifest))

    if prev_entries != curr_entries:
        prev_keys = set(prev_entries)
        curr_keys = set(curr_entries)
        missing = sorted(prev_keys - curr_keys)
        unexpected = sorted(curr_keys - prev_keys)
        changed = sorted(
            key
            for key in prev_keys & curr_keys
            if prev_entries[key] != curr_entries[key]
        )
        details = {
            "missing": missing,
            "unexpected": unexpected,
            "changed": changed,
        }
        raise AssertionError(f"Hash manifest mismatch detected: {json.dumps(details, indent=2)}")

"""Artifact contract tests for the Precise MRD CLI."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from precise_mrd.validation import assert_hashes_stable, validate_artifacts

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = REPO_ROOT / "src"


def _cli_env() -> dict[str, str]:
    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH")
    src_str = str(SRC_PATH)
    env["PYTHONPATH"] = f"{src_str}{os.pathsep}{pythonpath}" if pythonpath else src_str
    return env


def _write_minimal_config(directory: Path) -> Path:
    config_path = directory / "test_config.yaml"
    config_path.write_text(
        """
run_id: unit_test
seed: 11
simulation:
  allele_fractions: [0.01]
  umi_depths: [200]
  n_replicates: 2
  n_bootstrap: 10
umi:
  min_family_size: 2
  max_family_size: 10
  quality_threshold: 20
  consensus_threshold: 0.6
stats:
  test_type: poisson
  alpha: 0.05
  fdr_method: benjamini_hochberg
lod:
  detection_threshold: 0.95
  confidence_level: 0.95
""".strip(),
    )
    return config_path


def _run_cli(cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "precise_mrd.cli", *args],
        cwd=cwd,
        env=_cli_env(),
        text=True,
        capture_output=True,
    )


def _reports_dir(cwd: Path) -> Path:
    return cwd / "reports"


def test_smoke_artifacts_validate(tmp_path: Path) -> None:
    """Smoke command should produce schema-compliant artifacts."""

    config_path = _write_minimal_config(tmp_path)
    result = _run_cli(
        tmp_path,
        "--seed",
        "13",
        "--config",
        str(config_path),
        "smoke",
        "--out-dir",
        "data/out",
    )
    assert result.returncode == 0, result.stderr

    validate_artifacts(_reports_dir(tmp_path))


def test_manifest_consistency_between_runs(tmp_path: Path) -> None:
    """Running smoke twice with identical seeds should yield identical manifests."""

    config_path = _write_minimal_config(tmp_path)

    first = _run_cli(
        tmp_path,
        "--seed",
        "7",
        "--config",
        str(config_path),
        "smoke",
        "--out-dir",
        "data/run1",
    )
    assert first.returncode == 0, first.stderr

    reports_dir = _reports_dir(tmp_path)
    validate_artifacts(reports_dir)

    manifest_one = reports_dir / "hash_manifest.txt"
    snapshot = reports_dir / "hash_manifest_run1.txt"
    shutil.copy2(manifest_one, snapshot)

    second = _run_cli(
        tmp_path,
        "--seed",
        "7",
        "--config",
        str(config_path),
        "smoke",
        "--out-dir",
        "data/run2",
    )
    assert second.returncode == 0, second.stderr

    validate_artifacts(reports_dir)
    manifest_two = reports_dir / "hash_manifest.txt"

    assert_hashes_stable(snapshot, manifest_two)


@pytest.mark.slow
def test_detection_limit_artifacts(tmp_path: Path) -> None:
    """Evaluation commands should emit schema-compliant artifacts."""

    config_path = _write_minimal_config(tmp_path)

    commands = [
        ("smoke", ["--out-dir", "data/eval_base"]),
        ("eval-lob", ["--n-blank", "12"]),
        ("eval-lod", ["--replicates", "6"]),
        ("eval-loq", ["--replicates", "6"]),
        ("eval-contamination", []),
        ("eval-stratified", []),
    ]

    for subcommand, options in commands:
        result = _run_cli(
            tmp_path,
            "--seed",
            "9",
            "--config",
            str(config_path),
            subcommand,
            *options,
        )
        assert result.returncode == 0, f"{subcommand} failed: {result.stderr}"

    validate_artifacts(_reports_dir(tmp_path))

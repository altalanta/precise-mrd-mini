"""Unit tests for hash manifest utilities."""

from __future__ import annotations

from pathlib import Path

import pytest

from precise_mrd.validation import assert_hashes_stable


def _write_manifest(path: Path, entries: dict[str, str]) -> None:
    path.write_text("\n".join(f"{value}  {key}" for key, value in entries.items()) + "\n", encoding="utf-8")


def test_assert_hashes_stable_pass(tmp_path: Path) -> None:
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    entries = {"reports/metrics.json": "a" * 64, "reports/run_context.json": "b" * 64}
    _write_manifest(first, entries)
    _write_manifest(second, entries)

    assert_hashes_stable(first, second)


def test_assert_hashes_stable_fail(tmp_path: Path) -> None:
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    _write_manifest(first, {"reports/metrics.json": "a" * 64})
    _write_manifest(second, {"reports/metrics.json": "b" * 64})

    with pytest.raises(AssertionError):
        assert_hashes_stable(first, second)

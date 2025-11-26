"""Artifact hashing utilities for determinism verification."""

from __future__ import annotations

import hashlib
import pathlib
from collections.abc import Sequence


def hash_file(path: str | pathlib.Path) -> str:
    """Compute SHA256 hash of a file.

    Args:
        path: Path to file to hash

    Returns:
        SHA256 hash as hexadecimal string
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        # Read in chunks to handle large files efficiently
        for chunk in iter(lambda: f.read(1 << 20), b""):  # 1MB chunks
            h.update(chunk)
    return h.hexdigest()


def hash_dir(directory: str | pathlib.Path, pattern: str = "*") -> dict[str, str]:
    """Compute SHA256 hashes for all files matching pattern in directory.

    Args:
        directory: Directory to search
        pattern: Glob pattern for files to hash

    Returns:
        Dictionary mapping relative paths to SHA256 hashes
    """
    dir_path = pathlib.Path(directory)
    hashes = {}

    for file_path in dir_path.rglob(pattern):
        if file_path.is_file():
            relative_path = file_path.relative_to(dir_path)
            hashes[str(relative_path)] = hash_file(file_path)

    return hashes


def write_manifest(
    paths: Sequence[str | pathlib.Path],
    out_manifest: str | pathlib.Path = "reports/hash_manifest.txt",
) -> None:
    """Write hash manifest file for given paths.

    Creates a manifest file with SHA256 hashes for all specified paths,
    in the format: <hash>  <path>

    Args:
        paths: Sequence of file paths to hash
        out_manifest: Output manifest file path
    """
    manifest_path = pathlib.Path(out_manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    with open(manifest_path, "w") as f:
        for path in paths:
            path_obj = pathlib.Path(path)
            if path_obj.exists():
                file_hash = hash_file(path_obj)
                f.write(f"{file_hash}  {path}\n")
            else:
                f.write(f"MISSING  {path}\n")


def verify_manifest(manifest_path: str | pathlib.Path) -> dict[str, bool]:
    """Verify files against a hash manifest.

    Args:
        manifest_path: Path to manifest file

    Returns:
        Dictionary mapping file paths to verification status (True/False)
    """
    results = {}

    with open(manifest_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split("  ", 1)
            if len(parts) != 2:
                continue

            expected_hash, file_path = parts

            if expected_hash == "MISSING":
                results[file_path] = False
                continue

            try:
                actual_hash = hash_file(file_path)
                results[file_path] = actual_hash == expected_hash
            except Exception:
                results[file_path] = False

    return results

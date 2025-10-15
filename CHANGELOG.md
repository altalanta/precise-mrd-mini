# Changelog

## [Unreleased]
- unify CLI workflow around `precise-mrd` with global `--seed/--config` options and lock-based packaging via `uv`
- add artifact validation module with JSON schema versioning, Pandera contracts, and hash-manifest assertions
- introduce property-based statistical regression tests (Hypothesis) covering type-I error, LoD monotonicity, FDR monotonicity, bootstrap coverage, and contamination sensitivity
- refresh documentation and README with CLI-first quickstart, uv setup instructions, and schema-version guarantees
- overhaul CI/CD: uv-driven lint/type/test, Docker determinism checks, GHCR publishing on tags, docs deployment, and trusted PyPI release workflow

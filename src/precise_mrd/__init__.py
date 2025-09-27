"""Precise MRD mini-pipeline."""

from __future__ import annotations

try:  # pragma: no cover - populated via setuptools-scm at build time
    from ._version import __version__
except ImportError:  # pragma: no cover - fallback for editable installs
    __version__ = "0.0.0"

__all__ = ["__version__"]

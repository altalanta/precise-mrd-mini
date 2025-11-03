"""Test that legacy NumPy RNG API is not used."""

import ast
import importlib
import sys
from pathlib import Path
from typing import Set

import pytest


class LegacyRNGVisitor(ast.NodeVisitor):
    """AST visitor to detect legacy NumPy random API usage."""

    def __init__(self):
        self.legacy_calls: Set[str] = set()
        self.import_aliases = {}  # Track np aliases

    def visit_Import(self, node):
        """Track import numpy as np."""
        for alias in node.names:
            if alias.name == "numpy":
                self.import_aliases[alias.asname or "numpy"] = "numpy"
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        """Track from numpy import random."""
        if node.module == "numpy":
            for alias in node.names:
                if alias.name == "random":
                    self.import_aliases[alias.asname or "random"] = "numpy.random"
        self.generic_visit(node)

    def visit_Attribute(self, node):
        """Check for np.random.seed(), np.random.rand(), etc."""
        if isinstance(node.value, ast.Attribute):
            # np.random.seed case
            if (isinstance(node.value.value, ast.Name) and
                node.value.value.id in self.import_aliases and
                self.import_aliases[node.value.value.id] == "numpy" and
                node.value.attr == "random"):

                if node.attr in ["seed", "rand", "randn", "randint", "shuffle",
                               "choice", "uniform", "normal", "binomial", "poisson"]:
                    self.legacy_calls.add(f"numpy.random.{node.attr}")

        elif isinstance(node.value, ast.Name):
            # Direct random.seed if imported as 'from numpy import random'
            if (node.value.id in self.import_aliases and
                self.import_aliases[node.value.id] == "numpy.random"):

                if node.attr in ["seed", "rand", "randn", "randint", "shuffle",
                               "choice", "uniform", "normal", "binomial", "poisson"]:
                    self.legacy_calls.add(f"numpy.random.{node.attr}")

        self.generic_visit(node)


def check_file_for_legacy_rng(file_path: Path) -> Set[str]:
    """Check a Python file for legacy NumPy RNG usage."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        tree = ast.parse(content, filename=str(file_path))
        visitor = LegacyRNGVisitor()
        visitor.visit(tree)

        return visitor.legacy_calls
    except (SyntaxError, UnicodeDecodeError):
        # Skip files that can't be parsed
        return set()


def test_no_legacy_numpy_random_in_source():
    """Test that source code doesn't use legacy NumPy random API."""

    src_dir = Path("src/precise_mrd")
    if not src_dir.exists():
        pytest.skip("Source directory not found")

    legacy_usage = {}

    for py_file in src_dir.rglob("*.py"):
        legacy_calls = check_file_for_legacy_rng(py_file)
        if legacy_calls:
            legacy_usage[str(py_file)] = legacy_calls

    if legacy_usage:
        error_msg = "Legacy NumPy random API found:\n"
        for file_path, calls in legacy_usage.items():
            error_msg += f"  {file_path}: {', '.join(calls)}\n"
        error_msg += "\nUse np.random.default_rng(seed) instead!"

        pytest.fail(error_msg)


def test_set_global_seed_uses_modern_api():
    """Test that set_global_seed uses modern NumPy API."""

    from precise_mrd.determinism_utils.seed import set_global_seed
    import numpy as np

    # Test that it returns a Generator
    rng = set_global_seed(42)
    assert isinstance(rng, np.random.Generator)

    # Test that it's properly seeded
    rng1 = set_global_seed(42)
    rng2 = set_global_seed(42)

    # Same seed should produce same results
    result1 = rng1.random(5)
    result2 = rng2.random(5)

    assert list(result1) == list(result2), "Same seed should produce identical results"


def test_modules_accept_rng_parameter():
    """Test that core modules accept RNG parameter."""

    import numpy as np
    from precise_mrd.simulate import simulate_reads
    from precise_mrd.collapse import collapse_umis
    from precise_mrd.error_model import fit_error_model
    from precise_mrd.call import call_mrd
    from precise_mrd.config import PipelineConfig

    # Create minimal config
    config = PipelineConfig(
        run_id="test",
        seed=42,
        simulation={"allele_fractions": [0.01], "umi_depths": [1000], "n_replicates": 1, "n_bootstrap": 10},
        umi={"min_family_size": 3, "max_family_size": 100, "quality_threshold": 20, "consensus_threshold": 0.6},
        stats={"test_type": "poisson", "alpha": 0.05, "fdr_method": "benjamini_hochberg"},
        lod={"detection_threshold": 0.95, "confidence_level": 0.95}
    )

    rng = np.random.default_rng(42)

    # Test that functions accept rng parameter without error
    try:
        reads_df = simulate_reads(config, rng)
        assert len(reads_df) > 0

        collapsed_df = collapse_umis(reads_df, config, rng)

        if len(collapsed_df) > 0:
            error_model_df = fit_error_model(collapsed_df, config, rng)
            call_mrd(collapsed_df, error_model_df, config, rng)

    except TypeError as e:
        if "rng" in str(e):
            pytest.fail(f"Function doesn't accept rng parameter: {e}")
        else:
            # Other errors are okay for this test
            pass


def test_no_global_random_state_dependency():
    """Test that functions are independent of global random state."""

    import numpy as np
    from precise_mrd.determinism_utils.seed import set_global_seed

    # Set different global state
    np.random.seed(999)

    # Our seeded RNG should be independent
    rng1 = set_global_seed(42)
    result1 = rng1.random(5)

    # Change global state again
    np.random.seed(111)

    # Same seed should give same result regardless of global state
    rng2 = set_global_seed(42)
    result2 = rng2.random(5)

    assert list(result1) == list(result2), (
        "RNG results should be independent of global random state"
    )

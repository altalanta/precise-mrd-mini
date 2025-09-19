from __future__ import annotations

import numpy as np
import pandas as pd

from mrd import error_model


def test_p_alt_monotonic() -> None:
    alpha, beta = 2.0, 50.0
    depth = 100
    p_high = error_model.p_alt(10, depth, alpha, beta)
    p_low = error_model.p_alt(2, depth, alpha, beta)
    assert p_high < p_low


def test_method_of_moments_reasonable() -> None:
    fractions = np.array([0.0, 0.01, 0.02, 0.0, 0.03])
    alpha, beta = error_model.method_of_moments_beta_binomial(fractions)
    assert alpha > 0
    assert beta > alpha


def test_cli_panel_simulation(tmp_path) -> None:
    collapsed = pd.DataFrame(
        {"chrom": ["chr1"], "pos": [1], "alt_count": [0], "total_count": [100]}
    )
    path = tmp_path / "collapsed.parquet"
    collapsed.to_parquet(path, index=False)
    output = tmp_path / "params.parquet"
    error_model.run(collapsed_path=path, output_path=output, seed=1)
    result = pd.read_parquet(output)
    assert {"alpha", "beta"}.issubset(result.columns)

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from mrd import mrd_call


def test_mrd_call_outputs(tmp_path: Path) -> None:
    collapsed = pd.DataFrame(
        {
            "patient_id": ["P", "P"],
            "chrom": ["chr1", "chr1"],
            "pos": [1, 2],
            "alt_count": [10, 1],
            "total_count": [100, 100],
            "family_sizes": [[2, 3], [2, 2]],
        }
    )
    params = pd.DataFrame({"chrom": ["chr1", "chr1"], "pos": [1, 2], "alpha": [1.5, 1.5], "beta": [400.0, 400.0]})
    collapsed_path = tmp_path / "collapsed.parquet"
    params_path = tmp_path / "params.parquet"
    collapsed.to_parquet(collapsed_path, index=False)
    params.to_parquet(params_path, index=False)
    calls_path = tmp_path / "calls.csv"
    summary_path = tmp_path / "summary.json"
    mrd_call.run(collapsed_path=collapsed_path, error_model_path=params_path, output_path=calls_path, summary_path=summary_path)

    calls = pd.read_csv(calls_path)
    assert calls.loc[calls["pos"] == 1, "p_value"].iloc[0] < calls.loc[calls["pos"] == 2, "p_value"].iloc[0]
    assert (calls.sort_values("p_value")["q_value"].diff().fillna(0) >= -1e-9).all()
    summary = json.loads(summary_path.read_text())
    assert summary["total_variants"] == 2

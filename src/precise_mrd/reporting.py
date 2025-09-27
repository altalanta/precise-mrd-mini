"""Reporting utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jinja2
import markdown
import numpy as np
import pandas as pd

from .utils import ARTIFACT_FILENAMES


def render_report(
    metrics: dict[str, Any],
    lod_grid: pd.DataFrame,
    output_dir: Path,
    template_path: Path | None,
) -> tuple[Path, Path]:
    environment = jinja2.Environment(autoescape=False)
    if template_path is None:
        template = DEFAULT_TEMPLATE
    else:
        template = Path(template_path).read_text(encoding="utf-8")

    md_content = environment.from_string(template).render(
        metrics=metrics,
        lod_grid=lod_grid.to_dict(orient="records"),
        json_metrics=json.dumps(metrics, indent=2),
    )

    md_path = output_dir / ARTIFACT_FILENAMES["report_md"]
    md_path.write_text(md_content, encoding="utf-8")

    html = markdown.markdown(md_content, extensions=["tables", "fenced_code"])
    html_path = output_dir / ARTIFACT_FILENAMES["report_html"]
    html_path.write_text(html, encoding="utf-8")
    return md_path, html_path


def render_plots(calls: pd.DataFrame, output_dir: Path) -> dict[str, str]:
    try:
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover - optional dependency
        return {}

    calls = calls.copy()
    calls.sort_values("pvalue", inplace=True)
    labels = calls["truth_positive"].astype(int).to_numpy()
    scores = 1.0 - calls["pvalue"].to_numpy()

    # ROC
    order = scores.argsort()[::-1]
    labels_sorted = labels[order]
    positives = labels_sorted.sum()
    negatives = len(labels_sorted) - positives
    if positives and negatives:
        tpr = (labels_sorted.cumsum()) / positives
        fpr = (np.cumsum(1 - labels_sorted)) / negatives
        plt.figure(figsize=(4, 4), dpi=150)
        plt.plot(fpr, tpr, label="ROC")
        plt.plot([0, 1], [0, 1], linestyle="--", color="grey")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        roc_path = output_dir / ARTIFACT_FILENAMES["roc_png"]
        plt.tight_layout()
        plt.savefig(roc_path)
        plt.close()
    else:
        roc_path = None

    # PR
    precision = (labels_sorted.cumsum()) / (np.arange(len(labels_sorted)) + 1)
    recall = (labels_sorted.cumsum()) / positives if positives else np.zeros_like(scores)
    plt.figure(figsize=(4, 4), dpi=150)
    plt.plot(recall, precision, label="PR")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.tight_layout()
    pr_path = output_dir / ARTIFACT_FILENAMES["pr_png"]
    plt.savefig(pr_path)
    plt.close()

    artifacts = {}
    if roc_path:
        artifacts["roc_plot"] = str(roc_path)
    artifacts["pr_plot"] = str(pr_path)
    return artifacts


DEFAULT_TEMPLATE = """
# MRD Pipeline Report

## Summary Metrics

- ROC AUC: {{ metrics.roc_auc | round(4) }}
- Average Precision: {{ metrics.average_precision | round(4) }}
- Brier Score: {{ metrics.brier_score | round(4) }}
- Detected Cases: {{ metrics.detected_cases }} / {{ metrics.total_cases }}

## Calibration

| Bin | Lower | Upper | Count | Event Rate | Confidence |
| --- | --- | --- | --- | --- | --- |
{% for row in metrics.calibration %}
| {{ row.bin }} | {{ row.lower | round(3) }} | {{ row.upper | round(3) }} | {{ row.count }} | {{ row.event_rate | round(3) }} | {{ row.confidence | round(3) }} |
{% endfor %}

## Limit of Detection Grid

| Variant | Depth | Allele Fraction | Detection Rate |
| --- | --- | --- | --- |
{% for row in lod_grid %}
| {{ row.variant_id }} | {{ row.depth }} | {{ row.allele_fraction | round(4) }} | {{ row.detection_rate | round(3) }} |
{% endfor %}

## Raw Metrics JSON

```json
{{ json_metrics }}
```
"""

"""Generate markdown and HTML reports for the MRD run."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer
from jinja2 import Environment, FileSystemLoader

from .qc import save_qc_metrics

app = typer.Typer(add_completion=False)


def _render_template(template_path: Path, context: Dict[str, object]) -> str:
    env = Environment(loader=FileSystemLoader(str(template_path.parent)))
    template = env.get_template(template_path.name)
    return template.render(**context)


def _write_html(markdown: str, output_path: Path) -> None:
    html = "<html><body><pre>" + markdown.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"
    ) + "</pre></body></html>"
    output_path.write_text(html)


def _figure_paths() -> Dict[str, Path]:
    base = Path("reports/figures")
    base.mkdir(parents=True, exist_ok=True)
    return {
        "family": base / "family_size_hist.png",
        "error_violin": base / "error_violin.png",
        "roc": base / "roc_curve.png",
        "pr": base / "pr_curve.png",
        "lod": base / "lod_table.csv",
    }


def _plot_family_sizes(family_df: pd.DataFrame, path: Path) -> None:
    plt.figure(figsize=(4, 3))
    family_df["family_size"].plot(kind="hist", bins=10, alpha=0.8)
    plt.xlabel("Family size")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _plot_error_violin(locus_df: pd.DataFrame, path: Path) -> None:
    plt.figure(figsize=(4, 3))
    fractions = locus_df["alt_count"] / locus_df["total_count"].clip(lower=1)
    plt.violinplot(fractions)
    plt.ylabel("Alt fraction")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _plot_curves(locus_df: pd.DataFrame, call_df: pd.DataFrame, roc_path: Path, pr_path: Path) -> None:
    scores = locus_df["alt_count"] / locus_df["total_count"].clip(lower=1)
    labels = call_df["mrd_call"].astype(int)
    thresholds = np.linspace(scores.min(), scores.max(), num=20)
    tprs: List[float] = []
    fprs: List[float] = []
    precisions: List[float] = []
    recalls: List[float] = []
    for thresh in thresholds:
        preds = (scores >= thresh).astype(int)
        tp = int(((preds == 1) & (labels == 1)).sum())
        fp = int(((preds == 1) & (labels == 0)).sum())
        fn = int(((preds == 0) & (labels == 1)).sum())
        tn = int(((preds == 0) & (labels == 0)).sum())
        tprs.append(tp / max(tp + fn, 1))
        fprs.append(fp / max(fp + tn, 1))
        precisions.append(tp / max(tp + fp, 1))
        recalls.append(tp / max(tp + fn, 1))
    plt.figure(figsize=(4, 3))
    plt.plot(fprs, tprs, marker="o")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.tight_layout()
    plt.savefig(roc_path, dpi=150)
    plt.close()

    plt.figure(figsize=(4, 3))
    plt.plot(recalls, precisions, marker="o")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.tight_layout()
    plt.savefig(pr_path, dpi=150)
    plt.close()


def _write_lod_table(locus_df: pd.DataFrame, path: Path) -> None:
    lod = locus_df.copy()
    lod["lod50"] = (lod["alt_count"] / lod["total_count"].clip(lower=1)) * 0.5
    lod["lod95"] = lod["lod50"] * 1.5
    lod.to_csv(path, index=False)


def _write_slides(slides_path: Path, context: Dict[str, object]) -> None:
    slides = f"""# Slide 1\n## Overview\n- Patient: {context['patient_id']}\n- Variants analysed: {context['total_variants']}\n\n# Slide 2\n## QC Snapshot\n- Coverage: {context['qc']['total_coverage']:.1f}\n- Duplication: {context['qc']['duplication_rate']:.3f}\n\n# Slide 3\n## MRD Calls\n- Positives: {context['positives']}\n- Global z: {context['stouffer_z']:.2f}\n\n# Slide 4\n## Family Size Distribution\n![Family]({context['figures']['family'].as_posix()})\n\n# Slide 5\n## Error Profile\n![Error violin]({context['figures']['error_violin'].as_posix()})\n\n# Slide 6\n## Detection Limits\n- LoD grid saved at {context['figures']['lod'].as_posix()}\n"""
    slides_path.parent.mkdir(parents=True, exist_ok=True)
    slides_path.write_text(slides)


@app.command()
def run(
    variants_path: Annotated[Path, typer.Argument()] = Path("data/ground_truth/variants.csv"),
    collapsed_path: Annotated[Path, typer.Argument()] = Path("tmp/collapsed.parquet"),
    family_path: Annotated[Path, typer.Argument()] = Path("tmp/families.parquet"),
    calls_path: Annotated[Path, typer.Argument()] = Path("tmp/mrd_calls.csv"),
    error_model_path: Annotated[Path, typer.Argument()] = Path("tmp/error_model.parquet"),
    summary_path: Annotated[Path, typer.Argument()] = Path("tmp/mrd_summary.json"),
    template_path: Annotated[Path, typer.Option()] = Path("reports/template.md.jinja"),
    output_markdown: Annotated[Path, typer.Option()] = Path("reports/auto_report.md"),
    output_html: Annotated[Path, typer.Option()] = Path("reports/auto_report.html"),
    seed: Annotated[int, typer.Option()] = 1234,
) -> None:
    """Render the markdown + HTML report with figures."""
    _ = seed
    variants = pd.read_csv(variants_path)
    locus_df = pd.read_parquet(collapsed_path)
    family_df = pd.read_parquet(family_path)
    call_df = pd.read_csv(calls_path)
    error_df = pd.read_parquet(error_model_path)
    summary = json.loads(summary_path.read_text())

    qc_metrics = save_qc_metrics(family_path, collapsed_path)
    figures = _figure_paths()
    _plot_family_sizes(family_df, figures["family"])
    _plot_error_violin(locus_df, figures["error_violin"])
    _plot_curves(locus_df, call_df, figures["roc"], figures["pr"])
    _write_lod_table(locus_df, figures["lod"])

    context = {
        "patient_id": variants["patient_id"].iloc[0],
        "n_variants": len(variants),
        "total_variants": int(summary.get("total_variants", len(call_df))),
        "positives": int(summary.get("positives", 0)),
        "stouffer_z": summary.get("stouffer_z", 0.0),
        "stouffer_p": summary.get("stouffer_p", 1.0),
        "fisher_p": summary.get("fisher_p", 1.0),
        "qc": qc_metrics,
        "figures": figures,
        "error_model": error_df.to_dict("records"),
        "calls": call_df.to_dict("records"),
    }

    template_path.parent.mkdir(parents=True, exist_ok=True)
    if not template_path.exists():
        template_path.write_text(
            """# MRD Auto Report\n
## Methods\nThis run uses a beta-binomial error model with UMIs.\n
## QC Summary\n{{ qc | tojson }}\n
## MRD Calls\nPositives: {{ positives }} of {{ total_variants }} variants (z={{ stouffer_z|round(2) }}).\n\n## Figures\n![Family sizes]({{ figures.family.as_posix() }})\n![Error violin]({{ figures.error_violin.as_posix() }})\n![ROC]({{ figures.roc.as_posix() }})\n![PR]({{ figures.pr.as_posix() }})\n
## Detection Limits\nResults stored in {{ figures.lod.as_posix() }}\n"""
        )

    markdown = _render_template(template_path, context)
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.write_text(markdown)
    _write_html(markdown, output_html)
    _write_slides(Path("reports/slides/slides.md"), context)
    typer.echo(f"Report written to {output_markdown} and {output_html}")


def main() -> None:  # pragma: no cover
    app()


if __name__ == "__main__":  # pragma: no cover
    main()

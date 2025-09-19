"""UMI collapsing utilities."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import typer

app = typer.Typer(add_completion=False)


def hamming_distance(a: str, b: str) -> int:
    return sum(ch1 != ch2 for ch1, ch2 in zip(a, b)) + abs(len(a) - len(b))


@dataclass
class Cluster:
    representative: str
    rows: List[pd.Series]

    def add(self, row: pd.Series) -> None:
        self.rows.append(row)


def _cluster_family(rows: Iterable[pd.Series], max_distance: int) -> List[Cluster]:
    clusters: List[Cluster] = []
    for row in rows:
        assigned = False
        for cluster in clusters:
            if hamming_distance(row["umi"], cluster.representative) <= max_distance:
                cluster.add(row)
                assigned = True
                break
        if not assigned:
            clusters.append(Cluster(representative=row["umi"], rows=[row]))
    return clusters


def _consensus_base(sequences: List[str]) -> Tuple[str, Dict[str, int]]:
    counts: Dict[str, int] = {}
    consensus_chars: List[str] = []
    for pos in range(len(sequences[0])):
        column = [seq[pos] for seq in sequences]
        counter = Counter(column)
        consensus_char, _ = counter.most_common(1)[0]
        consensus_chars.append(consensus_char)
        for base, cnt in counter.items():
            counts[base] = counts.get(base, 0) + cnt
    return "".join(consensus_chars), counts


def collapse_reads(
    reads: pd.DataFrame,
    min_family_size: int = 2,
    max_distance: int = 1,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Collapse reads by UMI/start and produce per-family + per-locus summaries."""
    families: List[Dict[str, object]] = []
    locus_summary: Dict[Tuple[str, int], Dict[str, float]] = {}

    for (_, _, _), group in reads.groupby(["chrom", "start", "pos"]):
        clusters = _cluster_family((row for _, row in group.iterrows()), max_distance)
        for cluster in clusters:
            if len(cluster.rows) < min_family_size:
                continue
            sequences = [row.sequence for row in cluster.rows]
            consensus, _ = _consensus_base(sequences)
            true_alt = sum(int(row.true_alt) for row in cluster.rows)
            total = len(cluster.rows)
            families.append(
                {
                    "patient_id": cluster.rows[0].patient_id,
                    "chrom": cluster.rows[0].chrom,
                    "pos": cluster.rows[0].pos,
                    "umi_representative": cluster.representative,
                    "start": cluster.rows[0].start,
                    "consensus": consensus,
                    "family_size": total,
                    "alt_count": true_alt,
                }
            )
            key = (cluster.rows[0].chrom, cluster.rows[0].pos)
            entry = locus_summary.setdefault(
                key,
                {
                    "patient_id": cluster.rows[0].patient_id,
                    "chrom": cluster.rows[0].chrom,
                    "pos": cluster.rows[0].pos,
                    "alt_count": 0,
                    "total_count": 0,
                    "family_sizes": [],
                },
            )
            entry["alt_count"] += true_alt
            entry["total_count"] += total
            entry["family_sizes"].append(total)

    family_df = pd.DataFrame(families)
    locus_df = pd.DataFrame(
        [
            {
                **summary,
                "family_count": len(summary["family_sizes"]),
                "mean_family_size": float(np.mean(summary["family_sizes"]))
                if summary["family_sizes"]
                else 0.0,
            }
            for summary in locus_summary.values()
        ]
    )
    return family_df, locus_df


@app.command()
def run(
    reads_path: Annotated[Path, typer.Argument()] = Path("simulated_reads/reads.parquet"),
    output_family: Annotated[Path, typer.Option()] = Path("tmp/families.parquet"),
    output_locus: Annotated[Path, typer.Option()] = Path("tmp/collapsed.parquet"),
    min_family_size: Annotated[int, typer.Option()] = 2,
    seed: Annotated[int, typer.Option()] = 1234,  # kept for interface symmetry
) -> None:
    """Collapse UMIs and summarise per locus."""
    _ = seed
    reads = pd.read_parquet(reads_path)
    family_df, locus_df = collapse_reads(reads, min_family_size=min_family_size)
    output_family.parent.mkdir(parents=True, exist_ok=True)
    output_locus.parent.mkdir(parents=True, exist_ok=True)
    family_df.to_parquet(output_family, index=False)
    locus_df.to_parquet(output_locus, index=False)
    typer.echo(
        f"Collapsed {len(reads)} reads into {len(family_df)} families across {len(locus_df)} loci"
    )


def main() -> None:  # pragma: no cover
    app()


if __name__ == "__main__":  # pragma: no cover
    main()

"""Contamination and index-hopping stress testing for ctDNA/UMI assays.

This module implements contamination models including barcode index-hopping,
cross-sample contamination, and barcode collision effects on variant detection.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ..call import call_mrd
from ..collapse import collapse_umis
from ..config import PipelineConfig
from ..error_model import fit_error_model
from ..simulate import simulate_reads


class ContaminationSimulator:
    """Simulator for contamination effects in ctDNA/UMI sequencing."""

    def __init__(self, config: PipelineConfig, rng: np.random.Generator):
        self.config = config
        self.rng = rng
        self.contamination_results: dict[str, Any] | None = None

    def simulate_contamination_effects(
        self,
        hop_rates: list[float] | None = None,
        barcode_collision_rates: list[float] | None = None,
        cross_sample_proportions: list[float] | None = None,
        af_test_values: list[float] | None = None,
        depth_values: list[int] | None = None,
        n_replicates: int = 20,
    ) -> dict[str, Any]:
        """
        Simulates the effect of different contamination sources on MRD detection.
        """
        if hop_rates is None:
            hop_rates = [0.0, 0.001, 0.002, 0.005, 0.01]
        if barcode_collision_rates is None:
            barcode_collision_rates = [0.0, 0.0001, 0.0005, 0.001]
        if cross_sample_proportions is None:
            cross_sample_proportions = [0.0, 0.01, 0.05, 0.1]
        if af_test_values is None:
            af_test_values = [0.001, 0.005, 0.01]
        if depth_values is None:
            depth_values = [1000, 5000]
        print("Running contamination stress testing...")

        results = {}
        sensitivity_matrix = []

        # Test index-hopping effects
        print("  Testing index-hopping effects...")
        hop_results = self._test_index_hopping(
            hop_rates,
            af_test_values,
            depth_values,
            n_replicates,
        )
        results["index_hopping"] = hop_results

        # Test barcode collision effects
        print("  Testing barcode collision effects...")
        collision_results = self._test_barcode_collisions(
            barcode_collision_rates,
            af_test_values,
            depth_values,
            n_replicates,
        )
        results["barcode_collisions"] = collision_results

        # Test cross-sample contamination
        print("  Testing cross-sample contamination...")
        cross_contam_results = self._test_cross_sample_contamination(
            cross_sample_proportions,
            af_test_values,
            depth_values,
            n_replicates,
        )
        results["cross_sample_contamination"] = cross_contam_results

        # Create sensitivity matrix for heatmap
        sensitivity_matrix = self._create_sensitivity_matrix(
            results,
            af_test_values,
            depth_values,
        )
        results["sensitivity_matrix"] = sensitivity_matrix

        self.contamination_results = results
        return results

    def _test_index_hopping(
        self,
        hop_rates: list[float],
        af_values: list[float],
        depth_values: list[int],
        n_replicates: int,
    ) -> dict[str, Any]:
        """Test index-hopping contamination effects."""
        hop_results = {}

        for hop_rate in hop_rates:
            hop_results[hop_rate] = {}

            for af in af_values:
                hop_results[hop_rate][af] = {}

                for depth in depth_values:
                    sensitivity_scores = []

                    for rep in range(n_replicates):
                        # Create contaminated simulation
                        run_rng = np.random.default_rng(
                            self.config.seed + rep * 1000 + int(hop_rate * 1e6),
                        )

                        # Simulate with index hopping
                        contam_config = self._create_contamination_config(af, depth)
                        reads_df = self._simulate_with_index_hopping(
                            contam_config,
                            run_rng,
                            hop_rate,
                        )

                        # Process through pipeline
                        collapsed_df = collapse_umis(reads_df, contam_config, run_rng)
                        error_model = fit_error_model(
                            collapsed_df,
                            contam_config,
                            run_rng,
                        )
                        calls_df = call_mrd(
                            collapsed_df,
                            error_model,
                            contam_config,
                            run_rng,
                        )

                        # Calculate sensitivity (detection rate)
                        n_detected = len(calls_df[calls_df["variant_call"]])
                        # Approximate expected detections (crude estimate)
                        expected_detections = max(
                            1,
                            int(af * depth * (1 - hop_rate) * 0.9),
                        )  # 80% pipeline efficiency
                        sensitivity = (
                            min(1.0, n_detected / expected_detections)
                            if expected_detections > 0
                            else 0.0
                        )
                        sensitivity_scores.append(sensitivity)

                    hop_results[hop_rate][af][depth] = {
                        "mean_sensitivity": float(np.mean(sensitivity_scores)),
                        "std_sensitivity": float(np.std(sensitivity_scores)),
                        "sensitivity_scores": sensitivity_scores,
                        "n_replicates": n_replicates,
                    }

        return hop_results

    def _test_barcode_collisions(
        self,
        collision_rates: list[float],
        af_values: list[float],
        depth_values: list[int],
        n_replicates: int,
    ) -> dict[str, Any]:
        """Test barcode collision contamination effects."""
        collision_results = {}

        for collision_rate in collision_rates:
            collision_results[collision_rate] = {}

            for af in af_values:
                collision_results[collision_rate][af] = {}

                for depth in depth_values:
                    sensitivity_scores = []
                    false_positive_rates = []

                    for rep in range(n_replicates):
                        run_rng = np.random.default_rng(
                            self.config.seed + rep * 2000 + int(collision_rate * 1e6),
                        )

                        # Simulate with barcode collisions
                        contam_config = self._create_contamination_config(af, depth)
                        reads_df = self._simulate_with_barcode_collisions(
                            contam_config,
                            run_rng,
                            collision_rate,
                        )

                        # Process through pipeline
                        collapsed_df = collapse_umis(reads_df, contam_config, run_rng)
                        error_model = fit_error_model(
                            collapsed_df,
                            contam_config,
                            run_rng,
                        )
                        calls_df = call_mrd(
                            collapsed_df,
                            error_model,
                            contam_config,
                            run_rng,
                        )

                        # Calculate metrics
                        n_detected = len(calls_df[calls_df["variant_call"]])
                        expected_detections = max(1, int(af * depth * 0.8))
                        sensitivity = (
                            min(1.0, n_detected / expected_detections)
                            if expected_detections > 0
                            else 0.0
                        )

                        # Estimate false positive rate (excess detections)
                        excess_detections = max(0, n_detected - expected_detections)
                        fp_rate = excess_detections / depth if depth > 0 else 0.0

                        sensitivity_scores.append(sensitivity)
                        false_positive_rates.append(fp_rate)

                    collision_results[collision_rate][af][depth] = {
                        "mean_sensitivity": float(np.mean(sensitivity_scores)),
                        "std_sensitivity": float(np.std(sensitivity_scores)),
                        "mean_fp_rate": float(np.mean(false_positive_rates)),
                        "std_fp_rate": float(np.std(false_positive_rates)),
                        "n_replicates": n_replicates,
                    }

        return collision_results

    def _test_cross_sample_contamination(
        self,
        contamination_proportions: list[float],
        af_values: list[float],
        depth_values: list[int],
        n_replicates: int,
    ) -> dict[str, Any]:
        """Test cross-sample contamination effects."""
        cross_results = {}

        for contam_prop in contamination_proportions:
            cross_results[contam_prop] = {}

            for af in af_values:
                cross_results[contam_prop][af] = {}

                for depth in depth_values:
                    sensitivity_scores = []

                    for rep in range(n_replicates):
                        run_rng = np.random.default_rng(
                            self.config.seed + rep * 3000 + int(contam_prop * 1e6),
                        )

                        # Simulate with cross-sample contamination
                        contam_config = self._create_contamination_config(af, depth)
                        reads_df = self._simulate_with_cross_contamination(
                            contam_config,
                            run_rng,
                            contam_prop,
                        )

                        # Process through pipeline
                        collapsed_df = collapse_umis(reads_df, contam_config, run_rng)
                        error_model = fit_error_model(
                            collapsed_df,
                            contam_config,
                            run_rng,
                        )
                        calls_df = call_mrd(
                            collapsed_df,
                            error_model,
                            contam_config,
                            run_rng,
                        )

                        # Calculate sensitivity
                        n_detected = len(calls_df[calls_df["variant_call"]])
                        expected_detections = max(1, int(af * depth * 0.8))
                        sensitivity = (
                            min(1.0, n_detected / expected_detections)
                            if expected_detections > 0
                            else 0.0
                        )
                        sensitivity_scores.append(sensitivity)

                    cross_results[contam_prop][af][depth] = {
                        "mean_sensitivity": float(np.mean(sensitivity_scores)),
                        "std_sensitivity": float(np.std(sensitivity_scores)),
                        "n_replicates": n_replicates,
                    }

        return cross_results

    def _simulate_with_index_hopping(
        self,
        config: PipelineConfig,
        rng: np.random.Generator,
        hop_rate: float,
    ) -> pd.DataFrame:
        """Simulate reads with index-hopping contamination."""
        # Start with normal simulation
        reads_df = simulate_reads(config, rng)

        if hop_rate > 0:
            # Add hopped reads (contamination from other samples)
            n_reads = len(reads_df)
            n_hopped = rng.binomial(n_reads, hop_rate)

            if n_hopped > 0:
                # Create contaminating reads (different background pattern)
                contam_reads = reads_df.sample(n=n_hopped, random_state=rng).copy()

                # Modify contaminating reads (different barcode context)
                contam_reads["background_rate"] *= 2.0  # Higher error rate
                contam_reads["n_false_positives"] *= 1.5  # More artifacts
                contam_reads["sample_id"] = "hopped_" + contam_reads[
                    "sample_id"
                ].astype(str)

                # Combine with original reads
                reads_df = pd.concat([reads_df, contam_reads], ignore_index=True)

        return reads_df

    def _simulate_with_barcode_collisions(
        self,
        config: PipelineConfig,
        rng: np.random.Generator,
        collision_rate: float,
    ) -> pd.DataFrame:
        """Simulate reads with barcode collision artifacts."""
        reads_df = simulate_reads(config, rng)

        if collision_rate > 0:
            # Simulate UMI collisions leading to false consensus
            n_families = len(reads_df)
            n_collisions = rng.binomial(n_families, collision_rate)

            if n_collisions > 0:
                # Select families for collision
                collision_indices = rng.choice(
                    n_families,
                    size=n_collisions,
                    replace=False,
                )

                # Increase false positive rate for collided families
                reads_df.loc[collision_indices, "n_false_positives"] *= 2.0
                reads_df.loc[collision_indices, "background_rate"] *= 1.5

                # Reduce consensus quality
                reads_df.loc[collision_indices, "mean_quality"] *= 0.8

        return reads_df

    def _simulate_with_cross_contamination(
        self,
        config: PipelineConfig,
        rng: np.random.Generator,
        contam_proportion: float,
    ) -> pd.DataFrame:
        """Simulate reads with cross-sample contamination."""
        reads_df = simulate_reads(config, rng)

        if contam_proportion > 0:
            # Add contaminating sample with different AF
            n_reads = len(reads_df)
            n_contam = int(n_reads * contam_proportion)

            if n_contam > 0:
                # Create contaminating sample (higher AF)
                contam_af = min(
                    0.1,
                    reads_df["allele_fraction"].iloc[0] * 10,
                )  # 10x higher AF

                contam_config = self._create_contamination_config(
                    contam_af,
                    reads_df["target_depth"].iloc[0],
                )
                contam_reads = simulate_reads(contam_config, rng)

                # Take subset for contamination
                if len(contam_reads) > 0:
                    contam_subset = contam_reads.sample(
                        n=min(n_contam, len(contam_reads)),
                        random_state=rng,
                    ).copy()
                    contam_subset["sample_id"] = "contam_" + contam_subset[
                        "sample_id"
                    ].astype(str)

                    # Mix with original sample
                    reads_df = pd.concat([reads_df, contam_subset], ignore_index=True)

        return reads_df

    def _create_contamination_config(self, af: float, depth: int) -> PipelineConfig:
        """Create configuration for contamination testing."""
        return PipelineConfig(
            run_id=f"{self.config.run_id}_contam",
            seed=self.config.seed,
            simulation=type(self.config.simulation)(
                allele_fractions=[af],
                umi_depths=[depth],
                n_replicates=1,
                n_bootstrap=self.config.simulation.n_bootstrap,
            ),
            umi=self.config.umi,
            stats=self.config.stats,
            lod=self.config.lod,
        )

    def _create_sensitivity_matrix(
        self,
        results: dict[str, Any],
        af_values: list[float],
        depth_values: list[int],
    ) -> dict[str, Any]:
        """Create sensitivity matrix for heatmap visualization."""
        # Extract sensitivity data for different contamination types
        matrix_data = {}

        # Index hopping sensitivity matrix
        if "index_hopping" in results:
            hop_data = results["index_hopping"]
            hop_rates = list(hop_data.keys())

            sensitivity_matrix = np.zeros(
                (len(hop_rates), len(af_values) * len(depth_values)),
            )

            for i, hop_rate in enumerate(hop_rates):
                col_idx = 0
                for af in af_values:
                    for depth in depth_values:
                        if af in hop_data[hop_rate] and depth in hop_data[hop_rate][af]:
                            sensitivity = hop_data[hop_rate][af][depth][
                                "mean_sensitivity"
                            ]
                            sensitivity_matrix[i, col_idx] = sensitivity
                        col_idx += 1

            # Create labels
            condition_labels = [
                f"AF={af:.0e}, D={depth}" for af in af_values for depth in depth_values
            ]

            matrix_data["index_hopping"] = {
                "matrix": sensitivity_matrix.tolist(),
                "hop_rates": hop_rates,
                "condition_labels": condition_labels,
                "af_values": af_values,
                "depth_values": depth_values,
            }

        return matrix_data

    def generate_contamination_reports(self, output_dir: str = "reports") -> None:
        """Generate contamination analysis reports."""
        if not self.contamination_results:
            print("No contamination results to report")
            return

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Save contamination sensitivity results
        contam_payload = dict(self.contamination_results)
        contam_payload["schema_version"] = "1.0.0"
        contam_path = output_path / "contam_sensitivity.json"
        with open(contam_path, "w") as f:
            json.dump(contam_payload, f, indent=2)
        print(f"Contamination sensitivity results saved to {contam_path}")

        # Generate contamination heatmap
        self._plot_contamination_heatmap(output_path)

    def _plot_contamination_heatmap(self, output_path: Path) -> None:
        """Generate contamination sensitivity heatmap."""
        if "sensitivity_matrix" not in self.contamination_results:
            return

        matrix_data = self.contamination_results["sensitivity_matrix"]

        if "index_hopping" in matrix_data:
            hop_data = matrix_data["index_hopping"]

            plt.figure(figsize=(12, 8))

            # Create heatmap
            sensitivity_matrix = np.array(hop_data["matrix"])
            hop_rates = hop_data["hop_rates"]
            condition_labels = hop_data["condition_labels"]

            # Plot heatmap
            sns.heatmap(
                sensitivity_matrix,
                xticklabels=condition_labels,
                yticklabels=[f"{rate:.1%}" for rate in hop_rates],
                annot=True,
                fmt=".2f",
                cmap="RdYlBu_r",
                cbar_kws={"label": "Detection Sensitivity"},
            )

            plt.title(
                "Contamination Impact on Variant Detection\n(Index Hopping Effects)",
            )
            plt.xlabel("Test Conditions (AF, Depth)")
            plt.ylabel("Index Hopping Rate")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()

            heatmap_path = output_path / "contam_heatmap.png"
            plt.savefig(heatmap_path, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"Contamination heatmap saved to {heatmap_path}")


def run_contamination_stress_test(
    config: PipelineConfig,
    rng: np.random.Generator,
    output_dir: str = "reports",
) -> dict[str, Any]:
    """Run complete contamination stress testing suite.

    Args:
        config: Pipeline configuration
        rng: Random number generator
        output_dir: Output directory for reports

    Returns:
        Contamination analysis results
    """
    simulator = ContaminationSimulator(config, rng)

    # Run contamination simulations
    results = simulator.simulate_contamination_effects()

    # Generate reports
    simulator.generate_contamination_reports(output_dir)

    return results

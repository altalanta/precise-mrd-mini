"""Stratified power analysis and calibration by trinucleotide context and depth."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..call import call_mrd
from ..collapse import collapse_umis
from ..config import PipelineConfig
from ..error_model import fit_error_model
from ..simulate import simulate_reads


class StratifiedAnalyzer:
    """Analyzer for stratified power and calibration analysis."""

    def __init__(self, config: PipelineConfig, rng: np.random.Generator):
        self.config = config
        self.rng = rng
        self.power_results: dict[str, Any] | None = None
        self.calibration_results: dict[str, Any] | None = None

    def analyze_stratified_power(
        self,
        af_values: list[float] | None = None,
        depth_values: list[int] | None = None,
        contexts: list[str] | None = None,
        n_replicates: int = 50,
    ) -> dict[str, Any]:
        """Analyze detection power stratified by trinucleotide context and depth.

        Args:
            af_values: Allele fractions to test
            depth_values: UMI depths to test
            contexts: Trinucleotide contexts to stratify by
            n_replicates: Number of replicates per condition

        Returns:
            Dictionary with stratified power analysis results
        """
        print("Running stratified power analysis...")

        if af_values is None:
            af_values = [0.001, 0.005, 0.01, 0.05]
        if depth_values is None:
            depth_values = [1000, 5000, 10000]
        if contexts is None:
            contexts = ["CpG", "CHG", "CHH", "NpN"]

        power_results = {}

        for context in contexts:
            print(f"  Analyzing context: {context}")
            power_results[context] = {}

            for depth in depth_values:
                power_results[context][depth] = {}

                for af in af_values:
                    detection_rates = []

                    for rep in range(n_replicates):
                        run_rng = np.random.default_rng(
                            self.config.seed + rep * 1000 + hash(context) % 1000,
                        )

                        # Create context-specific config
                        context_config = self._create_context_config(af, depth, context)

                        # Run pipeline with context tagging
                        reads_df = self._simulate_with_context(
                            context_config,
                            run_rng,
                            context,
                        )
                        collapsed_df = collapse_umis(reads_df, context_config, run_rng)
                        error_model = fit_error_model(
                            collapsed_df,
                            context_config,
                            run_rng,
                        )
                        calls_df = call_mrd(
                            collapsed_df,
                            error_model,
                            context_config,
                            run_rng,
                        )

                        # Calculate detection rate for this context
                        if len(calls_df) > 0:
                            context_calls = calls_df[
                                calls_df.get("context", "NpN") == context
                            ]
                            detection_rate = len(
                                context_calls[context_calls["variant_call"]],
                            ) / max(1, len(context_calls))
                        else:
                            detection_rate = 0.0

                        detection_rates.append(detection_rate)

                    power_results[context][depth][af] = {
                        "mean_detection_rate": float(np.mean(detection_rates)),
                        "std_detection_rate": float(np.std(detection_rates)),
                        "detection_rates": detection_rates,
                        "n_replicates": n_replicates,
                    }

                    print(
                        f"    {context} @ depth={depth}, AF={af:.0e}: {np.mean(detection_rates):.3f} ± {np.std(detection_rates):.3f}",
                    )

        self.power_results = {
            "stratified_results": power_results,
            "af_values": af_values,
            "depth_values": depth_values,
            "contexts": contexts,
            "config_hash": self.config.config_hash(),
        }

        return self.power_results

    def analyze_calibration_by_bins(
        self,
        af_values: list[float] | None = None,
        depth_values: list[int] | None = None,
        n_bins: int = 10,
        n_replicates: int = 100,
    ) -> dict[str, Any]:
        """Analyze calibration stratified by AF and depth bins.

        Args:
            af_values: Allele fractions to test
            depth_values: UMI depths to test
            n_bins: Number of calibration bins
            n_replicates: Number of replicates per condition

        Returns:
            Dictionary with binned calibration results
        """
        print(f"Running calibration analysis with {n_bins} bins...")

        if af_values is None:
            af_values = [0.001, 0.005, 0.01, 0.05]
        if depth_values is None:
            depth_values = [1000, 5000, 10000]

        calibration_data = []

        for depth in depth_values:
            print(f"  Processing depth: {depth}")

            for af in af_values:
                predicted_probs = []
                true_labels = []

                for rep in range(n_replicates):
                    run_rng = np.random.default_rng(
                        self.config.seed + rep * 2000 + int(af * 1e6) + depth,
                    )

                    # Create test config
                    test_config = self._create_calibration_config(af, depth)

                    # Run pipeline
                    reads_df = simulate_reads(test_config, run_rng)
                    collapsed_df = collapse_umis(reads_df, test_config, run_rng)
                    error_model = fit_error_model(collapsed_df, test_config, run_rng)
                    calls_df = call_mrd(collapsed_df, error_model, test_config, run_rng)

                    # Extract predicted probabilities and true labels
                    if len(calls_df) > 0:
                        # Use p-value as proxy for predicted probability (inverted)
                        prob_scores = 1 - calls_df.get(
                            "p_value",
                            self.rng.uniform(0, 1, len(calls_df)),
                        )
                        true_positives = calls_df.get("variant_call", False)

                        predicted_probs.extend(prob_scores.tolist())
                        true_labels.extend(true_positives.tolist())

                if len(predicted_probs) > 0:
                    # Compute calibration metrics
                    calibration_metrics = self._compute_calibration_metrics(
                        np.array(predicted_probs),
                        np.array(true_labels),
                        n_bins,
                    )

                    calibration_data.append(
                        {
                            "depth": depth,
                            "af": af,
                            "ece": calibration_metrics["ece"],
                            "max_ce": calibration_metrics["max_ce"],
                            "bin_accuracies": calibration_metrics["bin_accuracies"],
                            "bin_confidences": calibration_metrics["bin_confidences"],
                            "bin_counts": calibration_metrics["bin_counts"],
                            "n_samples": len(predicted_probs),
                        },
                    )

        self.calibration_results = {
            "calibration_data": calibration_data,
            "af_values": af_values,
            "depth_values": depth_values,
            "n_bins": n_bins,
            "config_hash": self.config.config_hash(),
        }

        return self.calibration_results

    def _simulate_with_context(
        self,
        config: PipelineConfig,
        rng: np.random.Generator,
        context: str,
    ) -> pd.DataFrame:
        """Simulate reads with trinucleotide context tagging."""
        reads_df = simulate_reads(config, rng)

        # Add context-specific error modeling
        context_error_rates = {
            "CpG": 1.5,  # Higher error in CpG contexts
            "CHG": 1.2,  # Moderate error in CHG
            "CHH": 1.0,  # Baseline error in CHH
            "NpN": 0.8,  # Lower error in other contexts
        }

        error_multiplier = context_error_rates.get(context, 1.0)
        reads_df["background_rate"] *= error_multiplier
        reads_df["context"] = context

        # Adjust false positive counts based on context
        reads_df["n_false_positives"] = rng.binomial(
            reads_df["n_families"] - reads_df["n_true_variants"],
            reads_df["background_rate"] * error_multiplier,
        )

        return reads_df

    def _create_context_config(
        self,
        af: float,
        depth: int,
        context: str,
    ) -> PipelineConfig:
        """Create configuration for context-specific analysis."""
        return PipelineConfig(
            run_id=f"{self.config.run_id}_context_{context}",
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

    def _create_calibration_config(self, af: float, depth: int) -> PipelineConfig:
        """Create configuration for calibration analysis."""
        return PipelineConfig(
            run_id=f"{self.config.run_id}_calibration",
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

    def _compute_calibration_metrics(
        self,
        predicted_probs: np.ndarray,
        true_labels: np.ndarray,
        n_bins: int,
    ) -> dict[str, Any]:
        """Compute calibration metrics including ECE and binned accuracy."""
        # Create bins
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        bin_accuracies = []
        bin_confidences = []
        bin_counts = []

        ece = 0.0
        max_ce = 0.0

        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers, strict=False):
            # Find predictions in this bin
            in_bin = (predicted_probs > bin_lower) & (predicted_probs <= bin_upper)
            prop_in_bin = in_bin.mean()

            if prop_in_bin > 0:
                # Accuracy in this bin
                accuracy_in_bin = true_labels[in_bin].mean()
                avg_confidence_in_bin = predicted_probs[in_bin].mean()

                # Calibration error in this bin
                bin_ce = abs(avg_confidence_in_bin - accuracy_in_bin)
                ece += bin_ce * prop_in_bin
                max_ce = max(max_ce, bin_ce)

                bin_accuracies.append(accuracy_in_bin)
                bin_confidences.append(avg_confidence_in_bin)
                bin_counts.append(in_bin.sum())
            else:
                bin_accuracies.append(0.0)
                bin_confidences.append(0.0)
                bin_counts.append(0)

        return {
            "ece": float(ece),
            "max_ce": float(max_ce),
            "bin_accuracies": bin_accuracies,
            "bin_confidences": bin_confidences,
            "bin_counts": bin_counts,
        }

    def generate_stratified_reports(self, output_dir: str = "reports") -> None:
        """Generate stratified analysis reports."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Save power analysis results
        if self.power_results:
            power_payload = dict(self.power_results)
            power_payload["schema_version"] = "1.0.0"
            power_path = output_path / "power_by_stratum.json"
            with open(power_path, "w") as f:
                json.dump(power_payload, f, indent=2)
            print(f"Stratified power results saved to {power_path}")

        # Save calibration results
        if self.calibration_results:
            # Save as CSV for easier analysis
            calib_df = pd.DataFrame(self.calibration_results["calibration_data"])
            calib_path = output_path / "calibration_by_bin.csv"
            calib_df.to_csv(calib_path, index=False)
            print(f"Calibration by bin results saved to {calib_path}")

            # Also save as JSON
            calibration_payload = dict(self.calibration_results)
            calibration_payload["schema_version"] = "1.0.0"
            calib_json_path = output_path / "calibration_by_bin.json"
            with open(calib_json_path, "w") as f:
                json.dump(calibration_payload, f, indent=2)


def run_stratified_analysis(
    config: PipelineConfig,
    rng: np.random.Generator,
    output_dir: str = "reports",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run complete stratified power and calibration analysis.

    Args:
        config: Pipeline configuration
        rng: Random number generator
        output_dir: Output directory for reports

    Returns:
        Tuple of (power_results, calibration_results)
    """
    analyzer = StratifiedAnalyzer(config, rng)

    # Run stratified power analysis
    power_results = analyzer.analyze_stratified_power()

    # Run calibration analysis
    calibration_results = analyzer.analyze_calibration_by_bins()

    # Generate reports
    analyzer.generate_stratified_reports(output_dir)

    return power_results, calibration_results

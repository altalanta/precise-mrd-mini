"""Limit of Blank (LoB), Limit of Detection (LoD), and Limit of Quantification (LoQ) analysis.

This module implements formal detection limit analytics following CLSI guidelines
for clinical laboratory testing, adapted for ctDNA/UMI sequencing assays.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..call import call_mrd
from ..collapse import collapse_umis
from ..config import PipelineConfig
from ..error_model import fit_error_model
from ..simulate import simulate_reads


class LODAnalyzer:
    """Analyzer for Limit of Blank, Detection, and Quantification."""

    def __init__(self, config: PipelineConfig, rng: np.random.Generator):
        self.config = config
        self.rng = rng
        self.lob_results: dict[str, Any] | None = None
        self.lod_results: dict[str, Any] | None = None
        self.loq_results: dict[str, Any] | None = None

    def estimate_lob(self, n_blank_runs: int = 100) -> dict[str, Any]:
        """Estimate Limit of Blank (LoB).

        LoB represents the highest measurement result that is likely to be observed
        for a blank specimen. Calculated as the 95th percentile of blank measurements.

        Args:
            n_blank_runs: Number of blank (AF=0) simulations to run

        Returns:
            Dictionary with LoB results including:
            - lob_value: 95th percentile of blank test statistics
            - blank_mean: Mean of blank measurements
            - blank_std: Standard deviation of blank measurements
            - blank_measurements: All blank measurement values
        """
        print(f"Estimating LoB with {n_blank_runs} blank runs...")

        # Create blank configuration (AF = 0)
        blank_config = self._create_blank_config()
        blank_measurements = []

        for i in range(n_blank_runs):
            # Use different random seed for each run
            run_rng = np.random.default_rng(self.config.seed + i)

            # Simulate blank reads (no true variants)
            reads_df = simulate_reads(blank_config, run_rng)

            # Process through pipeline to get test statistic
            collapsed_df = collapse_umis(reads_df, blank_config, run_rng)
            error_model = fit_error_model(collapsed_df, blank_config, run_rng)
            calls_df = call_mrd(collapsed_df, error_model, blank_config, run_rng)

            # Extract test statistic (number of variant calls in blank)
            n_variant_calls = len(calls_df[calls_df["variant_call"]])
            blank_measurements.append(n_variant_calls)

        blank_measurements = np.array(blank_measurements)

        # Compute LoB as 95th percentile
        lob_value = np.percentile(blank_measurements, 95)
        blank_mean = np.mean(blank_measurements)
        blank_std = np.std(blank_measurements)

        self.lob_results = {
            "lob_value": float(lob_value),
            "blank_mean": float(blank_mean),
            "blank_std": float(blank_std),
            "blank_measurements": blank_measurements.tolist(),
            "n_blank_runs": n_blank_runs,
            "percentile": 95,
            "config_hash": self.config.config_hash(),
        }

        print(
            f"LoB estimated: {lob_value:.3f} (mean: {blank_mean:.3f}, std: {blank_std:.3f})",
        )
        return self.lob_results

    def estimate_lod(
        self,
        af_range: tuple[float, float] = (1e-4, 1e-2),
        depth_values: list[int] | None = None,
        n_replicates: int = 50,
        target_detection_rate: float = 0.95,
        alpha: float = 0.05,
        beta: float = 0.05,
    ) -> dict[str, Any]:
        """Estimate Limit of Detection (LoD).

        LoD is the lowest analyte concentration likely to be reliably detected.
        Calculated as the concentration yielding 95% detection probability.

        Args:
            af_range: Range of allele fractions to test (min, max)
            depth_values: UMI depths to test
            n_replicates: Number of replicates per AF/depth combination
            target_detection_rate: Target detection rate (e.g., 0.95 for 95%)
            alpha: Type I error rate
            beta: Type II error rate

        Returns:
            Dictionary with LoD results for each depth
        """
        print(f"Estimating LoD across AF range {af_range} at depths {depth_values}...")

        if depth_values is None:
            depth_values = [1000, 5000, 10000]

        # Generate AF grid (log-spaced)
        af_min, af_max = af_range
        af_values = np.logspace(np.log10(af_min), np.log10(af_max), 15)

        lod_results = {}

        for depth in depth_values:
            print(f"  Processing depth {depth}...")

            hit_rates = []
            af_tested = []

            for af in af_values:
                # Run detection experiments at this AF/depth
                detection_count = 0

                for rep in range(n_replicates):
                    run_rng = np.random.default_rng(
                        self.config.seed + rep * 1000 + int(af * 1e6),
                    )

                    # Create config for this AF/depth
                    test_config = self._create_lod_config(af, depth)

                    # Run pipeline
                    reads_df = simulate_reads(test_config, run_rng)
                    collapsed_df = collapse_umis(reads_df, test_config, run_rng)
                    error_model = fit_error_model(collapsed_df, test_config, run_rng)
                    calls_df = call_mrd(collapsed_df, error_model, test_config, run_rng)

                    # Check if variant was detected
                    if len(calls_df[calls_df["variant_call"]]) > 0:
                        detection_count += 1

                hit_rate = detection_count / n_replicates
                hit_rates.append(hit_rate)
                af_tested.append(af)

            # Fit logistic curve to hit rate vs AF
            lod_af = self._fit_detection_curve(
                af_tested,
                hit_rates,
                target_detection_rate,
            )

            # Compute confidence intervals using bootstrap
            lod_ci = self._bootstrap_lod_ci(
                af_tested,
                hit_rates,
                target_detection_rate,
                n_bootstrap=200,
            )

            lod_results[depth] = {
                "lod_af": float(lod_af),
                "lod_ci_lower": float(lod_ci[0]),
                "lod_ci_upper": float(lod_ci[1]),
                "af_values": [float(x) for x in af_tested],
                "hit_rates": hit_rates,
                "target_detection_rate": target_detection_rate,
                "n_replicates": n_replicates,
            }

            print(
                f"    LoD at depth {depth}: {lod_af:.2e} AF [{lod_ci[0]:.2e}, {lod_ci[1]:.2e}]",
            )

        self.lod_results = {
            "depth_results": lod_results,
            "af_range": af_range,
            "target_detection_rate": target_detection_rate,
            "alpha": alpha,
            "beta": beta,
            "config_hash": self.config.config_hash(),
        }

        return self.lod_results

    def estimate_loq(
        self,
        af_range: tuple[float, float] = (1e-4, 1e-2),
        depth_values: list[int] | None = None,
        n_replicates: int = 50,
        cv_threshold: float = 0.20,
        abs_error_threshold: float | None = None,
    ) -> dict[str, Any]:
        """Estimate Limit of Quantification (LoQ).

        LoQ is the lowest concentration at which quantitative measurements can be made
        with acceptable precision (typically CV ≤ 20% or absolute error ≤ threshold).

        Args:
            af_range: Range of allele fractions to test
            depth_values: UMI depths to test
            n_replicates: Number of replicates per AF/depth combination
            cv_threshold: Maximum acceptable coefficient of variation
            abs_error_threshold: Maximum acceptable absolute error (optional)

        Returns:
            Dictionary with LoQ results for each depth
        """
        print(f"Estimating LoQ with CV threshold {cv_threshold:.1%}...")

        if depth_values is None:
            depth_values = [1000, 5000, 10000]

        af_min, af_max = af_range
        af_values = np.logspace(np.log10(af_min), np.log10(af_max), 12)

        loq_results = {}

        for depth in depth_values:
            print(f"  Processing depth {depth}...")

            cv_values = []
            abs_errors = []
            af_tested = []

            for af in af_values:
                estimated_afs = []

                for rep in range(n_replicates):
                    run_rng = np.random.default_rng(
                        self.config.seed + rep * 2000 + int(af * 1e6),
                    )

                    # Create config for this AF/depth
                    test_config = self._create_lod_config(af, depth)

                    # Run pipeline and estimate AF
                    reads_df = simulate_reads(test_config, run_rng)
                    collapsed_df = collapse_umis(reads_df, test_config, run_rng)
                    error_model = fit_error_model(collapsed_df, test_config, run_rng)
                    calls_df = call_mrd(collapsed_df, error_model, test_config, run_rng)

                    # Estimate AF from variant calls
                    if not calls_df.empty:
                        # Simple AF estimation: variants detected / total UMIs
                        estimated_af = len(calls_df[calls_df["variant_call"]]) / len(
                            calls_df,
                        )
                        estimated_afs.append(estimated_af)
                    else:
                        estimated_afs.append(0.0)

                if len(estimated_afs) > 1:
                    mean_af = np.mean(estimated_afs)
                    std_af = np.std(estimated_afs)
                    cv = std_af / mean_af if mean_af > 0 else np.inf
                    abs_error = abs(mean_af - af)

                    cv_values.append(cv)
                    abs_errors.append(abs_error)
                    af_tested.append(af)

            # Find LoQ as lowest AF meeting precision criteria
            loq_af_cv = None
            loq_af_abs = None

            # CV-based LoQ
            for i, cv in enumerate(cv_values):
                if cv <= cv_threshold:
                    loq_af_cv = af_tested[i]
                    break

            # Absolute error-based LoQ (if threshold provided)
            if abs_error_threshold is not None:
                for i, abs_err in enumerate(abs_errors):
                    if abs_err <= abs_error_threshold:
                        loq_af_abs = af_tested[i]
                        break

            loq_results[depth] = {
                "loq_af_cv": float(loq_af_cv) if loq_af_cv is not None else None,
                "loq_af_abs_error": float(loq_af_abs)
                if loq_af_abs is not None
                else None,
                "af_values": [float(x) for x in af_tested],
                "cv_values": cv_values,
                "abs_errors": abs_errors,
                "cv_threshold": cv_threshold,
                "abs_error_threshold": abs_error_threshold,
                "n_replicates": n_replicates,
            }

            cv_str = f"{loq_af_cv:.2e}" if loq_af_cv else "Not found"
            print(f"    LoQ (CV) at depth {depth}: {cv_str} AF")

        self.loq_results = {
            "depth_results": loq_results,
            "af_range": af_range,
            "cv_threshold": cv_threshold,
            "abs_error_threshold": abs_error_threshold,
            "config_hash": self.config.config_hash(),
        }

        return self.loq_results

    def generate_reports(self, output_dir: str = "reports") -> None:
        """Generate LoB/LoD/LoQ reports and visualizations."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Save LoB results
        if self.lob_results:
            lob_payload = dict(self.lob_results)
            lob_payload["schema_version"] = "1.0.0"
            lob_path = output_path / "lob.json"
            with open(lob_path, "w") as f:
                json.dump(lob_payload, f, indent=2)
            print(f"LoB results saved to {lob_path}")

        # Save LoD results
        if self.lod_results:
            self._save_lod_table(output_path)
            self._plot_lod_curves(output_path)

        # Save LoQ results
        if self.loq_results:
            self._save_loq_table(output_path)

    def _create_blank_config(self) -> PipelineConfig:
        """Create configuration for blank (AF=0) experiments."""
        blank_config = PipelineConfig(
            run_id=f"{self.config.run_id}_blank",
            seed=self.config.seed,
            simulation=type(self.config.simulation)(
                allele_fractions=[0.0],  # Blank samples
                umi_depths=self.config.simulation.umi_depths[:1],  # Use first depth
                n_replicates=1,
                n_bootstrap=self.config.simulation.n_bootstrap,
            ),
            umi=self.config.umi,
            stats=self.config.stats,
            lod=self.config.lod,
        )
        return blank_config

    def _create_lod_config(self, af: float, depth: int) -> PipelineConfig:
        """Create configuration for LoD/LoQ experiments."""
        lod_config = PipelineConfig(
            run_id=f"{self.config.run_id}_lod",
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
        return lod_config

    def _fit_detection_curve(
        self,
        af_values: list[float],
        hit_rates: list[float],
        target_rate: float = 0.95,
    ) -> float:
        """Fit logistic curve to detection data and find LoD."""
        # Convert to log scale for fitting
        log_af = np.log10(af_values)

        # Fit logistic regression
        from scipy.optimize import curve_fit

        def logistic(x, a, b):
            return 1 / (1 + np.exp(-(a * x + b)))

        try:
            popt, _ = curve_fit(logistic, log_af, hit_rates, maxfev=2000)
            a, b = popt

            # Solve for AF giving target detection rate
            # target_rate = 1 / (1 + exp(-(a * log_af + b)))
            # Solving: log_af = (logit(target_rate) - b) / a
            logit_target = np.log(target_rate / (1 - target_rate))
            log_af_lod = (logit_target - b) / a

            return 10**log_af_lod

        except (ValueError, ZeroDivisionError, RuntimeError):
            # Fallback: linear interpolation
            from scipy.interpolate import interp1d

            if len(af_values) > 1:
                interp_func = interp1d(
                    hit_rates,
                    af_values,
                    bounds_error=False,
                    fill_value="extrapolate",
                )
                return float(interp_func(target_rate))
            else:
                return af_values[0]

    def _bootstrap_lod_ci(
        self,
        af_values: list[float],
        hit_rates: list[float],
        target_rate: float,
        n_bootstrap: int = 200,
    ) -> tuple[float, float]:
        """Compute bootstrap confidence intervals for LoD."""
        bootstrap_lods = []

        for _ in range(n_bootstrap):
            # Bootstrap resample
            indices = self.rng.choice(len(af_values), size=len(af_values), replace=True)
            boot_af = [af_values[i] for i in indices]
            boot_hit = [hit_rates[i] for i in indices]

            try:
                boot_lod = self._fit_detection_curve(boot_af, boot_hit, target_rate)
                bootstrap_lods.append(boot_lod)
            except (ValueError, ZeroDivisionError, RuntimeError):
                continue

        if bootstrap_lods:
            ci_lower = np.percentile(bootstrap_lods, 2.5)
            ci_upper = np.percentile(bootstrap_lods, 97.5)
            return ci_lower, ci_upper
        else:
            # Fallback
            return af_values[0], af_values[-1]

    def _save_lod_table(self, output_path: Path) -> None:
        """Save LoD results table as CSV."""
        if not self.lod_results:
            return

        lod_data = []
        for depth, results in self.lod_results["depth_results"].items():
            lod_data.append(
                {
                    "depth": depth,
                    "lod_af": results["lod_af"],
                    "lod_ci_lower": results["lod_ci_lower"],
                    "lod_ci_upper": results["lod_ci_upper"],
                    "target_detection_rate": results["target_detection_rate"],
                    "n_replicates": results["n_replicates"],
                },
            )

        lod_df = pd.DataFrame(lod_data)
        lod_path = output_path / "lod_table.csv"
        lod_df.to_csv(lod_path, index=False)
        print(f"LoD table saved to {lod_path}")

    def _save_loq_table(self, output_path: Path) -> None:
        """Save LoQ results table as CSV."""
        if not self.loq_results:
            return

        loq_data = []
        for depth, results in self.loq_results["depth_results"].items():
            loq_data.append(
                {
                    "depth": depth,
                    "loq_af_cv": results["loq_af_cv"],
                    "loq_af_abs_error": results["loq_af_abs_error"],
                    "cv_threshold": results["cv_threshold"],
                    "abs_error_threshold": results["abs_error_threshold"],
                    "n_replicates": results["n_replicates"],
                },
            )

        loq_df = pd.DataFrame(loq_data)
        loq_path = output_path / "loq_table.csv"
        loq_df.to_csv(loq_path, index=False)
        print(f"LoQ table saved to {loq_path}")

    def _plot_lod_curves(self, output_path: Path) -> None:
        """Generate LoD detection curves plot."""
        if not self.lod_results:
            return

        plt.style.use("default")
        fig, axes = plt.subplots(
            1,
            len(self.lod_results["depth_results"]),
            figsize=(5 * len(self.lod_results["depth_results"]), 4),
        )

        if len(self.lod_results["depth_results"]) == 1:
            axes = [axes]

        for i, (depth, results) in enumerate(self.lod_results["depth_results"].items()):
            ax = axes[i]

            af_values = results["af_values"]
            hit_rates = results["hit_rates"]
            lod_af = results["lod_af"]
            target_rate = results["target_detection_rate"]

            # Plot detection curve
            ax.semilogx(
                af_values,
                hit_rates,
                "o-",
                linewidth=2,
                markersize=6,
                label=f"Observed (n={results['n_replicates']})",
            )

            # Mark LoD
            ax.axvline(
                lod_af,
                color="red",
                linestyle="--",
                linewidth=2,
                label=f"LoD = {lod_af:.2e}",
            )
            ax.axhline(
                target_rate,
                color="gray",
                linestyle=":",
                alpha=0.7,
                label=f"{target_rate:.0%} Detection",
            )

            ax.set_xlabel("Allele Fraction")
            ax.set_ylabel("Detection Rate")
            ax.set_title(f"LoD Curve (Depth = {depth})")
            ax.grid(True, alpha=0.3)
            ax.legend()
            ax.set_ylim(-0.05, 1.05)

        plt.tight_layout()
        lod_plot_path = output_path / "lod_curves.png"
        plt.savefig(lod_plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"LoD curves saved to {lod_plot_path}")


def estimate_lob(
    config: PipelineConfig,
    rng: np.random.Generator,
    n_blank_runs: int = 100,
) -> dict[str, Any]:
    """Convenience function to estimate LoB."""
    analyzer = LODAnalyzer(config, rng)
    return analyzer.estimate_lob(n_blank_runs)


def estimate_lod(
    config: PipelineConfig,
    rng: np.random.Generator,
    af_range: tuple[float, float] = (1e-4, 1e-2),
    depth_values: list[int] | None = None,
    n_replicates: int = 50,
) -> dict[str, Any]:
    """Convenience function to estimate LoD."""
    analyzer = LODAnalyzer(config, rng)
    return analyzer.estimate_lod(af_range, depth_values, n_replicates)


def estimate_loq(
    config: PipelineConfig,
    rng: np.random.Generator,
    af_range: tuple[float, float] = (1e-4, 1e-2),
    depth_values: list[int] | None = None,
    n_replicates: int = 50,
    cv_threshold: float = 0.20,
) -> dict[str, Any]:
    """Convenience function to estimate LoQ."""
    analyzer = LODAnalyzer(config, rng)
    return analyzer.estimate_loq(af_range, depth_values, n_replicates, cv_threshold)

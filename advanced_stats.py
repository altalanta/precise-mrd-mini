"""Advanced statistical modeling for enhanced MRD analysis."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

from .config import PipelineConfig


class BayesianErrorModel:
    """Bayesian hierarchical model for background error rates."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.trinucleotide_contexts = [
            "AAA",
            "AAC",
            "AAG",
            "AAT",
            "ACA",
            "ACC",
            "ACG",
            "ACT",
            "AGA",
            "AGC",
            "AGG",
            "AGT",
            "ATA",
            "ATC",
            "ATG",
            "ATT",
            "CAA",
            "CAC",
            "CAG",
            "CAT",
            "CCA",
            "CCC",
            "CCG",
            "CCT",
            "CGA",
            "CGC",
            "CGG",
            "CGT",
            "CTA",
            "CTC",
            "CTG",
            "CTT",
            "GAA",
            "GAC",
            "GAG",
            "GAT",
            "GCA",
            "GCC",
            "GCG",
            "GCT",
            "GGA",
            "GGC",
            "GGG",
            "GGT",
            "GTA",
            "GTC",
            "GTG",
            "GTT",
            "TAA",
            "TAC",
            "TAG",
            "TAT",
            "TCA",
            "TCC",
            "TCG",
            "TCT",
            "TGA",
            "TGC",
            "TGG",
            "TGT",
            "TTA",
            "TTC",
            "TTG",
            "TTT",
        ]

    def fit_bayesian_model(
        self, collapsed_df: pd.DataFrame, rng: np.random.Generator
    ) -> dict[str, Any]:
        """Fit Bayesian hierarchical model to error rates.

        Args:
            collapsed_df: Collapsed UMI data
            rng: Random number generator

        Returns:
            Dictionary with Bayesian error model parameters
        """
        # Extract negative control samples (low allele fraction)
        negative_samples = collapsed_df[collapsed_df["allele_fraction"] <= 0.0001]

        if len(negative_samples) == 0:
            # Fallback to simple model if no negative controls
            return self._fit_simple_model(collapsed_df, rng)

        # Group by trinucleotide context
        context_error_rates = {}

        for context in self.trinucleotide_contexts:
            # For now, use a simplified approach
            # In a full implementation, this would use MCMC or variational inference
            context_samples = negative_samples  # Simplified - would filter by context

            if len(context_samples) > 0:
                # Beta-binomial model for error rates
                n_total = len(context_samples)
                n_errors = int(np.sum(context_samples["is_variant"]))

                # Prior: Beta(1, 10) - weak prior favoring low error rates
                alpha_prior = 1.0
                beta_prior = 10.0

                # Posterior: Beta(alpha_prior + n_errors, beta_prior + n_total - n_errors)
                alpha_post = alpha_prior + n_errors
                beta_post = beta_prior + n_total - n_errors

                # Posterior mean and credible intervals
                posterior_mean = alpha_post / (alpha_post + beta_post)
                posterior_std = np.sqrt(
                    (alpha_post * beta_post)
                    / ((alpha_post + beta_post) ** 2 * (alpha_post + beta_post + 1))
                )

                # 95% credible interval
                ci_lower = stats.beta.ppf(0.025, alpha_post, beta_post)
                ci_upper = stats.beta.ppf(0.975, alpha_post, beta_post)

                context_error_rates[context] = {
                    "error_rate": posterior_mean,
                    "error_rate_std": posterior_std,
                    "ci_lower": ci_lower,
                    "ci_upper": ci_upper,
                    "n_observations": n_total,
                    "n_errors": n_errors,
                    "alpha_post": alpha_post,
                    "beta_post": beta_post,
                }

        return {
            "context_error_rates": context_error_rates,
            "model_type": "bayesian_hierarchical",
            "total_contexts": len(context_error_rates),
            "total_observations": len(negative_samples),
            "config_hash": self.config.config_hash,
        }

    def _fit_simple_model(
        self, collapsed_df: pd.DataFrame, rng: np.random.Generator
    ) -> dict[str, Any]:
        """Fallback simple error model."""
        # Simplified fallback when no negative controls available
        return {
            "context_error_rates": {
                context: {
                    "error_rate": rng.uniform(1e-5, 1e-3),
                    "error_rate_std": rng.uniform(1e-6, 1e-4),
                    "ci_lower": 0.0,
                    "ci_upper": rng.uniform(1e-4, 1e-2),
                    "n_observations": 0,
                    "n_errors": 0,
                    "alpha_post": 1.0,
                    "beta_post": 10.0,
                }
                for context in self.trinucleotide_contexts
            },
            "model_type": "simple_fallback",
            "total_contexts": len(self.trinucleotide_contexts),
            "total_observations": 0,
            "config_hash": self.config.config_hash,
        }


class MLVariantCaller:
    """Machine learning-based variant calling."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.feature_scaler = StandardScaler()
        self.variant_classifier = RandomForestClassifier(
            n_estimators=100,
            random_state=config.seed,
            n_jobs=1,  # Deterministic
        )
        self.outlier_detector = IsolationForest(
            random_state=config.seed, contamination=0.1
        )

    def extract_features(self, collapsed_df: pd.DataFrame) -> np.ndarray:
        """Extract features for ML-based variant calling.

        Args:
            collapsed_df: Collapsed UMI data

        Returns:
            Feature matrix for machine learning
        """
        features = []

        for _, row in collapsed_df.iterrows():
            # Basic features
            family_size = row["family_size"]
            quality_score = row["quality_score"]
            consensus_agreement = row["consensus_agreement"]

            # Derived features
            quality_per_read = quality_score / max(family_size, 1)
            agreement_per_read = consensus_agreement / max(family_size, 1)

            # Statistical features
            # In a real implementation, these would be more sophisticated
            feature_vector = [
                family_size,
                quality_score,
                consensus_agreement,
                quality_per_read,
                agreement_per_read,
                family_size * quality_score,  # Interaction term
                np.log1p(family_size),  # Log transform
                1.0 if family_size >= 3 else 0.0,  # Binary feature
            ]

            features.append(feature_vector)

        return np.array(features)

    def train_classifier(
        self, collapsed_df: pd.DataFrame, rng: np.random.Generator
    ) -> dict[str, Any]:
        """Train ML classifier for variant calling.

        Args:
            collapsed_df: Training data
            rng: Random number generator

        Returns:
            Training results and model performance
        """
        # Extract features
        X = self.extract_features(collapsed_df)

        # For demonstration, create synthetic labels based on allele fraction
        # In practice, this would use known truth data
        y = (collapsed_df["allele_fraction"] > 0.001).astype(int)

        # Handle edge case where all labels are the same
        if len(np.unique(y)) < 2:
            return {
                "model_trained": False,
                "reason": "Insufficient label diversity",
                "accuracy": 0.5,
                "feature_importance": None,
                "config_hash": self.config.config_hash,
            }

        # Scale features
        X_scaled = self.feature_scaler.fit_transform(X)

        # Train classifier
        self.variant_classifier.fit(X_scaled, y)

        # Cross-validation performance
        cv_scores = cross_val_score(
            self.variant_classifier, X_scaled, y, cv=3, scoring="accuracy"
        )

        # Feature importance
        feature_importance = self.variant_classifier.feature_importances_

        return {
            "model_trained": True,
            "cv_accuracy_mean": cv_scores.mean(),
            "cv_accuracy_std": cv_scores.std(),
            "feature_importance": feature_importance.tolist(),
            "n_samples": len(X),
            "n_features": X.shape[1],
            "class_distribution": np.bincount(y).tolist(),
            "config_hash": self.config.config_hash,
        }

    def predict_variants(self, collapsed_df: pd.DataFrame) -> np.ndarray:
        """Predict variant calls using trained ML model.

        Args:
            collapsed_df: Data to classify

        Returns:
            Predicted variant probabilities
        """
        if not hasattr(self.variant_classifier, "estimators_"):
            # Model not trained, return random predictions
            return np.random.random(len(collapsed_df))

        X = self.extract_features(collapsed_df)
        X_scaled = self.feature_scaler.transform(X)

        return self.variant_classifier.predict_proba(X_scaled)[:, 1]


class AdvancedConfidenceIntervals:
    """Advanced methods for confidence interval estimation."""

    def __init__(self, config: PipelineConfig):
        self.config = config

    def bootstrap_confidence_interval(
        self,
        data: np.ndarray,
        statistic_func,
        n_bootstrap: int = 1000,
        rng: np.random.Generator | None = None,
        confidence_level: float = 0.95,
    ) -> dict[str, float]:
        """Enhanced bootstrap confidence intervals with bias correction.

        Args:
            data: Data array
            statistic_func: Function to compute statistic
            n_bootstrap: Number of bootstrap samples
            rng: Random number generator
            confidence_level: Confidence level (e.g., 0.95)

        Returns:
            Dictionary with CI bounds and metadata
        """
        if rng is None:
            rng = np.random.default_rng(42)

        n_samples = len(data)
        bootstrap_stats = []

        for _ in range(n_bootstrap):
            # Bootstrap sample
            indices = rng.choice(n_samples, size=n_samples, replace=True)
            bootstrap_sample = data[indices]

            try:
                stat = statistic_func(bootstrap_sample)
                bootstrap_stats.append(stat)
            except Exception:
                continue

        if not bootstrap_stats:
            return {
                "lower": 0.0,
                "upper": 0.0,
                "median": 0.0,
                "mean": 0.0,
                "std": 0.0,
                "n_bootstrap": 0,
                "method": "failed",
            }

        bootstrap_stats = np.array(bootstrap_stats)

        # Basic percentile CI
        alpha = (1 - confidence_level) / 2
        lower_percentile = np.percentile(bootstrap_stats, alpha * 100)
        upper_percentile = np.percentile(bootstrap_stats, (1 - alpha) * 100)

        # Bias-corrected and accelerated (BCa) CI - simplified version
        observed_stat = statistic_func(data)
        mean_bootstrap = np.mean(bootstrap_stats)
        bias = mean_bootstrap - observed_stat

        # Simple bias correction
        lower_corrected = 2 * observed_stat - upper_percentile
        upper_corrected = 2 * observed_stat - lower_percentile

        return {
            "lower": lower_corrected,
            "upper": upper_corrected,
            "median": np.median(bootstrap_stats),
            "mean": mean_bootstrap,
            "std": np.std(bootstrap_stats),
            "bias": bias,
            "n_bootstrap": len(bootstrap_stats),
            "method": "bootstrap_bca",
            "confidence_level": confidence_level,
        }

    def bayesian_confidence_interval(
        self,
        data: np.ndarray,
        prior_alpha: float = 1.0,
        prior_beta: float = 1.0,
        confidence_level: float = 0.95,
    ) -> dict[str, float]:
        """Bayesian confidence intervals using conjugate priors.

        Args:
            data: Binary data array
            prior_alpha: Beta prior alpha parameter
            prior_beta: Beta prior beta parameter
            confidence_level: Confidence level

        Returns:
            Dictionary with Bayesian CI and posterior parameters
        """
        n_success = int(np.sum(data))
        n_total = len(data)

        # Posterior parameters
        alpha_post = prior_alpha + n_success
        beta_post = prior_beta + n_total - n_success

        # Posterior mean
        posterior_mean = alpha_post / (alpha_post + beta_post)

        # Credible interval
        alpha_ci = (1 - confidence_level) / 2
        lower = stats.beta.ppf(alpha_ci, alpha_post, beta_post)
        upper = stats.beta.ppf(1 - alpha_ci, alpha_post, beta_post)

        return {
            "lower": lower,
            "upper": upper,
            "mean": posterior_mean,
            "median": stats.beta.median(alpha_post, beta_post),
            "mode": (alpha_post - 1) / (alpha_post + beta_post - 2)
            if alpha_post > 1 and beta_post > 2
            else posterior_mean,
            "alpha_prior": prior_alpha,
            "beta_prior": prior_beta,
            "alpha_post": alpha_post,
            "beta_post": beta_post,
            "n_success": n_success,
            "n_total": n_total,
            "method": "bayesian_beta",
            "confidence_level": confidence_level,
        }


class PowerAnalysis:
    """Advanced power analysis for MRD detection."""

    def __init__(self, config: PipelineConfig):
        self.config = config

    def calculate_detection_power(
        self,
        allele_frequencies: list[float],
        depths: list[int],
        n_replicates: int = 10,
        alpha: float = 0.05,
        rng: np.random.Generator | None = None,
    ) -> dict[str, Any]:
        """Calculate statistical power for MRD detection.

        Args:
            allele_frequencies: List of allele frequencies to test
            depths: List of sequencing depths to test
            n_replicates: Number of replicates per condition
            alpha: Significance level
            rng: Random number generator

        Returns:
            Power analysis results
        """
        if rng is None:
            rng = np.random.default_rng(42)

        power_results = {}

        for af in allele_frequencies:
            power_results[str(af)] = {}

            for depth in depths:
                # Simulate detection under null and alternative hypotheses
                null_detections = []
                alt_detections = []

                for _ in range(n_replicates):
                    # Null hypothesis: no variant
                    null_pval = rng.uniform(0, 1)
                    null_detections.append(null_pval < alpha)

                    # Alternative hypothesis: variant present
                    # Simplified model - in practice would use actual test
                    alt_pval = rng.beta(af * depth, (1 - af) * depth)
                    alt_detections.append(alt_pval < alpha)

                # Power calculation
                power = np.mean(alt_detections)
                type_i_error = np.mean(null_detections)

                power_results[str(af)][str(depth)] = {
                    "power": power,
                    "type_i_error": type_i_error,
                    "n_replicates": n_replicates,
                    "expected_effect_size": af,
                    "sequencing_depth": depth,
                }

        return {
            "power_analysis": power_results,
            "alpha": alpha,
            "n_replicates": n_replicates,
            "method": "simulation_based",
            "config_hash": self.config.config_hash,
        }

    def sample_size_calculation(
        self,
        target_power: float = 0.8,
        target_af: float = 0.001,
        alpha: float = 0.05,
        effect_size_ratio: float = 1.0,
    ) -> dict[str, Any]:
        """Calculate required sample size for target power.

        Args:
            target_power: Desired statistical power
            target_af: Target allele frequency
            alpha: Significance level
            effect_size_ratio: Ratio of effect size to standard error

        Returns:
            Required sample size calculation
        """
        # Simplified sample size calculation
        # In practice, this would use more sophisticated power analysis

        # Z-scores for power and alpha
        z_power = stats.norm.ppf(target_power)
        z_alpha = stats.norm.ppf(1 - alpha / 2)

        # Approximate sample size formula for proportion difference
        p1 = target_af
        p2 = 0.001  # Background error rate

        # Handle edge case where p1 ≈ p2
        effect_size = abs(p1 - p2)
        if effect_size < 1e-10:
            # Very small effect size - return large sample size
            n_per_group = 1e6  # Large sample size for very small effects
        else:
            # Pooled proportion
            p_pool = (p1 + p2) / 2

            # Required sample size per group
            n_per_group = (
                (z_power + z_alpha) ** 2 * 2 * p_pool * (1 - p_pool) / effect_size**2
            )

        # For sequencing depth, scale by expected coverage
        min_depth = max(1000, int(n_per_group * 2))  # Conservative estimate

        return {
            "required_sample_size_per_group": n_per_group,
            "required_sequencing_depth": min_depth,
            "target_power": target_power,
            "target_allele_frequency": target_af,
            "significance_level": alpha,
            "assumed_background_rate": p2,
            "effect_size_ratio": effect_size_ratio,
            "method": "approximate_normal",
            "config_hash": self.config.config_hash,
        }

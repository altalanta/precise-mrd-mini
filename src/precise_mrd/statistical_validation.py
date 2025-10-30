"""Advanced statistical validation and uncertainty quantification for MRD analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
from typing import Dict, List, Tuple, Optional, Any, Callable
import warnings

from .config import PipelineConfig
from .performance import IntelligentCache, CacheStrategy


class CrossValidator:
    """Advanced cross-validation for MRD detection models."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.cache = IntelligentCache(memory_limit_mb=200, disk_cache_dir=".cv_cache")

    def k_fold_cross_validation(self,
                               X: np.ndarray,
                               y: np.ndarray,
                               model_func: Callable,
                               k_folds: int = 5,
                               scoring: str = 'roc_auc',
                               stratified: bool = True) -> Dict[str, Any]:
        """Perform k-fold cross-validation with comprehensive metrics.

        Args:
            X: Feature matrix
            y: Target labels
            model_func: Function that returns a fitted model
            k_folds: Number of folds
            scoring: Scoring metric ('roc_auc', 'average_precision', 'accuracy')
            stratified: Whether to use stratified k-fold

        Returns:
            Cross-validation results with detailed statistics
        """
        cache_key = f"cv_{k_folds}_{scoring}_{stratified}_{len(X)}_{X.shape[1] if len(X.shape) > 1 else 0}"

        cached_result = self.cache.get(cache_key)
        if cached_result is not None:
            return cached_result

        # Choose cross-validation strategy
        if stratified and len(np.unique(y)) > 1:
            cv = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=self.config.seed)
        else:
            cv = KFold(n_splits=k_folds, shuffle=True, random_state=self.config.seed)

        # Perform cross-validation
        scores = []
        fold_results = []

        for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Fit model
            model = model_func(X_train, y_train)

            # Make predictions
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_test)[:, 1]
            else:
                y_pred_proba = model.decision_function(X_test)

            # Calculate metrics for this fold
            fold_metrics = self._calculate_fold_metrics(y_test, y_pred_proba, scoring)

            scores.append(fold_metrics['score'])
            fold_results.append({
                'fold': fold_idx,
                'train_size': len(X_train),
                'test_size': len(X_test),
                'metrics': fold_metrics
            })

        # Calculate comprehensive statistics
        scores = np.array(scores)
        results = {
            'mean_score': np.mean(scores),
            'std_score': np.std(scores),
            'scores': scores.tolist(),
            'confidence_interval': self._bootstrap_ci(scores),
            'fold_results': fold_results,
            'k_folds': k_folds,
            'scoring_metric': scoring,
            'total_samples': len(X),
            'n_features': X.shape[1] if len(X.shape) > 1 else 1,
            'class_distribution': np.bincount(y).tolist() if len(y) > 0 else [],
            'config_hash': self.config.config_hash
        }

        self.cache.put(cache_key, results, CacheStrategy.MEMORY, ttl=3600)
        return results

    def _calculate_fold_metrics(self, y_true: np.ndarray, y_pred_proba: np.ndarray, scoring: str) -> Dict[str, float]:
        """Calculate metrics for a single fold."""
        if scoring == 'roc_auc':
            score = roc_auc_score(y_true, y_pred_proba)
        elif scoring == 'average_precision':
            score = average_precision_score(y_true, y_pred_proba)
        elif scoring == 'accuracy':
            y_pred = (y_pred_proba > 0.5).astype(int)
            score = np.mean(y_true == y_pred)
        else:
            raise ValueError(f"Unknown scoring metric: {scoring}")

        return {
            'score': score,
            'n_samples': len(y_true),
            'positive_rate': np.mean(y_true)
        }

    def _bootstrap_ci(self, scores: np.ndarray, confidence: float = 0.95) -> Tuple[float, float]:
        """Calculate bootstrap confidence interval for CV scores."""
        n_bootstrap = 1000
        bootstrap_scores = []

        for _ in range(n_bootstrap):
            indices = np.random.choice(len(scores), size=len(scores), replace=True)
            bootstrap_scores.append(np.mean(scores[indices]))

        bootstrap_scores = np.array(bootstrap_scores)
        alpha = (1 - confidence) / 2

        return (
            np.percentile(bootstrap_scores, alpha * 100),
            np.percentile(bootstrap_scores, (1 - alpha) * 100)
        )


class UncertaintyQuantifier:
    """Advanced uncertainty quantification methods."""

    def __init__(self, config: PipelineConfig):
        self.config = config

    def bayesian_uncertainty(self,
                           posterior_samples: List[np.ndarray],
                           credible_interval: float = 0.95) -> Dict[str, Any]:
        """Quantify uncertainty using Bayesian credible intervals.

        Args:
            posterior_samples: List of posterior samples from MCMC
            credible_interval: Width of credible interval (e.g., 0.95)

        Returns:
            Uncertainty quantification results
        """
        if not posterior_samples:
            return {'error': 'No posterior samples provided'}

        # Stack samples for analysis
        samples_array = np.array(posterior_samples)

        # Calculate credible intervals
        alpha = (1 - credible_interval) / 2
        lower_percentile = np.percentile(samples_array, alpha * 100, axis=0)
        upper_percentile = np.percentile(samples_array, (1 - alpha) * 100, axis=0)
        median = np.median(samples_array, axis=0)
        mean = np.mean(samples_array, axis=0)

        # Calculate highest posterior density interval (HPDI)
        hpdi_lower, hpdi_upper = self._hpdi(samples_array, credible_interval)

        return {
            'mean': mean.tolist(),
            'median': median.tolist(),
            'credible_interval_lower': lower_percentile.tolist(),
            'credible_interval_upper': upper_percentile.tolist(),
            'hpdi_lower': hpdi_lower.tolist(),
            'hpdi_upper': hpdi_upper.tolist(),
            'credible_interval_width': credible_interval,
            'n_samples': len(posterior_samples),
            'n_parameters': samples_array.shape[1] if len(samples_array.shape) > 1 else 1,
            'method': 'bayesian_credible_intervals'
        }

    def _hpdi(self, samples: np.ndarray, credible_interval: float) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate highest posterior density interval."""
        # Simplified HPDI calculation
        # In practice, this would use more sophisticated methods
        n_samples = samples.shape[0]
        n_to_include = int(np.ceil(credible_interval * n_samples))

        # For each parameter, find the shortest interval containing n_to_include samples
        hpdi_lower = []
        hpdi_upper = []

        for param_idx in range(samples.shape[1]):
            param_samples = samples[:, param_idx]
            sorted_samples = np.sort(param_samples)

            # Find the shortest interval
            min_width = np.inf
            best_start = 0

            for start in range(n_samples - n_to_include + 1):
                end = start + n_to_include
                width = sorted_samples[end - 1] - sorted_samples[start]
                if width < min_width:
                    min_width = width
                    best_start = start

            hpdi_lower.append(sorted_samples[best_start])
            hpdi_upper.append(sorted_samples[best_start + n_to_include - 1])

        return np.array(hpdi_lower), np.array(hpdi_upper)

    def conformal_prediction(self,
                           X_train: np.ndarray,
                           y_train: np.ndarray,
                           X_test: np.ndarray,
                           model_func: Callable,
                           confidence_level: float = 0.95) -> Dict[str, Any]:
        """Apply conformal prediction for uncertainty quantification.

        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            model_func: Function that returns a fitted model
            confidence_level: Desired confidence level

        Returns:
            Conformal prediction results
        """
        # Fit model on training data
        model = model_func(X_train, y_train)

        # Get predictions on training data for calibration
        if hasattr(model, 'predict_proba'):
            train_pred_proba = model.predict_proba(X_train)[:, 1]
        else:
            train_pred_proba = model.decision_function(X_train)

        # Calculate conformity scores (absolute residuals)
        train_scores = np.abs(y_train - train_pred_proba)

        # Get predictions on test data
        if hasattr(model, 'predict_proba'):
            test_pred_proba = model.predict_proba(X_test)[:, 1]
        else:
            test_pred_proba = model.decision_function(X_test)

        # Calculate prediction intervals
        alpha = 1 - confidence_level
        q_hat = np.quantile(train_scores, 1 - alpha)

        prediction_intervals = []
        for pred in test_pred_proba:
            lower = max(0, pred - q_hat)
            upper = min(1, pred + q_hat)
            prediction_intervals.append((lower, upper))

        return {
            'test_predictions': test_pred_proba.tolist(),
            'prediction_intervals': prediction_intervals,
            'confidence_level': confidence_level,
            'calibration_quantile': q_hat,
            'n_train_samples': len(X_train),
            'n_test_samples': len(X_test),
            'method': 'conformal_prediction'
        }


class StatisticalTester:
    """Advanced statistical testing with multiple comparison corrections."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.alpha = config.stats.alpha

    def multiple_testing_correction(self,
                                  p_values: np.ndarray,
                                  method: str = 'benjamini_hochberg') -> Dict[str, Any]:
        """Apply multiple testing correction to p-values.

        Args:
            p_values: Array of p-values
            method: Correction method ('bonferroni', 'benjamini_hochberg', 'holm')

        Returns:
            Multiple testing correction results
        """
        p_values = np.asarray(p_values, dtype=float)
        m = len(p_values)

        if method == 'bonferroni':
            # Bonferroni correction
            adjusted_p = np.minimum(p_values * m, 1.0)
            rejected = adjusted_p < self.alpha

        elif method == 'benjamini_hochberg':
            # Benjamini-Hochberg FDR correction
            sorted_indices = np.argsort(p_values)
            sorted_p = p_values[sorted_indices]

            # BH procedure
            rejected = np.zeros(m, dtype=bool)
            for i in range(m - 1, -1, -1):
                if sorted_p[i] <= (i + 1) / m * self.alpha:
                    rejected[sorted_indices[:i + 1]] = True
                    break

            # Adjusted p-values
            adjusted_p = np.zeros_like(sorted_p)
            for i in range(m):
                adjusted_p[i] = sorted_p[i] * m / (i + 1)

            # Ensure monotonicity
            for i in range(m - 2, -1, -1):
                adjusted_p[i] = min(adjusted_p[i], adjusted_p[i + 1])

            adjusted_p = np.clip(adjusted_p, 0, 1)
            final_adjusted = np.zeros_like(p_values)
            final_adjusted[sorted_indices] = adjusted_p

        elif method == 'holm':
            # Holm-Bonferroni method
            sorted_indices = np.argsort(p_values)
            sorted_p = p_values[sorted_indices]

            adjusted_p = np.zeros_like(sorted_p)
            rejected = np.zeros(m, dtype=bool)

            for i in range(m):
                adjusted_p[i] = sorted_p[i] * (m - i)
                if adjusted_p[i] < self.alpha:
                    rejected[sorted_indices[i]] = True
                else:
                    break

            # Unadjusted p-values for non-rejected tests
            for i in range(len(adjusted_p)):
                if not rejected[sorted_indices[i]]:
                    adjusted_p[i] = sorted_p[i]

            final_adjusted = np.zeros_like(p_values)
            final_adjusted[sorted_indices] = adjusted_p

        else:
            raise ValueError(f"Unknown correction method: {method}")

        return {
            'original_p_values': p_values.tolist(),
            'adjusted_p_values': final_adjusted.tolist(),
            'rejected': rejected.tolist(),
            'n_tests': m,
            'correction_method': method,
            'alpha': self.alpha,
            'n_rejected': int(np.sum(rejected))
        }

    def power_analysis(self,
                      effect_sizes: List[float],
                      sample_sizes: List[int],
                      alpha: float = 0.05,
                      n_simulations: int = 1000) -> Dict[str, Any]:
        """Statistical power analysis for different effect sizes and sample sizes.

        Args:
            effect_sizes: List of effect sizes to test
            sample_sizes: List of sample sizes to test
            alpha: Significance level
            n_simulations: Number of simulations per condition

        Returns:
            Power analysis results
        """
        rng = np.random.default_rng(self.config.seed)
        power_results = {}

        for effect_size in effect_sizes:
            power_results[str(effect_size)] = {}

            for sample_size in sample_sizes:
                # Simulate null and alternative distributions
                null_rejections = 0
                alt_rejections = 0

                for _ in range(n_simulations):
                    # Null hypothesis: no effect
                    null_p = rng.uniform(0, 1)
                    if null_p < alpha:
                        null_rejections += 1

                    # Alternative hypothesis: effect present
                    # Simplified model - in practice would use proper test statistic
                    alt_statistic = rng.normal(effect_size, 1)  # Effect + noise
                    alt_p = 2 * (1 - stats.norm.cdf(abs(alt_statistic)))
                    if alt_p < alpha:
                        alt_rejections += 1

                power = alt_rejections / n_simulations
                type_i_error = null_rejections / n_simulations

                power_results[str(effect_size)][str(sample_size)] = {
                    'power': power,
                    'type_i_error': type_i_error,
                    'n_simulations': n_simulations,
                    'effect_size': effect_size,
                    'sample_size': sample_size,
                    'alpha': alpha
                }

        return {
            'power_analysis': power_results,
            'parameters': {
                'effect_sizes': effect_sizes,
                'sample_sizes': sample_sizes,
                'alpha': alpha,
                'n_simulations': n_simulations
            },
            'method': 'simulation_based_power_analysis'
        }


class RobustnessAnalyzer:
    """Analyze robustness of MRD detection to various perturbations."""

    def __init__(self, config: PipelineConfig):
        self.config = config

    def sensitivity_analysis(self,
                           collapsed_df: pd.DataFrame,
                           perturbation_types: List[str] = None,
                           perturbation_magnitudes: List[float] = None) -> Dict[str, Any]:
        """Perform sensitivity analysis on model parameters.

        Args:
            collapsed_df: Collapsed UMI data
            perturbation_types: Types of perturbations to test
            perturbation_magnitudes: Magnitudes of perturbations

        Returns:
            Sensitivity analysis results
        """
        if perturbation_types is None:
            perturbation_types = ['family_size', 'quality_score', 'background_rate']

        if perturbation_magnitudes is None:
            perturbation_magnitudes = [0.8, 0.9, 1.1, 1.2]  # 20% perturbations

        sensitivity_results = {}

        for pert_type in perturbation_types:
            sensitivity_results[pert_type] = {}

            for magnitude in perturbation_magnitudes:
                # Apply perturbation
                perturbed_df = self._apply_perturbation(collapsed_df.copy(), pert_type, magnitude)

                # Calculate metrics on perturbed data
                # This would integrate with the main pipeline
                # For now, calculate basic statistics
                sensitivity_results[pert_type][str(magnitude)] = {
                    'perturbation_type': pert_type,
                    'magnitude': magnitude,
                    'n_samples': len(perturbed_df),
                    'mean_family_size': perturbed_df['family_size'].mean(),
                    'mean_quality': perturbed_df['quality_score'].mean(),
                    'variant_rate': perturbed_df['is_variant'].mean()
                }

        return {
            'sensitivity_analysis': sensitivity_results,
            'perturbation_types': perturbation_types,
            'perturbation_magnitudes': perturbation_magnitudes,
            'method': 'parameter_sensitivity_analysis'
        }

    def _apply_perturbation(self, df: pd.DataFrame, pert_type: str, magnitude: float) -> pd.DataFrame:
        """Apply perturbation to specified parameter."""
        if pert_type == 'family_size':
            df['family_size'] = (df['family_size'] * magnitude).astype(int)
            df['family_size'] = np.clip(df['family_size'], 1, 1000)
        elif pert_type == 'quality_score':
            df['quality_score'] = df['quality_score'] * magnitude
            df['quality_score'] = np.clip(df['quality_score'], 0, 40)
        elif pert_type == 'background_rate':
            df['background_rate'] = df['background_rate'] * magnitude
            df['background_rate'] = np.clip(df['background_rate'], 1e-6, 1e-2)

        return df

    def bootstrap_robustness(self,
                           collapsed_df: pd.DataFrame,
                           n_bootstrap: int = 100,
                           metrics_func: Optional[Callable] = None) -> Dict[str, Any]:
        """Assess robustness using bootstrap resampling.

        Args:
            collapsed_df: Collapsed UMI data
            n_bootstrap: Number of bootstrap samples
            metrics_func: Function to calculate metrics on bootstrap samples

        Returns:
            Bootstrap robustness analysis
        """
        if metrics_func is None:
            def default_metrics(df):
                return {
                    'variant_rate': df['is_variant'].mean(),
                    'mean_family_size': df['family_size'].mean(),
                    'mean_quality': df['quality_score'].mean()
                }
            metrics_func = default_metrics

        rng = np.random.default_rng(self.config.seed)
        bootstrap_results = []

        for _ in range(n_bootstrap):
            # Bootstrap sample
            indices = rng.choice(len(collapsed_df), size=len(collapsed_df), replace=True)
            bootstrap_sample = collapsed_df.iloc[indices]

            # Calculate metrics
            metrics = metrics_func(bootstrap_sample)
            bootstrap_results.append(metrics)

        # Calculate statistics across bootstrap samples
        metrics_df = pd.DataFrame(bootstrap_results)

        robustness_stats = {}
        for column in metrics_df.columns:
            values = metrics_df[column].values
            robustness_stats[column] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'ci_lower': np.percentile(values, 2.5),
                'ci_upper': np.percentile(values, 97.5),
                'coefficient_of_variation': np.std(values) / abs(np.mean(values)) if np.mean(values) != 0 else 0
            }

        return {
            'robustness_statistics': robustness_stats,
            'n_bootstrap': n_bootstrap,
            'n_metrics': len(metrics_df.columns),
            'original_sample_size': len(collapsed_df),
            'method': 'bootstrap_robustness_analysis'
        }


class ModelValidator:
    """Comprehensive model validation and comparison."""

    def __init__(self, config: PipelineConfig):
        self.config = config

    def compare_models(self,
                     X: np.ndarray,
                     y: np.ndarray,
                     model_functions: Dict[str, Callable],
                     scoring: str = 'roc_auc') -> Dict[str, Any]:
        """Compare multiple models using cross-validation.

        Args:
            X: Feature matrix
            y: Target labels
            model_functions: Dictionary of model functions
            scoring: Scoring metric

        Returns:
            Model comparison results
        """
        cv_results = {}

        for model_name, model_func in model_functions.items():
            print(f"  Evaluating model: {model_name}")

            # Perform cross-validation
            cv = CrossValidator(self.config)
            results = cv.k_fold_cross_validation(X, y, model_func, k_folds=5, scoring=scoring)

            cv_results[model_name] = results

        # Statistical comparison between models
        model_names = list(cv_results.keys())
        if len(model_names) >= 2:
            # Compare model performances
            scores_dict = {name: results['scores'] for name, results in cv_results.items()}

            # Simple t-test for model comparison (could be enhanced)
            comparison_results = {}
            for i, name1 in enumerate(model_names):
                for name2 in model_names[i+1:]:
                    scores1 = np.array(scores_dict[name1])
                    scores2 = np.array(scores_dict[name2])

                    t_stat, p_value = stats.ttest_rel(scores1, scores2)

                    comparison_results[f"{name1}_vs_{name2}"] = {
                        't_statistic': t_stat,
                        'p_value': p_value,
                        'mean_score_1': np.mean(scores1),
                        'mean_score_2': np.mean(scores2),
                        'score_1_better': np.mean(scores1) > np.mean(scores2)
                    }

        return {
            'model_results': cv_results,
            'model_comparison': comparison_results if len(model_names) >= 2 else {},
            'best_model': max(cv_results.items(), key=lambda x: x[1]['mean_score'])[0] if cv_results else None,
            'scoring_metric': scoring,
            'n_models': len(model_functions)
        }

    def calibration_analysis(self,
                          y_true: np.ndarray,
                          y_pred_proba: np.ndarray,
                          n_bins: int = 10) -> Dict[str, Any]:
        """Analyze prediction calibration across probability bins.

        Args:
            y_true: True binary labels
            y_pred_proba: Predicted probabilities
            n_bins: Number of calibration bins

        Returns:
            Calibration analysis results
        """
        # Create calibration bins
        bins = np.linspace(0, 1, n_bins + 1)
        calibration_data = []

        for i in range(n_bins):
            bin_lower = bins[i]
            bin_upper = bins[i + 1]

            # Find predictions in this bin
            in_bin = (y_pred_proba >= bin_lower) & (y_pred_proba < bin_upper)

            if i == n_bins - 1:  # Last bin includes upper bound
                in_bin = (y_pred_proba >= bin_lower) & (y_pred_proba <= bin_upper)

            if np.sum(in_bin) == 0:
                calibration_data.append({
                    'bin': i,
                    'lower': bin_lower,
                    'upper': bin_upper,
                    'count': 0,
                    'event_rate': 0.0,
                    'confidence': 0.0,
                    'ece_contribution': 0.0
                })
                continue

            bin_count = np.sum(in_bin)
            bin_event_rate = np.mean(y_true[in_bin])
            bin_confidence = np.mean(y_pred_proba[in_bin])

            # Expected calibration error contribution
            ece_contribution = bin_count * abs(bin_event_rate - bin_confidence)

            calibration_data.append({
                'bin': i,
                'lower': bin_lower,
                'upper': bin_upper,
                'count': int(bin_count),
                'event_rate': float(bin_event_rate),
                'confidence': float(bin_confidence),
                'ece_contribution': float(ece_contribution)
            })

        # Calculate overall calibration metrics
        total_samples = len(y_true)
        ece = sum(item['ece_contribution'] for item in calibration_data) / total_samples

        # Maximum calibration error
        max_calibration_error = max(abs(item['event_rate'] - item['confidence'])
                                  for item in calibration_data if item['count'] > 0)

        return {
            'calibration_bins': calibration_data,
            'expected_calibration_error': ece,
            'max_calibration_error': max_calibration_error,
            'n_bins': n_bins,
            'total_samples': total_samples,
            'well_calibrated_threshold': 0.05,  # ECE < 5% considered well-calibrated
            'is_well_calibrated': ece < 0.05
        }









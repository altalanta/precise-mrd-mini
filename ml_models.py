"""Advanced machine learning models for MRD variant calling."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .config import PipelineConfig
from .performance import get_ml_performance_tracker, ml_performance_decorator

# ML model imports with fallbacks
try:
    import xgboost as xgb

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb

    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler


class EnhancedFeatureEngineer:
    """Enhanced automated feature engineering for variant calling."""

    def __init__(self, config: PipelineConfig):
        """Initialize enhanced feature engineer.

        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.feature_names = []

    def extract_comprehensive_features(
        self,
        collapsed_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, list[str]]:
        """Extract comprehensive features from collapsed UMI data.

        Args:
            collapsed_df: DataFrame with collapsed UMI data

        Returns:
            Tuple of (feature_matrix, feature_names)
        """
        features = []

        # Basic features
        basic_features = [
            "family_size",
            "quality_score",
            "consensus_agreement",
            "allele_fraction",
        ]

        for feature in basic_features:
            if feature in collapsed_df.columns:
                features.append(collapsed_df[feature].values.reshape(-1, 1))

        # Interaction features
        if (
            "family_size" in collapsed_df.columns
            and "quality_score" in collapsed_df.columns
        ):
            family_quality_ratio = collapsed_df["family_size"] / (
                collapsed_df["quality_score"] + 1
            )
            features.append(family_quality_ratio.values.reshape(-1, 1))

        if (
            "consensus_agreement" in collapsed_df.columns
            and "allele_fraction" in collapsed_df.columns
        ):
            confidence_af = (
                collapsed_df["consensus_agreement"] * collapsed_df["allele_fraction"]
            )
            features.append(confidence_af.values.reshape(-1, 1))

        # Statistical features
        if "family_size" in collapsed_df.columns:
            # Family size statistics
            family_sizes = collapsed_df["family_size"].values
            features.append(np.log1p(family_sizes).reshape(-1, 1))  # Log transform
            features.append(
                (family_sizes - np.mean(family_sizes))
                / (np.std(family_sizes) + 1e-8).reshape(-1, 1),
            )  # Z-score

        # Quality-based features
        if "quality_score" in collapsed_df.columns:
            quality_scores = collapsed_df["quality_score"].values
            features.append(
                np.sqrt(quality_scores).reshape(-1, 1),
            )  # Square root transform

        # Combine all features
        if features:
            X = np.hstack(features)
            self.feature_names = basic_features[: len(features)]

            # Add derived feature names
            derived_names = []
            if len(features) > len(basic_features):
                derived_names.extend(["family_quality_ratio", "confidence_af"])
            if len(features) > len(basic_features) + 2:
                derived_names.extend(["log_family_size", "zscore_family_size"])
            if len(features) > len(basic_features) + 3:
                derived_names.extend(["sqrt_quality"])

            self.feature_names.extend(derived_names)
        else:
            # Fallback if no features available
            X = np.zeros((len(collapsed_df), 1))
            self.feature_names = ["dummy_feature"]

        return pd.DataFrame(X, columns=self.feature_names), self.feature_names

    def select_optimal_features(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        method: str = "mutual_info",
        k: int = None,
    ) -> tuple[pd.DataFrame, list[str]]:
        """Select optimal features using multiple methods.

        Args:
            X: Feature matrix
            y: Target labels
            method: Feature selection method
            k: Number of features to select (None for automatic)

        Returns:
            Tuple of (selected_features, selected_feature_names)
        """
        if k is None:
            # Automatic selection based on data size
            k = min(max(5, X.shape[1] // 3), X.shape[1])

        if method == "f_classif":
            selector = SelectKBest(score_func=f_classif, k=k)
        elif method == "mutual_info":
            selector = SelectKBest(score_func=mutual_info_classif, k=k)
        else:
            raise ValueError(f"Unknown feature selection method: {method}")

        X_selected = selector.fit_transform(X, y)
        selected_features = X.columns[selector.get_support()].tolist()

        return pd.DataFrame(X_selected, columns=selected_features), selected_features


class GradientBoostedVariantCaller:
    """Gradient-boosted models for enhanced variant calling."""

    def __init__(self, config: PipelineConfig, model_type: str = "xgboost"):
        """Initialize gradient-boosted variant caller.

        Args:
            config: Pipeline configuration
            model_type: Type of gradient boosting model ('xgboost', 'lightgbm', 'sklearn_gbm')
        """
        self.config = config
        self.model_type = model_type
        self.model = None
        self.feature_engineer = EnhancedFeatureEngineer(config)
        self.scaler = StandardScaler()
        self.selected_features = []

    @ml_performance_decorator
    def train_model(
        self,
        collapsed_df: pd.DataFrame,
        rng: np.random.Generator,
    ) -> dict[str, Any]:
        """Train gradient-boosted model for variant calling.

        Args:
            collapsed_df: DataFrame with collapsed UMI data
            rng: Random number generator for reproducibility

        Returns:
            Dictionary with training results and performance metrics
        """
        print(f"  Training {self.model_type} model...")

        # Extract features and labels
        X, feature_names = self.feature_engineer.extract_comprehensive_features(
            collapsed_df,
        )

        # Use is_variant as target (or generate synthetic labels if not available)
        if "is_variant" in collapsed_df.columns:
            y = collapsed_df["is_variant"].values
        else:
            # Generate synthetic labels based on allele fraction and quality
            y = self._generate_synthetic_labels(collapsed_df, rng)

        # Handle class imbalance
        pos_ratio = np.mean(y)
        print(f"  Positive class ratio: {pos_ratio:.3f}")

        # Feature selection
        X_selected, selected_features = self.feature_engineer.select_optimal_features(
            X,
            y,
            method="mutual_info",
        )
        self.selected_features = selected_features

        print(
            f"  Selected {len(selected_features)} features: {selected_features[:5]}{'...' if len(selected_features) > 5 else ''}",
        )

        # Scale features
        X_scaled = self.scaler.fit_transform(X_selected)

        # Create model based on type
        if self.model_type == "xgboost" and XGBOOST_AVAILABLE:
            model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.config.seed,
                eval_metric="logloss",
            )
        elif self.model_type == "lightgbm" and LIGHTGBM_AVAILABLE:
            model = lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.config.seed,
                verbosity=-1,
            )
        else:
            # Fallback to sklearn GBM
            model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                random_state=self.config.seed,
            )

        # Cross-validation for model evaluation
        cv_scores = cross_val_score(
            model,
            X_scaled,
            y,
            cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=self.config.seed),
            scoring="roc_auc",
        )

        print(f"  CV AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

        # Train model on full data
        model.fit(X_scaled, y)
        self.model = model

        # Get predictions for evaluation
        predictions = model.predict_proba(X_scaled)[:, 1]

        # Calculate optimal threshold
        from .advanced_stats import AdaptiveThresholdOptimizer

        threshold_optimizer = AdaptiveThresholdOptimizer(self.config)
        optimal_threshold = threshold_optimizer.optimize_threshold(
            predictions,
            y,
            metric="f1",
        )

        # Record performance metrics
        tracker = get_ml_performance_tracker()
        model_metrics = {
            "roc_auc": cv_scores.mean(),
            "cv_std": cv_scores.std(),
            "n_features": len(selected_features),
            "positive_ratio": pos_ratio,
        }

        # Get additional metrics if ground truth is available
        if "is_variant" in collapsed_df.columns:
            try:
                from sklearn.metrics import average_precision_score, roc_auc_score

                model_metrics["test_roc_auc"] = roc_auc_score(y, predictions)
                model_metrics["test_avg_precision"] = average_precision_score(
                    y,
                    predictions,
                )
            except Exception:
                pass

        tracker.record_model_metrics(f"{self.model_type}_model", model_metrics)

        # Record feature importance
        importance = self.get_feature_importance()
        if importance:
            tracker.record_feature_importance(f"{self.model_type}_model", importance)

        # Record prediction distribution
        tracker.record_prediction_distribution(f"{self.model_type}_model", predictions)

        return {
            "model_type": self.model_type,
            "feature_names": selected_features,
            "cv_scores": cv_scores,
            "predictions": predictions,
            "optimal_threshold": optimal_threshold,
            "positive_ratio": pos_ratio,
            "n_features": len(selected_features),
            "model": model,
            "performance_metrics": model_metrics,
        }

    def predict_variants(self, collapsed_df: pd.DataFrame) -> np.ndarray:
        """Predict variants using trained gradient-boosted model.

        Args:
            collapsed_df: DataFrame with collapsed UMI data

        Returns:
            Array of variant probabilities
        """
        if self.model is None:
            raise ValueError("Model must be trained before prediction")

        # Extract features
        X, _ = self.feature_engineer.extract_comprehensive_features(collapsed_df)

        # Feature selection (use same features as training)
        X_selected = X[self.selected_features]

        # Scale features
        X_scaled = self.scaler.transform(X_selected)

        # Get predictions
        return self.model.predict_proba(X_scaled)[:, 1]

    def _generate_synthetic_labels(
        self,
        collapsed_df: pd.DataFrame,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Generate synthetic labels for training when ground truth is not available."""
        # Use allele fraction as primary signal
        af_signal = collapsed_df.get(
            "allele_fraction",
            pd.Series([0.001] * len(collapsed_df)),
        )

        # Add noise based on quality and consensus
        quality_factor = (
            collapsed_df.get("quality_score", pd.Series([25] * len(collapsed_df)))
            / 50.0
        )
        consensus_factor = collapsed_df.get(
            "consensus_agreement",
            pd.Series([0.8] * len(collapsed_df)),
        )

        # Combine signals with more sophisticated modeling
        combined_signal = af_signal * quality_factor * consensus_factor

        # Add interaction terms
        if "family_size" in collapsed_df.columns:
            family_factor = (
                np.log1p(collapsed_df["family_size"]) / 5.0
            )  # Log transform and scale
            combined_signal *= 1 + family_factor

        # Convert to probabilities with noise
        probabilities = np.clip(combined_signal * 50, 0, 1)  # Scale to reasonable range

        # Add realistic noise based on biological variability
        noise = rng.normal(0, 0.15, len(probabilities))
        noisy_probabilities = np.clip(probabilities + noise, 0, 1)

        # Convert to binary labels using adaptive threshold
        threshold = np.percentile(noisy_probabilities, 80)  # Top 20% as variants
        return (noisy_probabilities > threshold).astype(int)

    def get_feature_importance(self) -> dict[str, float]:
        """Get feature importance scores from trained model.

        Returns:
            Dictionary mapping feature names to importance scores
        """
        if self.model is None:
            return {}

        if hasattr(self.model, "feature_importances_"):
            return dict(
                zip(
                    self.selected_features,
                    self.model.feature_importances_,
                    strict=False,
                ),
            )
        elif hasattr(self.model, "coef_"):
            # For linear models
            return dict(zip(self.selected_features, self.model.coef_[0], strict=False))

        return {}


class EnsembleVariantCaller:
    """Ensemble of multiple ML models for robust variant calling."""

    def __init__(self, config: PipelineConfig):
        """Initialize ensemble variant caller.

        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.models = {}
        self.model_weights = {}
        self.feature_engineer = EnhancedFeatureEngineer(config)
        self.scaler = StandardScaler()
        self.selected_features = []

        # Initialize ensemble components
        self._initialize_ensemble()

    def _initialize_ensemble(self):
        """Initialize ensemble of different model types."""
        if XGBOOST_AVAILABLE:
            self.models["xgboost"] = GradientBoostedVariantCaller(
                self.config,
                "xgboost",
            )
        if LIGHTGBM_AVAILABLE:
            self.models["lightgbm"] = GradientBoostedVariantCaller(
                self.config,
                "lightgbm",
            )

        # Always include sklearn models as fallback
        self.models["gbm"] = GradientBoostedVariantCaller(self.config, "sklearn_gbm")
        self.models["rf"] = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=self.config.seed,
            n_jobs=1,
        )

    @ml_performance_decorator
    def train_ensemble(
        self,
        collapsed_df: pd.DataFrame,
        rng: np.random.Generator,
    ) -> dict[str, Any]:
        """Train ensemble of ML models.

        Args:
            collapsed_df: DataFrame with collapsed UMI data
            rng: Random number generator for reproducibility

        Returns:
            Dictionary with ensemble training results
        """
        print("  Training ensemble of ML models...")

        # Extract features once for all models
        X, feature_names = self.feature_engineer.extract_comprehensive_features(
            collapsed_df,
        )

        # Generate labels
        if "is_variant" in collapsed_df.columns:
            y = collapsed_df["is_variant"].values
        else:
            y = self.models["gbm"]._generate_synthetic_labels(collapsed_df, rng)

        pos_ratio = np.mean(y)
        print(f"  Positive class ratio: {pos_ratio:.3f}")

        # Feature selection
        X_selected, selected_features = self.feature_engineer.select_optimal_features(
            X,
            y,
        )
        self.selected_features = selected_features

        print(f"  Selected {len(selected_features)} features for ensemble")

        # Scale features
        X_scaled = self.scaler.fit_transform(X_selected)

        ensemble_results = {}
        model_performances = {}

        # Train each model and evaluate
        for model_name, model in self.models.items():
            print(f"    Training {model_name}...")

            if hasattr(model, "train_model"):
                # Gradient-boosted models
                results = model.train_model(collapsed_df, rng)
                cv_score = results["cv_scores"].mean()
            else:
                # Traditional sklearn models
                cv_scores = cross_val_score(
                    model,
                    X_scaled,
                    y,
                    cv=StratifiedKFold(
                        n_splits=3,
                        shuffle=True,
                        random_state=self.config.seed,
                    ),
                    scoring="roc_auc",
                )
                cv_score = cv_scores.mean()

                # Train model
                model.fit(X_scaled, y)

                results = {
                    "cv_scores": cv_scores,
                    "predictions": model.predict_proba(X_scaled)[:, 1]
                    if hasattr(model, "predict_proba")
                    else model.predict(X_scaled),
                }

            model_performances[model_name] = cv_score
            ensemble_results[model_name] = results

            print(f"    {model_name} CV AUC: {cv_score:.3f}")

        # Calculate model weights based on performance
        max_performance = max(model_performances.values())
        for model_name, performance in model_performances.items():
            # Weight by relative performance (avoid division by zero)
            self.model_weights[model_name] = (
                performance / max_performance if max_performance > 0 else 0.5
            )

        print(f"  Model weights: {self.model_weights}")

        # Generate ensemble predictions
        ensemble_predictions = np.zeros(len(y))
        for model_name, model in self.models.items():
            weight = self.model_weights[model_name]

            if hasattr(model, "predict_variants"):
                # Use pre-computed predictions if available
                predictions = ensemble_results[model_name]["predictions"]
            else:
                # Re-predict for sklearn models
                predictions = (
                    model.predict_proba(X_scaled)[:, 1]
                    if hasattr(model, "predict_proba")
                    else model.predict(X_scaled)
                )

            ensemble_predictions += weight * predictions

        # Normalize by total weight
        total_weight = sum(self.model_weights.values())
        if total_weight > 0:
            ensemble_predictions /= total_weight

        # Calculate optimal threshold for ensemble
        from .advanced_stats import AdaptiveThresholdOptimizer

        threshold_optimizer = AdaptiveThresholdOptimizer(self.config)
        optimal_threshold = threshold_optimizer.optimize_threshold(
            ensemble_predictions,
            y,
            metric="f1",
        )

        # Record ensemble performance metrics
        tracker = get_ml_performance_tracker()

        # Calculate ensemble metrics
        ensemble_metrics = {
            "roc_auc": np.mean(list(model_performances.values())),
            "ensemble_threshold": optimal_threshold,
            "n_models": len(self.models),
            "positive_ratio": pos_ratio,
            "n_features": len(selected_features),
        }

        # Get additional metrics if ground truth is available
        if "is_variant" in collapsed_df.columns:
            try:
                from sklearn.metrics import average_precision_score, roc_auc_score

                ensemble_metrics["test_roc_auc"] = roc_auc_score(
                    y,
                    ensemble_predictions,
                )
                ensemble_metrics["test_avg_precision"] = average_precision_score(
                    y,
                    ensemble_predictions,
                )
            except Exception:
                pass

        tracker.record_model_metrics("ensemble_model", ensemble_metrics)

        # Record ensemble feature importance
        ensemble_importance = self.get_feature_importance()
        if ensemble_importance:
            tracker.record_feature_importance("ensemble_model", ensemble_importance)

        # Record ensemble prediction distribution
        tracker.record_prediction_distribution("ensemble_model", ensemble_predictions)

        return {
            "model_performances": model_performances,
            "model_weights": self.model_weights,
            "ensemble_predictions": ensemble_predictions,
            "optimal_threshold": optimal_threshold,
            "positive_ratio": pos_ratio,
            "n_features": len(selected_features),
            "individual_results": ensemble_results,
            "performance_metrics": ensemble_metrics,
        }

    def predict_variants(self, collapsed_df: pd.DataFrame) -> np.ndarray:
        """Predict variants using trained ensemble.

        Args:
            collapsed_df: DataFrame with collapsed UMI data

        Returns:
            Array of variant probabilities
        """
        # Extract features
        X, _ = self.feature_engineer.extract_comprehensive_features(collapsed_df)
        X_selected = X[self.selected_features]
        X_scaled = self.scaler.transform(X_selected)

        # Get predictions from each model
        ensemble_predictions = np.zeros(X_scaled.shape[0])

        for model_name, model in self.models.items():
            weight = self.model_weights.get(model_name, 0.0)

            if weight > 0:
                if hasattr(model, "predict_variants"):
                    predictions = model.predict_variants(collapsed_df)
                else:
                    predictions = (
                        model.predict_proba(X_scaled)[:, 1]
                        if hasattr(model, "predict_proba")
                        else model.predict(X_scaled)
                    )

                ensemble_predictions += weight * predictions

        # Normalize by total weight
        total_weight = sum(self.model_weights.values())
        if total_weight > 0:
            ensemble_predictions /= total_weight

        return ensemble_predictions

    def get_feature_importance(self) -> dict[str, float]:
        """Get average feature importance across ensemble models.

        Returns:
            Dictionary mapping feature names to average importance scores
        """
        total_importance = {}

        for model_name, model in self.models.items():
            if hasattr(model, "get_feature_importance"):
                importance = model.get_feature_importance()
                weight = self.model_weights.get(model_name, 0.0)

                for feat_name, imp_score in importance.items():
                    total_importance[feat_name] = (
                        total_importance.get(feat_name, 0.0) + weight * imp_score
                    )

        # Normalize by total weight
        total_weight = sum(self.model_weights.values())
        if total_weight > 0:
            for feat_name in total_importance:
                total_importance[feat_name] /= total_weight

        return total_importance

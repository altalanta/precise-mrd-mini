"""Advanced statistical methods and machine learning for MRD variant calling."""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
import warnings

from .config import PipelineConfig


class FeatureEngineer:
    """Automated feature engineering for variant calling."""

    def __init__(self, config: PipelineConfig):
        """Initialize feature engineer.

        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.feature_names = []

    def extract_features(self, collapsed_df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Extract features from collapsed UMI data.

        Args:
            collapsed_df: DataFrame with collapsed UMI data

        Returns:
            Tuple of (feature_matrix, feature_names)
        """
        features = []

        # Basic features from existing data
        basic_features = [
            'family_size',
            'quality_score',
            'consensus_agreement',
            'allele_fraction'
        ]

        for feature in basic_features:
            if feature in collapsed_df.columns:
                features.append(collapsed_df[feature].values.reshape(-1, 1))

        # Derived features
        if 'family_size' in collapsed_df.columns and 'quality_score' in collapsed_df.columns:
            # Interaction features
            family_quality_ratio = collapsed_df['family_size'] / (collapsed_df['quality_score'] + 1)
            features.append(family_quality_ratio.values.reshape(-1, 1))

        if 'consensus_agreement' in collapsed_df.columns and 'allele_fraction' in collapsed_df.columns:
            # Confidence-weighted allele fraction
            confidence_af = collapsed_df['consensus_agreement'] * collapsed_df['allele_fraction']
            features.append(confidence_af.values.reshape(-1, 1))

        # Combine all features
        if features:
            X = np.hstack(features)
            self.feature_names = basic_features[:len(features)]

            # Add derived feature names
            if len(features) > len(basic_features):
                self.feature_names.extend(['family_quality_ratio', 'confidence_af'])

        else:
            # Fallback if no features available
            X = np.zeros((len(collapsed_df), 1))
            self.feature_names = ['dummy_feature']

        return pd.DataFrame(X, columns=self.feature_names), self.feature_names

    def select_features(self, X: pd.DataFrame, y: np.ndarray, method: str = 'mutual_info',
                       k: int = 10) -> Tuple[pd.DataFrame, List[str]]:
        """Select most informative features.

        Args:
            X: Feature matrix
            y: Target labels
            method: Feature selection method ('f_classif', 'mutual_info')
            k: Number of top features to select

        Returns:
            Tuple of (selected_features, selected_feature_names)
        """
        if method == 'f_classif':
            selector = SelectKBest(score_func=f_classif, k=min(k, X.shape[1]))
        elif method == 'mutual_info':
            selector = SelectKBest(score_func=mutual_info_classif, k=min(k, X.shape[1]))
        else:
            raise ValueError(f"Unknown feature selection method: {method}")

        X_selected = selector.fit_transform(X, y)
        selected_features = X.columns[selector.get_support()].tolist()

        return pd.DataFrame(X_selected, columns=selected_features), selected_features


class MLVariantCaller:
    """Machine learning-based variant caller for MRD detection."""

    def __init__(self, config: PipelineConfig, model_type: str = 'ensemble'):
        """Initialize ML variant caller.

        Args:
            config: Pipeline configuration
            model_type: Type of ML model ('rf', 'gbm', 'svm', 'lr', 'ensemble')
        """
        self.config = config
        self.model_type = model_type
        self.models = {}
        self.feature_engineer = FeatureEngineer(config)
        self.scaler = StandardScaler()

        # Initialize models
        self._initialize_models()

    def _initialize_models(self):
        """Initialize ML models based on model_type."""
        if self.model_type == 'rf':
            self.models['rf'] = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=self.config.seed,
                n_jobs=1  # For deterministic behavior
            )
        elif self.model_type == 'gbm':
            self.models['gbm'] = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=self.config.seed
            )
        elif self.model_type == 'svm':
            self.models['svm'] = SVC(
                probability=True,
                random_state=self.config.seed
            )
        elif self.model_type == 'lr':
            self.models['lr'] = LogisticRegression(
                random_state=self.config.seed,
                max_iter=1000
            )
        elif self.model_type == 'ensemble':
            # Ensemble of multiple models
            self.models['rf'] = RandomForestClassifier(
                n_estimators=50,
                max_depth=8,
                random_state=self.config.seed,
                n_jobs=1
            )
            self.models['gbm'] = GradientBoostingClassifier(
                n_estimators=50,
                max_depth=4,
                random_state=self.config.seed
            )
            self.models['lr'] = LogisticRegression(
                random_state=self.config.seed,
                max_iter=1000
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def train_classifier(self, collapsed_df: pd.DataFrame, rng: np.random.Generator) -> Dict[str, Any]:
        """Train the ML classifier on collapsed UMI data.

        Args:
            collapsed_df: DataFrame with collapsed UMI data
            rng: Random number generator for reproducibility

        Returns:
            Dictionary with training results and performance metrics
        """
        print(f"  Training {self.model_type} model...")

        # Extract features and labels
        X, feature_names = self.feature_engineer.extract_features(collapsed_df)

        # Use is_variant as target (or generate synthetic labels if not available)
        if 'is_variant' in collapsed_df.columns:
            y = collapsed_df['is_variant'].values
        else:
            # Generate synthetic labels based on allele fraction and quality
            # This is a fallback for when we don't have ground truth
            y = self._generate_synthetic_labels(collapsed_df, rng)

        # Handle class imbalance
        pos_ratio = np.mean(y)
        print(f"  Positive class ratio: {pos_ratio:.3f}")

        if pos_ratio < 0.01:  # Very imbalanced
            print("  Using class weighting due to imbalanced data")
            # Adjust for class imbalance
            class_weight = {0: 1.0, 1: 1.0 / pos_ratio if pos_ratio > 0 else 2.0}
        else:
            class_weight = None

        # Feature selection
        X_selected, selected_features = self.feature_engineer.select_features(
            X, y, method='mutual_info', k=min(10, X.shape[1])
        )

        print(f"  Selected {len(selected_features)} features: {selected_features}")

        # Scale features
        X_scaled = self.scaler.fit_transform(X_selected)

        training_results = {}

        if self.model_type == 'ensemble':
            # Train multiple models and ensemble them
            ensemble_predictions = np.zeros(len(y))
            model_weights = {}

            for model_name, model in self.models.items():
                print(f"    Training {model_name}...")

                # Set class weight if needed
                if class_weight and hasattr(model, 'class_weight'):
                    model.set_params(class_weight=class_weight)

                # Cross-validation for model evaluation
                cv_scores = cross_val_score(
                    model, X_scaled, y,
                    cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=self.config.seed),
                    scoring='roc_auc'
                )

                print(f"    {model_name} CV AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

                # Train model on full data
                model.fit(X_scaled, y)

                # Get predictions for ensemble
                if hasattr(model, 'predict_proba'):
                    probas = model.predict_proba(X_scaled)[:, 1]
                else:
                    probas = model.decision_function(X_scaled)
                    # Convert to probabilities (simple sigmoid approximation)
                    probas = 1 / (1 + np.exp(-probas))

                # Weight by CV performance
                weight = cv_scores.mean()
                ensemble_predictions += weight * probas
                model_weights[model_name] = weight

            # Normalize ensemble predictions
            total_weight = sum(model_weights.values())
            ensemble_predictions /= total_weight

            # Convert to binary predictions
            threshold = np.percentile(ensemble_predictions, 100 * (1 - pos_ratio))
            ensemble_binary = (ensemble_predictions > threshold).astype(int)

            training_results = {
                'model_type': 'ensemble',
                'models_trained': list(self.models.keys()),
                'model_weights': model_weights,
                'feature_names': selected_features,
                'cv_scores': cv_scores,
                'ensemble_predictions': ensemble_predictions,
                'ensemble_binary': ensemble_binary,
                'threshold': threshold,
                'positive_ratio': pos_ratio
            }

        else:
            # Single model training
            model = list(self.models.values())[0]
            print(f"    Training {self.model_type}...")

            if class_weight and hasattr(model, 'class_weight'):
                model.set_params(class_weight=class_weight)

            # Cross-validation
            cv_scores = cross_val_score(
                model, X_scaled, y,
                cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=self.config.seed),
                scoring='roc_auc'
            )

            print(f"    CV AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

            # Train on full data
            model.fit(X_scaled, y)

            # Get predictions
            if hasattr(model, 'predict_proba'):
                predictions = model.predict_proba(X_scaled)[:, 1]
            else:
                predictions = model.decision_function(X_scaled)
                predictions = 1 / (1 + np.exp(-predictions))

            training_results = {
                'model_type': self.model_type,
                'feature_names': selected_features,
                'cv_scores': cv_scores,
                'predictions': predictions,
                'positive_ratio': pos_ratio
            }

        return training_results

    def predict_variants(self, collapsed_df: pd.DataFrame) -> np.ndarray:
        """Predict variants using trained ML model.

        Args:
            collapsed_df: DataFrame with collapsed UMI data

        Returns:
            Array of variant probabilities
        """
        # Extract features
        X, _ = self.feature_engineer.extract_features(collapsed_df)

        # Feature selection (use same features as training)
        if hasattr(self, '_selected_features'):
            X_selected = X[self._selected_features]
        else:
            # If not trained yet, use all features
            X_selected = X

        # Scale features
        X_scaled = self.scaler.transform(X_selected)

        if self.model_type == 'ensemble':
            # Ensemble prediction
            ensemble_predictions = np.zeros(X_scaled.shape[0])

            for model_name, model in self.models.items():
                if hasattr(model, 'predict_proba'):
                    probas = model.predict_proba(X_scaled)[:, 1]
                else:
                    probas = model.decision_function(X_scaled)
                    probas = 1 / (1 + np.exp(-probas))

                weight = getattr(self, '_model_weights', {}).get(model_name, 1.0)
                ensemble_predictions += weight * probas

            total_weight = sum(getattr(self, '_model_weights', {}).values()) or len(self.models)
            ensemble_predictions /= total_weight

            return ensemble_predictions
        else:
            # Single model prediction
            model = list(self.models.values())[0]
            if hasattr(model, 'predict_proba'):
                return model.predict_proba(X_scaled)[:, 1]
            else:
                predictions = model.decision_function(X_scaled)
                return 1 / (1 + np.exp(-predictions))

    def _generate_synthetic_labels(self, collapsed_df: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
        """Generate synthetic labels for training when ground truth is not available.

        This is a fallback method that creates reasonable synthetic labels based on
        allele fraction and quality metrics.
        """
        # Use allele fraction as primary signal
        af_signal = collapsed_df.get('allele_fraction', pd.Series([0.001] * len(collapsed_df)))

        # Add noise based on quality and consensus
        quality_factor = collapsed_df.get('quality_score', pd.Series([25] * len(collapsed_df))) / 50.0
        consensus_factor = collapsed_df.get('consensus_agreement', pd.Series([0.8] * len(collapsed_df)))

        # Combine signals
        combined_signal = af_signal * quality_factor * consensus_factor

        # Convert to binary labels with some noise
        # Higher signal = more likely to be variant
        probabilities = np.clip(combined_signal * 100, 0, 1)  # Scale to 0-1

        # Add some randomness
        noise = rng.normal(0, 0.1, len(probabilities))
        noisy_probabilities = np.clip(probabilities + noise, 0, 1)

        # Convert to binary labels
        return (noisy_probabilities > 0.5).astype(int)

    def evaluate_model(self, X_test: pd.DataFrame, y_test: np.ndarray) -> Dict[str, float]:
        """Evaluate trained model on test data.

        Args:
            X_test: Test features
            y_test: Test labels

        Returns:
            Dictionary with evaluation metrics
        """
        predictions = self.predict_variants(X_test)

        # Convert predictions to binary for classification metrics
        threshold = 0.5  # Default threshold
        binary_predictions = (predictions > threshold).astype(int)

        metrics = {}

        # ROC AUC
        try:
            metrics['roc_auc'] = roc_auc_score(y_test, predictions)
        except ValueError:
            metrics['roc_auc'] = 0.5

        # Average precision
        try:
            metrics['avg_precision'] = average_precision_score(y_test, predictions)
        except ValueError:
            metrics['avg_precision'] = np.mean(y_test)

        # Classification report (convert to dict)
        try:
            report = classification_report(y_test, binary_predictions, output_dict=True, zero_division=0)
            metrics.update({
                'precision': report['weighted avg']['precision'],
                'recall': report['weighted avg']['recall'],
                'f1_score': report['weighted avg']['f1-score']
            })
        except Exception:
            metrics.update({'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0})

        return metrics

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores from trained model.

        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.models:
            return {}

        # For ensemble, average importance across models
        if self.model_type == 'ensemble':
            total_importance = {}
            for model_name, model in self.models.items():
                if hasattr(model, 'feature_importances_'):
                    importance = model.feature_importances_
                    for i, feat_name in enumerate(getattr(self, '_selected_features', [])):
                        total_importance[feat_name] = total_importance.get(feat_name, 0) + importance[i]

            # Normalize by number of models
            for feat_name in total_importance:
                total_importance[feat_name] /= len(self.models)

            return total_importance
        else:
            # Single model
            model = list(self.models.values())[0]
            if hasattr(model, 'feature_importances_'):
                selected_features = getattr(self, '_selected_features', [])
                return {
                    feat_name: importance
                    for feat_name, importance in zip(selected_features, model.feature_importances_)
                }
            elif hasattr(model, 'coef_'):
                # For linear models
                selected_features = getattr(self, '_selected_features', [])
                return {
                    feat_name: abs(coef)
                    for feat_name, coef in zip(selected_features, model.coef_[0])
                }

        return {}


class AdaptiveThresholdOptimizer:
    """Adaptive threshold optimization for ML-based variant calling."""

    def __init__(self, config: PipelineConfig):
        """Initialize threshold optimizer.

        Args:
            config: Pipeline configuration
        """
        self.config = config

    def optimize_threshold(self, predictions: np.ndarray, true_labels: np.ndarray,
                          metric: str = 'f1') -> float:
        """Find optimal threshold for variant calling.

        Args:
            predictions: Predicted probabilities
            true_labels: True labels (if available)
            metric: Metric to optimize ('f1', 'precision', 'recall')

        Returns:
            Optimal threshold value
        """
        if true_labels is None:
            # If no ground truth, use statistical approach
            # Use median or 95th percentile based on expected false positive rate
            expected_fpr = self.config.stats.alpha
            if metric == 'f1':
                # For F1 optimization, use a threshold that balances precision and recall
                return np.percentile(predictions, 100 * (1 - expected_fpr * 2))
            else:
                return np.percentile(predictions, 100 * (1 - expected_fpr))

        # If we have ground truth, optimize threshold
        thresholds = np.linspace(0.01, 0.99, 50)
        best_threshold = 0.5
        best_score = 0.0

        for threshold in thresholds:
            binary_predictions = (predictions > threshold).astype(int)

            if metric == 'f1':
                try:
                    from sklearn.metrics import f1_score
                    score = f1_score(true_labels, binary_predictions, zero_division=0)
                except:
                    score = 0.0
            elif metric == 'precision':
                try:
                    from sklearn.metrics import precision_score
                    score = precision_score(true_labels, binary_predictions, zero_division=0)
                except:
                    score = 0.0
            elif metric == 'recall':
                try:
                    from sklearn.metrics import recall_score
                    score = recall_score(true_labels, binary_predictions, zero_division=0)
                except:
                    score = 0.0
            else:
                score = 0.0

            if score > best_score:
                best_score = score
                best_threshold = threshold

        return best_threshold

    def get_confidence_intervals(self, predictions: np.ndarray, threshold: float,
                               n_bootstrap: int = 100) -> Dict[str, float]:
        """Calculate confidence intervals for predictions using bootstrap.

        Args:
            predictions: Predicted probabilities
            threshold: Decision threshold
            n_bootstrap: Number of bootstrap samples

        Returns:
            Dictionary with confidence intervals
        """
        binary_predictions = (predictions > threshold).astype(int)
        n_positives = np.sum(binary_predictions)

        if n_positives == 0:
            return {'lower': 0.0, 'upper': 0.0, 'mean': 0.0}

        # Bootstrap confidence intervals
        bootstrap_proportions = []

        for _ in range(n_bootstrap):
            # Sample with replacement
            indices = np.random.choice(len(predictions), size=len(predictions), replace=True)
            boot_predictions = predictions[indices]
            boot_binary = (boot_predictions > threshold).astype(int)
            boot_proportion = np.mean(boot_binary)
            bootstrap_proportions.append(boot_proportion)

        bootstrap_proportions = np.array(bootstrap_proportions)
        lower_ci = np.percentile(bootstrap_proportions, 2.5)
        upper_ci = np.percentile(bootstrap_proportions, 97.5)
        mean_proportion = np.mean(bootstrap_proportions)

        return {
            'lower': lower_ci,
            'upper': upper_ci,
            'mean': mean_proportion
        }


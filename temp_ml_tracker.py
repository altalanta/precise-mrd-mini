class MLPerformanceTracker:
    """Track ML model performance metrics."""

    def __init__(self):
        """Initialize ML performance tracker."""
        self.model_metrics = {}
        self.feature_importance = {}
        self.prediction_distributions = {}
        self.training_times = {}

    def record_model_metrics(self, model_name: str, metrics: Dict[str, float]):
        """Record ML model performance metrics.

        Args:
            model_name: Name of the ML model
            metrics: Dictionary of performance metrics
        """
        self.model_metrics[model_name] = metrics.copy()

    def record_feature_importance(self, model_name: str, importance: Dict[str, float]):
        """Record feature importance scores.

        Args:
            model_name: Name of the ML model
            importance: Dictionary mapping feature names to importance scores
        """
        self.feature_importance[model_name] = importance.copy()

    def record_prediction_distribution(self, model_name: str, predictions: np.ndarray):
        """Record prediction score distribution.

        Args:
            model_name: Name of the ML model
            predictions: Array of prediction scores
        """
        self.prediction_distributions[model_name] = {
            'mean': np.mean(predictions),
            'std': np.std(predictions),
            'min': np.min(predictions),
            'max': np.max(predictions),
            'median': np.median(predictions),
            'q25': np.percentile(predictions, 25),
            'q75': np.percentile(predictions, 75)
        }

    def record_training_time(self, model_name: str, training_time: float):
        """Record model training time.

        Args:
            model_name: Name of the ML model
            training_time: Training time in seconds
        """
        self.training_times[model_name] = training_time

    def get_ml_report(self) -> Dict[str, Any]:
        """Get comprehensive ML performance report.

        Returns:
            Dictionary with ML performance metrics
        """
        return {
            'model_metrics': self.model_metrics,
            'feature_importance': self.feature_importance,
            'prediction_distributions': self.prediction_distributions,
            'training_times': self.training_times,
            'n_models_tracked': len(self.model_metrics)
        }

    def compare_models(self) -> Dict[str, Any]:
        """Compare performance across different models.

        Returns:
            Dictionary with model comparison results
        """
        if not self.model_metrics:
            return {}

        comparison = {
            'best_model': None,
            'best_metric': None,
            'model_ranking': []
        }

        # Find best model by ROC AUC
        best_auc = 0.0
        best_model = None

        for model_name, metrics in self.model_metrics.items():
            auc = metrics.get('roc_auc', 0.0)
            if auc > best_auc:
                best_auc = auc
                best_model = model_name

        comparison['best_model'] = best_model
        comparison['best_metric'] = best_auc

        # Rank models by AUC
        model_ranking = sorted(
            [(name, metrics.get('roc_auc', 0.0)) for name, metrics in self.model_metrics.items()],
            key=lambda x: x[1],
            reverse=True
        )
        comparison['model_ranking'] = model_ranking

        return comparison


# Global ML performance tracker

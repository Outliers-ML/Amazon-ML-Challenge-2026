"""Ensembling strategies module.

Includes:
1. `WeightedBlender`: Linear convex combination with Optuna / SciPy weight optimization.
2. `rank_average`: Rank transformation before averaging (robust to scale calibration mismatches).
3. `StackingEnsemble`: Meta-model trained on out-of-fold validation predictions.
"""

from typing import List, Optional
import numpy as np
from scipy.optimize import minimize
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression, Ridge

from src.evaluation.metrics import get_metric
from src.utils.logger import get_logger

logger = get_logger("Ensemble")


def rank_average(predictions_list: List[np.ndarray], weights: Optional[List[float]] = None) -> np.ndarray:
    """Compute rank-averaged predictions across multiple model prediction arrays.

    Normalizes predictions to percentile ranks before averaging, making it invariant to
    differing calibration curves and probability scalings between disparate models.

    Args:
        predictions_list: List of 1D prediction arrays from individual models.
        weights: Optional list of relative model weights.

    Returns:
        1D array of blended percentile ranks in [0, 1].
    """
    n_models = len(predictions_list)
    if weights is None:
        weights = [1.0 / n_models] * n_models
    else:
        weights = [w / sum(weights) for w in weights]

    blended = np.zeros_like(predictions_list[0], dtype=np.float64)
    for preds, w in zip(predictions_list, weights):
        # Rank from 0 to 1
        ranks = (rankdata(preds) - 1.0) / (len(preds) - 1.0)
        blended += w * ranks

    return blended


class WeightedBlender:
    """Finds optimal ensemble weights on out-of-fold validation predictions."""

    def __init__(self, metric_name: str = "rmse", maximize: bool = False) -> None:
        """Initialize WeightedBlender.

        Args:
            metric_name: Target metric to optimize.
            maximize: True if higher metric is better (e.g. F1, ROC-AUC), False for error (e.g. RMSE).
        """
        self.metric_name = metric_name
        self.maximize = maximize
        self.metric_fn = get_metric(metric_name)
        self.weights: Optional[np.ndarray] = None

    def fit(self, oof_predictions: List[np.ndarray], y_true: np.ndarray) -> "WeightedBlender":
        """Optimize non-negative weights that sum to 1 using SciPy SLSQP.

        Args:
            oof_predictions: List of out-of-fold prediction arrays from each model.
            y_true: True ground truth values.

        Returns:
            Self.
        """
        preds_matrix = np.column_stack(oof_predictions)
        n_models = preds_matrix.shape[1]
        initial_weights = np.ones(n_models) / n_models

        def loss_func(w):
            blend = np.dot(preds_matrix, w)
            score = self.metric_fn(y_true, blend)
            return -score if self.maximize else score

        bounds = [(0.0, 1.0) for _ in range(n_models)]
        constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

        res = minimize(
            loss_func,
            initial_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )
        self.weights = res.x
        logger.info(f"Optimized blend weights: {np.round(self.weights, 4)}")
        return self

    def predict(self, test_predictions: List[np.ndarray]) -> np.ndarray:
        """Blend test predictions using learned weights."""
        if self.weights is None:
            raise ValueError("Blender has not been fitted yet.")
        preds_matrix = np.column_stack(test_predictions)
        return np.dot(preds_matrix, self.weights)


class StackingEnsemble:
    """Out-of-fold Stacking Meta-Learner."""

    def __init__(self, is_classification: bool = True) -> None:
        """Initialize StackingEnsemble.

        Args:
            is_classification: True for classification, False for regression.
        """
        self.is_classification = is_classification
        if is_classification:
            self.meta_model = LogisticRegression(C=1.0, max_iter=1000)
        else:
            self.meta_model = Ridge(alpha=1.0)

    def fit(self, oof_predictions: List[np.ndarray], y_true: np.ndarray) -> "StackingEnsemble":
        """Train meta-model on out-of-fold predictions."""
        X_meta = np.column_stack(oof_predictions)
        self.meta_model.fit(X_meta, y_true)
        logger.info("Stacking meta-model fitted.")
        return self

    def predict(self, test_predictions: List[np.ndarray]) -> np.ndarray:
        """Generate final predictions from test model outputs."""
        X_test_meta = np.column_stack(test_predictions)
        return self.meta_model.predict(X_test_meta)

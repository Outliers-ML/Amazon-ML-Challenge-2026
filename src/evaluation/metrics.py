"""Evaluation metrics implementation for classification and regression tasks.

Provides metric calculation functions commonly used in Amazon ML Challenges:
Macro/Micro/Weighted F1, Accuracy, Log Loss, ROC-AUC, RMSE, MAE, MAPE, SMAPE.
"""

from typing import Callable, Dict
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric Mean Absolute Percentage Error (SMAPE).

    Calculation: 100/n * sum( 2 * |y_pred - y_true| / (|y_true| + |y_pred| + eps) )

    Args:
        y_true: Ground truth target values.
        y_pred: Predicted target values.

    Returns:
        SMAPE score percentage as float.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    diff = np.abs(y_pred - y_true)
    # Avoid zero division
    mask = denominator != 0
    if not np.any(mask):
        return 0.0
    return float(np.mean(diff[mask] / denominator[mask]) * 100.0)


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error (MAPE).

    Args:
        y_true: Ground truth target values.
        y_pred: Predicted target values.

    Returns:
        MAPE score percentage.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mask = y_true != 0
    if not np.any(mask):
        return 0.0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error (RMSE)."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


METRIC_REGISTRY: Dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
    # Classification
    "f1_macro": lambda yt, yp: float(f1_score(yt, yp, average="macro", zero_division=0)),
    "f1_micro": lambda yt, yp: float(f1_score(yt, yp, average="micro", zero_division=0)),
    "f1_weighted": lambda yt, yp: float(f1_score(yt, yp, average="weighted", zero_division=0)),
    "accuracy": lambda yt, yp: float(accuracy_score(yt, yp)),
    "log_loss": lambda yt, yp: float(log_loss(yt, yp)),
    "roc_auc": lambda yt, yp: float(roc_auc_score(yt, yp)),
    # Regression
    "rmse": rmse,
    "mse": lambda yt, yp: float(mean_squared_error(yt, yp)),
    "mae": lambda yt, yp: float(mean_absolute_error(yt, yp)),
    "mape": mape,
    "smape": smape,
}


def get_metric(metric_name: str) -> Callable[[np.ndarray, np.ndarray], float]:
    """Retrieve scoring function by name.

    Args:
        metric_name: Metric identifier string.

    Returns:
        Callable taking (y_true, y_pred) and returning float score.

    Raises:
        KeyError: If metric_name is not registered.
    """
    metric_key = metric_name.lower().strip()
    if metric_key not in METRIC_REGISTRY:
        available = ", ".join(METRIC_REGISTRY.keys())
        raise KeyError(f"Metric '{metric_name}' not recognized. Available metrics: {available}")
    return METRIC_REGISTRY[metric_key]


def compute_metric(metric_name: str, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute score for given metric name and predictions.

    Args:
        metric_name: Target metric identifier.
        y_true: Ground truth array.
        y_pred: Predicted array (or class indices).

    Returns:
        Float metric score.
    """
    fn = get_metric(metric_name)
    return fn(y_true, y_pred)

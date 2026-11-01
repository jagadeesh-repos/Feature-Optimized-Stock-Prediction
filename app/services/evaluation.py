"""Evaluate continuous model outputs with regression metrics.

This module follows the user-supplied notebook's MSE, MAE, and RMSE evaluation.
It does not compute binary-classification metrics because the current modeling
baseline is a regressor predicting forward drawdown.
"""

from typing import Any

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def evaluate_regression(y_true: Any, y_predicted: Any) -> dict[str, float]:
    """Return MSE, MAE, and RMSE after validating aligned finite vectors."""
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_predicted, dtype=float)
    if actual.ndim != 1 or predicted.ndim != 1:
        raise ValueError("actual and predicted values must be one-dimensional")
    if actual.size == 0 or actual.shape != predicted.shape:
        raise ValueError("actual and predicted values must be non-empty and aligned")
    if not np.isfinite(actual).all() or not np.isfinite(predicted).all():
        raise ValueError("actual and predicted values must be finite")

    mse = float(mean_squared_error(actual, predicted))
    mae = float(mean_absolute_error(actual, predicted))
    return {"mse": mse, "mae": mae, "rmse": float(np.sqrt(mse))}

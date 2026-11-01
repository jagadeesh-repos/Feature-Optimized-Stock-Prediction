"""Tests for the regression metric contract used by the supplied baseline."""

import numpy as np
import pytest

from app.services.evaluation import evaluate_regression


def test_evaluate_regression_returns_expected_metrics():
    """Metric values are calculated from the supplied true and predicted values."""
    result = evaluate_regression([1.0, 2.0, 3.0], [1.0, 3.0, 2.0])

    assert result["mse"] == pytest.approx(2 / 3)
    assert result["mae"] == pytest.approx(2 / 3)
    assert result["rmse"] == pytest.approx(np.sqrt(2 / 3))


@pytest.mark.parametrize(
    ("actual", "predicted"),
    [([], []), ([1.0], [1.0, 2.0]), ([[1.0]], [[1.0]]), ([np.nan], [1.0])],
)
def test_evaluate_regression_rejects_invalid_vectors(actual, predicted):
    """Invalid shapes, lengths, and non-finite values cannot yield metrics."""
    with pytest.raises(ValueError):
        evaluate_regression(actual, predicted)

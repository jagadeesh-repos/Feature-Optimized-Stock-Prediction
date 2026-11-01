"""Tests for the training-only RFE and Boruta selector."""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from app.services.feature_selection import select_features_hybrid


def _estimators():
    """Use small deterministic trees so the test stays fast and offline."""
    return {
        "rfe_estimator": XGBRegressor(
            n_estimators=8, max_depth=2, random_state=7, n_jobs=1, verbosity=0
        ),
        "boruta_estimator": RandomForestRegressor(
            n_estimators=12, max_depth=3, random_state=7, n_jobs=1
        ),
    }


def test_hybrid_selection_returns_names_and_intersection_union_metadata():
    """Both methods run on the supplied data and selected names are valid inputs."""
    random = np.random.default_rng(42)
    values = random.normal(size=(80, 5))
    target = 3 * values[:, 0] - 2 * values[:, 2] + random.normal(size=80) * 0.1
    columns = [f"indicator_{index}" for index in range(values.shape[1])]

    result = select_features_hybrid(
        pd.DataFrame(values, columns=columns),
        target,
        n_features_to_select=2,
        boruta_max_iter=5,
        **_estimators(),
    )

    assert result.rfe_features
    assert set(result.rfe_features).issubset(columns)
    assert set(result.boruta_features).issubset(columns)
    assert result.selected_features == (result.intersection or result.union)
    assert result.config["selection_rule"] == "intersection_fallback_union"


def test_default_selection_count_is_bounded_by_feature_count():
    """The notebook's five-feature minimum is bounded for narrow inputs."""
    random = np.random.default_rng(9)
    values = random.normal(size=(40, 3))
    target = values[:, 0] + random.normal(size=40) * 0.2

    result = select_features_hybrid(values, target, boruta_max_iter=3, **_estimators())

    assert result.config["rfe_n_features_to_select"] == 3


@pytest.mark.parametrize(
    ("features", "target"),
    [
        (np.ones((4, 2)), np.ones((4, 1))),
        (np.ones((4, 2)), np.ones(3)),
        (np.array([[1.0, np.nan], [2.0, 3.0]]), np.array([1.0, 2.0])),
    ],
)
def test_invalid_training_arrays_are_rejected(features, target):
    """Selector input validation prevents silent shape and missing-data issues."""
    with pytest.raises(ValueError):
        select_features_hybrid(
            features,
            target,
            n_features_to_select=1,
            boruta_max_iter=2,
            **_estimators(),
        )

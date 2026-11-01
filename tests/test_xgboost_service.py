"""Offline unit tests for XGBoost training and artifact persistence."""

import numpy as np
import pandas as pd
import pytest
from xgboost import XGBRegressor

from app.services.xgboost_service import (
    XGBoostConfig,
    load_xgboost_model,
    save_xgboost_model,
    train_xgboost_regressor,
    xgboost_metadata,
)


def _training_data():
    """Create a deterministic non-market regression fixture."""
    random = np.random.default_rng(42)
    features = pd.DataFrame(
        random.normal(size=(60, 3)),
        columns=["indicator_a", "indicator_b", "indicator_c"],
    )
    target = 2 * features["indicator_a"] - features["indicator_c"]
    return features, target


def _config(**overrides):
    """Keep the tests small while exercising the same service path."""
    values = {
        "n_estimators": 20,
        "learning_rate": 0.1,
        "early_stopping_rounds": 3,
        "max_depth": 2,
        "n_jobs": 1,
    }
    values.update(overrides)
    return XGBoostConfig(**values)


def test_train_regressor_uses_train_and_optional_validation_data():
    """The model fits and exposes predictions for the separately held-out rows."""
    features, target = _training_data()
    model = train_xgboost_regressor(
        features.iloc[:40],
        target.iloc[:40],
        validation_data=(features.iloc[40:50], target.iloc[40:50]),
        config=_config(),
    )

    assert isinstance(model, XGBRegressor)
    assert model.predict(features.iloc[50:]).shape == (10,)
    assert model.feature_names_in_.tolist() == list(features.columns)


def test_train_without_validation_does_not_enable_early_stopping():
    """The service cannot accidentally use an evaluation/test fold for stopping."""
    features, target = _training_data()
    model = train_xgboost_regressor(
        features, target, config=_config(early_stopping_rounds=4)
    )

    assert model.get_params()["early_stopping_rounds"] is None


def test_model_can_be_saved_and_loaded_within_the_requested_path(tmp_path):
    """Serialization round-trips the fitted estimator without fixed global paths."""
    features, target = _training_data()
    model = train_xgboost_regressor(features, target, config=_config())
    destination = save_xgboost_model(model, tmp_path / "models" / "model.joblib")
    restored = load_xgboost_model(destination)

    np.testing.assert_allclose(
        restored.predict(features), model.predict(features), rtol=0, atol=0
    )


def test_metadata_is_json_compatible():
    """Model settings can be persisted directly in the report JSON field."""
    config = _config()

    assert xgboost_metadata(config)["n_estimators"] == 20


def test_invalid_training_shapes_are_rejected():
    """Bad training arrays fail before fitting starts."""
    with pytest.raises(ValueError, match="match feature rows"):
        train_xgboost_regressor(np.ones((4, 2)), np.ones(3), config=_config())

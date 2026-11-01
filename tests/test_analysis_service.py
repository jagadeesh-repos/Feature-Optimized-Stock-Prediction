"""Tests for chronological purging and the analysis orchestration boundary."""

import numpy as np
import pandas as pd
import pytest

from app.services import analysis_service
from app.services.analysis_service import AnalysisConfig, _purged_chronological_split
from app.services.feature_engineering import FeatureWindows
from app.services.feature_selection import FeatureSelectionResult
from app.services.xgboost_service import XGBoostConfig


def _labeled_rows(count: int) -> pd.DataFrame:
    """Create ordered rows with one feature and one forward target."""
    return pd.DataFrame(
        {
            "Date": pd.date_range("2020-01-01", periods=count, freq="B"),
            "signal": np.arange(count, dtype=float),
            "future_drawdown": np.arange(count, dtype=float) / count,
        }
    )


def test_purged_chronological_split_leaves_horizon_gaps():
    """Boundary targets cannot look into the adjacent validation or test period."""
    train, validation, test, metadata = _purged_chronological_split(
        _labeled_rows(100),
        prediction_horizon=5,
        training_fraction=0.70,
        validation_fraction=0.15,
    )

    assert len(train) == 65
    assert len(validation) == 10
    assert len(test) == 15
    assert train.index[-1] + 5 < validation.index[0]
    assert validation.index[-1] + 5 < test.index[0]
    assert metadata["purged_rows_at_each_boundary"] == 5


def test_split_rejects_histories_too_short_for_purged_partitions():
    """A short history fails clearly instead of producing empty folds."""
    with pytest.raises(ValueError, match="not enough labeled observations"):
        _purged_chronological_split(
            _labeled_rows(20),
            prediction_horizon=8,
            training_fraction=0.70,
            validation_fraction=0.15,
        )


def test_run_analysis_keeps_test_data_out_of_feature_selection_and_fitting(
    monkeypatch,
):
    """The pipeline passes only training rows to HFS and evaluates held-out rows."""
    count = 180
    dates = pd.date_range("2020-01-01", periods=count, freq="B")
    close = 100 + np.arange(count, dtype=float) * 0.1
    bars = pd.DataFrame(
        {
            "Date": dates,
            "Open": close,
            "High": close + 1,
            "Low": close - 1,
            "Close": close,
            "Volume": 1000 + np.arange(count, dtype=float),
        }
    )
    monkeypatch.setattr(analysis_service, "collect_historical_data", lambda *args: bars)

    captured = {}
    selection = FeatureSelectionResult(
        selected_features=("return_1",),
        rfe_features=("return_1",),
        boruta_features=("return_1",),
        intersection=("return_1",),
        union=("return_1",),
        config={"selection_rule": "test"},
    )

    def fake_select(features, target, names, **kwargs):
        captured["feature_selection_rows"] = len(features)
        captured["feature_selection_last_date"] = features.index[-1]
        return selection

    class FakeXGBoost:
        feature_importances_ = np.array([1.0])

        def predict(self, features):
            captured.setdefault("predicted_rows", []).append(len(features))
            return np.full(len(features), 0.02)

    def fake_train(features, target, **kwargs):
        captured["fit_rows"] = len(features)
        captured["validation_rows"] = len(kwargs["validation_data"][0])
        return FakeXGBoost()

    monkeypatch.setattr(analysis_service, "select_features_hybrid", fake_select)
    monkeypatch.setattr(analysis_service, "train_xgboost_regressor", fake_train)

    config = AnalysisConfig(
        feature_windows=FeatureWindows(
            sma=(2,),
            ema=(2,),
            macd_fast=2,
            macd_slow=3,
            macd_signal=2,
            rsi=2,
            volatility=2,
            momentum=(1,),
        ),
        xgboost=XGBoostConfig(n_estimators=5, early_stopping_rounds=None),
        boruta_max_iter=2,
    )
    result = analysis_service.run_analysis(
        "TEST.NS",
        "2020-01-01",
        "2021-01-01",
        prediction_horizon=3,
        crisis_threshold=0.10,
        selected_model="xgboost",
        config=config,
    )

    assert captured["feature_selection_rows"] == result["split"]["training_rows"]
    assert captured["fit_rows"] == result["split"]["training_rows"]
    assert captured["validation_rows"] == result["split"]["validation_rows"]
    assert (
        len(result["models"]["xgboost"]["test_predictions"])
        == result["split"]["test_rows"]
    )
    assert result["models"]["xgboost"]["latest_predicted_drawdown"] == 0.02
    assert result["models"]["xgboost"]["latest_predicted_crisis"] is False

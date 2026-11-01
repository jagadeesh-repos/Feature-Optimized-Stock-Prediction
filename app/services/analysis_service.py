"""Orchestrate the user-supplied regression workflow without test leakage.

Split fractions are project defaults, not paper-derived settings. The input
history is split chronologically; ``prediction_horizon`` rows are purged at
each train/validation and validation/test boundary so forward-label windows do
not cross those boundaries. Feature selection, scaling, and early stopping use
training and validation partitions only. The final test partition is evaluated
once and never passed into fitting or feature selection.
"""

from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from app.services.crisis_labeling import (
    CRISIS_EVENT_COLUMN,
    FORWARD_DRAWDOWN_COLUMN,
    create_crisis_targets,
)
from app.services.data_collection import collect_historical_data
from app.services.dnn_service import (
    DNNConfig,
    predict_dnn,
    train_dnn_regressor,
)
from app.services.evaluation import evaluate_regression
from app.services.feature_engineering import FeatureWindows, engineer_technical_features
from app.services.feature_selection import select_features_hybrid
from app.services.preprocessing import preprocess_ohlcv
from app.services.xgboost_service import (
    XGBoostConfig,
    train_xgboost_regressor,
)


@dataclass(frozen=True)
class AnalysisConfig:
    """Project split ratios and configurable feature/model settings."""

    training_fraction: float = 0.70
    validation_fraction: float = 0.15
    feature_windows: FeatureWindows = FeatureWindows()
    xgboost: XGBoostConfig = XGBoostConfig()
    dnn: DNNConfig = DNNConfig()
    boruta_max_iter: int = 100
    random_seed: int = 42

    def __post_init__(self) -> None:
        """Ensure chronological split fractions leave a positive test partition."""
        if not 0 < self.training_fraction < 1:
            raise ValueError("training_fraction must be between 0 and 1")
        if not 0 < self.validation_fraction < 1:
            raise ValueError("validation_fraction must be between 0 and 1")
        if self.training_fraction + self.validation_fraction >= 1:
            raise ValueError("training and validation fractions must leave test data")
        if self.boruta_max_iter < 1:
            raise ValueError("boruta_max_iter must be positive")


def _purged_chronological_split(
    data: pd.DataFrame,
    *,
    prediction_horizon: int,
    training_fraction: float,
    validation_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Partition labeled rows chronologically with horizon-sized boundary gaps."""
    if isinstance(prediction_horizon, bool) or not isinstance(prediction_horizon, int):
        raise ValueError("prediction_horizon must be a positive integer")
    if prediction_horizon < 1:
        raise ValueError("prediction_horizon must be a positive integer")
    if data.empty:
        raise ValueError("at least one labeled row is required")
    if "Date" not in data or not pd.to_datetime(data["Date"]).is_monotonic_increasing:
        raise ValueError("data must have chronologically ordered Date values")

    row_count = len(data)
    train_boundary = int(row_count * training_fraction)
    validation_boundary = int(row_count * (training_fraction + validation_fraction))
    train_stop = train_boundary - prediction_horizon
    validation_stop = validation_boundary - prediction_horizon

    train = data.iloc[:train_stop].copy()
    validation = data.iloc[train_boundary:validation_stop].copy()
    test = data.iloc[validation_boundary:].copy()
    if train.empty or validation.empty or test.empty:
        raise ValueError(
            "not enough labeled observations for chronological partitions and purges"
        )

    split_metadata = {
        "training_rows": len(train),
        "validation_rows": len(validation),
        "test_rows": len(test),
        "purged_rows_at_each_boundary": prediction_horizon,
        "training_end": pd.Timestamp(train["Date"].iloc[-1]).isoformat(),
        "validation_start": pd.Timestamp(validation["Date"].iloc[0]).isoformat(),
        "validation_end": pd.Timestamp(validation["Date"].iloc[-1]).isoformat(),
        "test_start": pd.Timestamp(test["Date"].iloc[0]).isoformat(),
    }
    return train, validation, test, split_metadata


def _nonnegative_predictions(model: Any, features: pd.DataFrame) -> np.ndarray:
    """Constrain regression estimates to the non-negative drawdown domain."""
    return np.maximum(np.asarray(model.predict(features), dtype=float).reshape(-1), 0.0)


def run_analysis(
    ticker: str,
    start_date: date | str,
    end_date: date | str,
    *,
    prediction_horizon: int,
    crisis_threshold: float,
    selected_model: str = "both",
    provider: Any | None = None,
    config: AnalysisConfig = AnalysisConfig(),
) -> dict[str, Any]:
    """Collect, featurize, select, fit, evaluate, and forecast a stock series.

    ``prediction_horizon`` and ``crisis_threshold`` are intentionally required;
    neither the paper's exact per-stock label nor a project default is verified.
    Returned regression forecasts are drawdown estimates, not crisis probabilities.
    """
    ticker = ticker.strip() if isinstance(ticker, str) else ""
    if not ticker:
        raise ValueError("ticker is required")
    if selected_model not in {"xgboost", "dnn", "both"}:
        raise ValueError("selected_model must be 'xgboost', 'dnn', or 'both'")

    raw_data = collect_historical_data(ticker, start_date, end_date, provider)
    cleaned = preprocess_ohlcv(raw_data)
    featured = engineer_technical_features(cleaned, config.feature_windows)
    labeled = create_crisis_targets(
        featured,
        prediction_horizon=prediction_horizon,
        crisis_threshold=crisis_threshold,
    )

    excluded = {"Date", FORWARD_DRAWDOWN_COLUMN, CRISIS_EVENT_COLUMN}
    feature_names = [
        column
        for column in labeled.select_dtypes(include=[np.number]).columns
        if column not in excluded
    ]
    if not feature_names:
        raise ValueError("no numeric model features are available")
    complete = labeled.dropna(subset=[*feature_names, FORWARD_DRAWDOWN_COLUMN]).copy()
    if complete.empty:
        raise ValueError("not enough complete feature and target observations")

    train, validation, test, split_metadata = _purged_chronological_split(
        complete,
        prediction_horizon=prediction_horizon,
        training_fraction=config.training_fraction,
        validation_fraction=config.validation_fraction,
    )
    selection = select_features_hybrid(
        train.loc[:, feature_names],
        train[FORWARD_DRAWDOWN_COLUMN],
        feature_names,
        boruta_max_iter=config.boruta_max_iter,
        random_seed=config.random_seed,
    )
    selected_features = list(selection.selected_features)
    if not selected_features:
        raise ValueError("hybrid feature selection returned no features")

    train_x = train.loc[:, selected_features]
    train_y = train[FORWARD_DRAWDOWN_COLUMN]
    validation_x = validation.loc[:, selected_features]
    validation_y = validation[FORWARD_DRAWDOWN_COLUMN]
    test_x = test.loc[:, selected_features]
    test_y = test[FORWARD_DRAWDOWN_COLUMN].to_numpy(dtype=float)
    actual_test_events = test[CRISIS_EVENT_COLUMN].astype(int).tolist()

    # Latest feature rows can be unlabeled because their full future window is
    # not observable yet. They are forecast-only and never enter model fitting.
    latest_feature_row = featured.dropna(subset=feature_names).iloc[[-1]]
    latest_prediction_date = pd.Timestamp(
        latest_feature_row["Date"].iloc[0]
    ).isoformat()
    latest_features = latest_feature_row.loc[:, selected_features]

    model_results: dict[str, Any] = {}
    if selected_model in {"xgboost", "both"}:
        xgb = train_xgboost_regressor(
            train_x,
            train_y,
            validation_data=(validation_x, validation_y),
            config=config.xgboost,
        )
        test_prediction = _nonnegative_predictions(xgb, test_x)
        latest_prediction = float(_nonnegative_predictions(xgb, latest_features)[0])
        importance = getattr(
            xgb, "feature_importances_", np.zeros(len(selected_features))
        )
        model_results["xgboost"] = {
            "metrics": evaluate_regression(test_y, test_prediction),
            "test_predictions": test_prediction.tolist(),
            "test_predicted_crisis_events": (test_prediction >= crisis_threshold)
            .astype(int)
            .tolist(),
            "latest_predicted_drawdown": latest_prediction,
            "latest_predicted_crisis": latest_prediction >= crisis_threshold,
            "feature_importance": {
                feature: float(value)
                for feature, value in zip(selected_features, importance, strict=True)
            },
            "metadata": config.xgboost.__dict__,
        }

    if selected_model in {"dnn", "both"}:
        dnn = train_dnn_regressor(
            train_x,
            train_y,
            validation_data=(validation_x, validation_y),
            config=config.dnn,
        )
        test_prediction = np.maximum(predict_dnn(dnn, test_x), 0.0)
        latest_prediction = float(np.maximum(predict_dnn(dnn, latest_features), 0.0)[0])
        model_results["dnn"] = {
            "metrics": evaluate_regression(test_y, test_prediction),
            "test_predictions": test_prediction.tolist(),
            "test_predicted_crisis_events": (test_prediction >= crisis_threshold)
            .astype(int)
            .tolist(),
            "latest_predicted_drawdown": latest_prediction,
            "latest_predicted_crisis": latest_prediction >= crisis_threshold,
            "training": {
                "best_epoch": dnn.best_epoch,
                "train_losses": dnn.train_losses,
                "validation_losses": dnn.validation_losses,
            },
            "metadata": config.dnn.__dict__,
        }

    return {
        "ticker": ticker,
        "start_date": str(pd.Timestamp(cleaned["Date"].iloc[0]).date()),
        "end_date": str(pd.Timestamp(cleaned["Date"].iloc[-1]).date()),
        "crisis_threshold": float(crisis_threshold),
        "prediction_horizon": int(prediction_horizon),
        "target_definition": "maximum future low drawdown from current close",
        "selected_model": selected_model,
        "selected_features": selected_features,
        "feature_selection": selection.config,
        "split": split_metadata,
        "test_dates": [pd.Timestamp(value).isoformat() for value in test["Date"]],
        "actual_test_drawdowns": test_y.tolist(),
        "actual_test_crisis_events": actual_test_events,
        "market_history": {
            "dates": [pd.Timestamp(value).isoformat() for value in cleaned["Date"]],
            "close": cleaned["Close"].astype(float).tolist(),
        },
        "latest_prediction_date": latest_prediction_date,
        "models": model_results,
    }

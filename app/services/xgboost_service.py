"""Train and persist the XGBoost regression model from the supplied notebook.

The service accepts explicit training and validation arrays. It never accepts a
test fold for early stopping, so callers can reserve test data for final metrics.
Configuration follows the user-supplied code where specified and does not claim
to reproduce unverified paper hyperparameters.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor


@dataclass(frozen=True)
class XGBoostConfig:
    """Model parameters drawn from the supplied notebook or safe runtime needs."""

    n_estimators: int = 500
    learning_rate: float = 0.05
    random_state: int = 42
    early_stopping_rounds: int | None = 20
    max_depth: int = 6
    n_jobs: int = 1

    def __post_init__(self) -> None:
        """Validate settings before constructing an estimator."""
        if self.n_estimators < 1 or self.max_depth < 1 or self.n_jobs < 1:
            raise ValueError("n_estimators, max_depth, and n_jobs must be positive")
        if not 0 < self.learning_rate <= 1:
            raise ValueError("learning_rate must be greater than 0 and at most 1")
        if self.early_stopping_rounds is not None and self.early_stopping_rounds < 1:
            raise ValueError("early_stopping_rounds must be positive or None")


def _matrix(values: Any, name: str) -> np.ndarray:
    """Validate an input feature matrix without fitting transformations."""
    try:
        result = np.asarray(values, dtype=np.float32)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain numeric features") from error
    if result.ndim != 2 or result.shape[0] < 1 or result.shape[1] < 1:
        raise ValueError(f"{name} must have at least one row and one feature")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must not contain missing or infinite values")
    return result


def _target(values: Any, row_count: int, name: str) -> np.ndarray:
    """Validate a one-dimensional finite regression target."""
    result = np.asarray(values, dtype=np.float32)
    if result.ndim != 1 or result.shape[0] != row_count:
        raise ValueError(f"{name} must be one-dimensional and match feature rows")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must not contain missing or infinite values")
    return result


def train_xgboost_regressor(
    X_train: pd.DataFrame | np.ndarray,
    y_train: Any,
    *,
    validation_data: tuple[pd.DataFrame | np.ndarray, Any] | None = None,
    config: XGBoostConfig = XGBoostConfig(),
) -> XGBRegressor:
    """Fit XGBoost on training data with optional validation-only early stopping."""
    train_features = _matrix(X_train, "X_train")
    train_target = _target(y_train, train_features.shape[0], "y_train")

    if isinstance(X_train, pd.DataFrame):
        feature_names = [str(column) for column in X_train.columns]
        if len(set(feature_names)) != len(feature_names):
            raise ValueError("X_train column names must be unique")
    else:
        feature_names = None
    train_input = (
        pd.DataFrame(train_features, columns=feature_names)
        if feature_names is not None
        else train_features
    )

    validation_features = None
    validation_target = None
    if validation_data is not None:
        validation_features = _matrix(validation_data[0], "X_validation")
        validation_target = _target(
            validation_data[1], validation_features.shape[0], "y_validation"
        )
        if validation_features.shape[1] != train_features.shape[1]:
            raise ValueError(
                "validation feature count must match training feature count"
            )
        if isinstance(X_train, pd.DataFrame) and isinstance(
            validation_data[0], pd.DataFrame
        ):
            if list(validation_data[0].columns) != list(X_train.columns):
                raise ValueError(
                    "validation columns must match training columns in order"
                )

    model = XGBRegressor(
        n_estimators=config.n_estimators,
        learning_rate=config.learning_rate,
        random_state=config.random_state,
        verbosity=0,
        max_depth=config.max_depth,
        n_jobs=config.n_jobs,
        early_stopping_rounds=(
            config.early_stopping_rounds if validation_data is not None else None
        ),
    )
    fit_args: dict[str, Any] = {}
    if validation_features is not None and validation_target is not None:
        validation_input = (
            pd.DataFrame(validation_features, columns=feature_names)
            if feature_names is not None
            else validation_features
        )
        fit_args["eval_set"] = [(validation_input, validation_target)]
        fit_args["verbose"] = False
    model.fit(train_input, train_target, **fit_args)
    return model


def save_xgboost_model(model: XGBRegressor, path: str | Path) -> Path:
    """Persist a fitted estimator to an explicit caller-provided path."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, destination)
    return destination


def load_xgboost_model(path: str | Path) -> XGBRegressor:
    """Load a previously saved XGBoost estimator from an explicit path."""
    model = joblib.load(Path(path))
    if not isinstance(model, XGBRegressor):
        raise TypeError("saved artifact is not an XGBRegressor")
    return model


def xgboost_metadata(config: XGBoostConfig) -> dict[str, Any]:
    """Return JSON-compatible model configuration for report persistence."""
    return asdict(config)

"""Hybrid feature selection using the supplied RFE and Boruta design.

Both selectors are fit only on the training arrays passed by the caller. The
notebook's intersection-first, union-fallback rule is retained. The approach is
a project implementation choice from user-supplied prior work, not a verified
claim about the original paper's exact HFS algorithm.
"""

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd
from boruta import BorutaPy
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
from xgboost import XGBRegressor


@dataclass(frozen=True)
class FeatureSelectionResult:
    """Selected names and audit metadata from the two training-only selectors."""

    selected_features: tuple[str, ...]
    rfe_features: tuple[str, ...]
    boruta_features: tuple[str, ...]
    intersection: tuple[str, ...]
    union: tuple[str, ...]
    config: dict[str, Any]


def _numeric_matrix(values: Any, label: str) -> np.ndarray:
    """Convert a feature table to a finite two-dimensional numeric matrix."""
    try:
        matrix = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must contain only numeric values") from error
    if matrix.ndim != 2 or matrix.shape[0] < 2 or matrix.shape[1] < 1:
        raise ValueError(f"{label} must have at least two rows and one feature")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{label} must not contain missing or infinite values")
    return matrix


def select_features_hybrid(
    X_train: pd.DataFrame | np.ndarray,
    y_train: Sequence[float] | np.ndarray | pd.Series,
    feature_names: Sequence[str] | None = None,
    *,
    rfe_estimator: Any | None = None,
    n_features_to_select: int | None = None,
    boruta_estimator: Any | None = None,
    boruta_max_iter: int = 100,
    random_seed: int = 42,
) -> FeatureSelectionResult:
    """Fit XGBoost-backed RFE and Boruta on training rows and select features.

    If ``feature_names`` is omitted, DataFrame column names are used or numeric
    arrays receive stable ``feature_<index>`` names. The default RFE count keeps
    approximately twice the square root of the input width, bounded by the
    number of available features. If the intersection is empty, the union is
    selected as in the supplied notebook.
    """
    matrix = _numeric_matrix(X_train, "X_train")
    target = np.asarray(y_train, dtype=float)
    if target.ndim != 1 or target.shape[0] != matrix.shape[0]:
        raise ValueError("y_train must be one-dimensional and match X_train rows")
    if not np.isfinite(target).all():
        raise ValueError("y_train must not contain missing or infinite values")
    if isinstance(boruta_max_iter, bool) or not isinstance(boruta_max_iter, int):
        raise ValueError("boruta_max_iter must be a positive integer")
    if boruta_max_iter < 1:
        raise ValueError("boruta_max_iter must be a positive integer")

    if feature_names is None:
        names = (
            [str(name) for name in X_train.columns]
            if isinstance(X_train, pd.DataFrame)
            else [f"feature_{index}" for index in range(matrix.shape[1])]
        )
    else:
        names = [str(name) for name in feature_names]
    if len(names) != matrix.shape[1] or len(set(names)) != len(names):
        raise ValueError("feature_names must be unique and match X_train columns")

    if n_features_to_select is None:
        selected_count = min(matrix.shape[1], max(5, int(np.sqrt(len(names)) * 2)))
    else:
        if isinstance(n_features_to_select, bool) or not isinstance(
            n_features_to_select, int
        ):
            raise ValueError("n_features_to_select must be a positive integer")
        if not 1 <= n_features_to_select <= matrix.shape[1]:
            raise ValueError("n_features_to_select must not exceed feature count")
        selected_count = n_features_to_select

    rfe_model = (
        XGBRegressor(
            n_estimators=200,
            random_state=random_seed,
            verbosity=0,
            n_jobs=1,
        )
        if rfe_estimator is None
        else rfe_estimator
    )
    rfe = RFE(
        estimator=rfe_model,
        n_features_to_select=selected_count,
        step=0.1,
    )
    rfe.fit(matrix, target)
    rfe_features = tuple(sorted(np.asarray(names)[rfe.support_].tolist()))

    boruta_model = (
        RandomForestRegressor(
            n_jobs=-1,
            random_state=random_seed,
            n_estimators=500,
        )
        if boruta_estimator is None
        else boruta_estimator
    )
    boruta = BorutaPy(
        estimator=boruta_model,
        n_estimators="auto",
        verbose=0,
        random_state=random_seed,
        max_iter=boruta_max_iter,
    )
    boruta.fit(matrix, target)
    boruta_features = tuple(sorted(np.asarray(names)[boruta.support_].tolist()))

    intersection = tuple(sorted(set(rfe_features).intersection(boruta_features)))
    union = tuple(sorted(set(rfe_features).union(boruta_features)))
    selected = intersection or union

    return FeatureSelectionResult(
        selected_features=selected,
        rfe_features=rfe_features,
        boruta_features=boruta_features,
        intersection=intersection,
        union=union,
        config={
            "rfe_n_features_to_select": selected_count,
            "rfe_step": 0.1,
            "rfe_estimator": type(rfe_model).__name__,
            "boruta_estimator": type(boruta_model).__name__,
            "boruta_max_iter": boruta_max_iter,
            "random_seed": random_seed,
            "selection_rule": "intersection_fallback_union",
        },
    )

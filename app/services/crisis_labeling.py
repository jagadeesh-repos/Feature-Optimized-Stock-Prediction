"""Create forward drawdown targets for supervised stock-crisis experiments.

The threshold and look-ahead horizon are required arguments because the
research paper's exact per-stock labeling rule has not been verified. This
implementation defines an explicit project target: the largest fall from the
current close to any daily low in the next ``horizon`` observed sessions.
It is a configurable implementation choice, not a claim of paper reproduction.
"""

from numbers import Integral, Real

import pandas as pd

FORWARD_DRAWDOWN_COLUMN = "future_drawdown"
CRISIS_EVENT_COLUMN = "is_crisis"


def create_crisis_targets(
    data: pd.DataFrame,
    *,
    prediction_horizon: int,
    crisis_threshold: float,
    close_column: str = "Close",
    low_column: str = "Low",
) -> pd.DataFrame:
    """Append continuous drawdown and threshold-event targets to OHLCV rows.

    The horizon counts subsequent rows (observed trading sessions), excluding
    the current row. The largest future low is compared with the current close.
    Drawdowns are floored at zero, making the regression target non-negative.
    The final ``prediction_horizon`` rows have unknown targets and are retained
    as nullable values so callers can separate trainable history from live data.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    if data.empty:
        raise ValueError("data must contain at least one observation")
    if isinstance(prediction_horizon, bool) or not isinstance(
        prediction_horizon, Integral
    ):
        raise ValueError("prediction_horizon must be a positive integer")
    if prediction_horizon < 1:
        raise ValueError("prediction_horizon must be a positive integer")
    if isinstance(crisis_threshold, bool) or not isinstance(crisis_threshold, Real):
        raise ValueError("crisis_threshold must be a number between 0 and 1")
    if not 0 < crisis_threshold < 1:
        raise ValueError("crisis_threshold must be greater than 0 and less than 1")

    required = {"Date", close_column, low_column}
    missing_columns = required.difference(data.columns)
    if missing_columns:
        raise ValueError(
            "data is missing required columns: " + ", ".join(sorted(missing_columns))
        )

    dates = pd.to_datetime(data["Date"], errors="coerce")
    if (
        dates.isna().any()
        or not dates.is_monotonic_increasing
        or dates.duplicated().any()
    ):
        raise ValueError("data must have valid, strictly increasing unique dates")

    close = pd.to_numeric(data[close_column], errors="coerce")
    low = pd.to_numeric(data[low_column], errors="coerce")
    if close.isna().any() or low.isna().any():
        raise ValueError("close and low prices must be complete numeric values")
    if (close <= 0).any() or (low <= 0).any():
        raise ValueError("close and low prices must be positive")

    future_lows = pd.concat(
        [low.shift(-step).rename(step) for step in range(1, prediction_horizon + 1)],
        axis=1,
    ).min(axis=1, skipna=False)
    future_drawdown = ((close - future_lows) / close).clip(lower=0)
    crisis_event = (future_drawdown >= crisis_threshold).astype("Int8")
    unknown_target = future_lows.isna()
    future_drawdown = future_drawdown.mask(unknown_target)
    crisis_event = crisis_event.mask(unknown_target, pd.NA)

    result = data.copy()
    result[FORWARD_DRAWDOWN_COLUMN] = future_drawdown
    result[CRISIS_EVENT_COLUMN] = crisis_event
    return result

"""Tests for configurable forward-drawdown targets and unavailable labels."""

import pandas as pd
import pytest

from app.services.crisis_labeling import (
    CRISIS_EVENT_COLUMN,
    FORWARD_DRAWDOWN_COLUMN,
    create_crisis_targets,
)


def _bars() -> pd.DataFrame:
    """Create chronological OHLCV rows with deliberately known future lows."""
    return pd.DataFrame(
        {
            "Date": pd.date_range("2024-01-01", periods=5, freq="B"),
            "Close": [100.0, 100.0, 100.0, 100.0, 100.0],
            "Low": [99.0, 95.0, 88.0, 97.0, 96.0],
        }
    )


def test_target_uses_only_subsequent_rows_and_marks_threshold_events():
    """The current bar is excluded and equality with the threshold is a crisis."""
    result = create_crisis_targets(_bars(), prediction_horizon=2, crisis_threshold=0.12)

    assert result.loc[0, FORWARD_DRAWDOWN_COLUMN] == pytest.approx(0.12)
    assert result.loc[0, CRISIS_EVENT_COLUMN] == 1
    assert result.loc[1, FORWARD_DRAWDOWN_COLUMN] == pytest.approx(0.12)
    assert result.loc[2, FORWARD_DRAWDOWN_COLUMN] == pytest.approx(0.04)
    assert result.loc[2, CRISIS_EVENT_COLUMN] == 0


def test_incomplete_forward_windows_are_unknown_not_negative_examples():
    """Rows without a full future window remain nullable, not normal-class labels."""
    result = create_crisis_targets(_bars(), prediction_horizon=2, crisis_threshold=0.10)

    assert result[FORWARD_DRAWDOWN_COLUMN].iloc[-2:].isna().all()
    assert result[CRISIS_EVENT_COLUMN].iloc[-2:].isna().all()


def test_positive_future_lows_without_a_drop_have_zero_drawdown():
    """A rally in the future window does not produce a negative drawdown target."""
    data = _bars().iloc[:3].copy()
    data.loc[data.index[1:], "Low"] = [105.0, 110.0]

    result = create_crisis_targets(data, prediction_horizon=2, crisis_threshold=0.10)

    assert result.loc[0, FORWARD_DRAWDOWN_COLUMN] == 0.0
    assert result.loc[0, CRISIS_EVENT_COLUMN] == 0


@pytest.mark.parametrize(
    ("prediction_horizon", "crisis_threshold"),
    [(0, 0.1), (1.5, 0.1), (True, 0.1), (1, 0.0), (1, 1.0)],
)
def test_invalid_target_parameters_are_rejected(prediction_horizon, crisis_threshold):
    """The API does not silently substitute illustrative research settings."""
    with pytest.raises(ValueError):
        create_crisis_targets(
            _bars(),
            prediction_horizon=prediction_horizon,
            crisis_threshold=crisis_threshold,
        )


def test_unsorted_or_duplicate_dates_are_rejected():
    """Forward shifts require an unambiguous chronological order."""
    data = _bars().iloc[::-1]

    with pytest.raises(ValueError, match="strictly increasing unique dates"):
        create_crisis_targets(data, prediction_horizon=2, crisis_threshold=0.1)

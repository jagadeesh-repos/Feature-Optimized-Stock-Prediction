"""Tests for indicator formulas, warm-up behavior, and absence of future leakage."""

import numpy as np
import pandas as pd
import pytest

from app.services.feature_engineering import FeatureWindows, engineer_technical_features


def _ohlcv(periods: int = 12) -> pd.DataFrame:
    """Build deterministic bars with changing prices and nonzero volume."""
    return pd.DataFrame(
        {
            "Date": pd.date_range("2024-01-01", periods=periods, freq="B"),
            "Close": np.arange(100.0, 100.0 + periods),
            "Volume": np.arange(1000.0, 1000.0 + periods * 10, 10.0),
        }
    )


def test_sma_returns_and_volume_change_use_expected_observations():
    """Rolling and percentage features match their documented definitions."""
    windows = FeatureWindows(
        sma=(3,),
        ema=(3,),
        macd_fast=2,
        macd_slow=4,
        macd_signal=2,
        rsi=3,
        volatility=3,
        momentum=(1, 2),
    )
    result = engineer_technical_features(_ohlcv(), windows)

    assert result.loc[2, "sma_3"] == pytest.approx((100 + 101 + 102) / 3)
    assert result.loc[2, "return_1"] == pytest.approx(102 / 101 - 1)
    assert result.loc[2, "return_2"] == pytest.approx(102 / 100 - 1)
    assert result.loc[2, "volume_change_1"] == pytest.approx(10 / 1010)
    assert pd.isna(result.loc[0, "sma_3"])


def test_rsi_is_bounded_and_handles_no_loss_series():
    """A monotonic rise has RSI 100 after the configured warm-up period."""
    windows = FeatureWindows(
        sma=(2,),
        ema=(2,),
        macd_fast=2,
        macd_slow=3,
        macd_signal=2,
        rsi=2,
        volatility=2,
        momentum=(1,),
    )
    result = engineer_technical_features(_ohlcv(6), windows)
    rsi = result["rsi_2"].dropna()

    assert not rsi.empty
    assert rsi.between(0, 100).all()
    assert (rsi == 100).all()


def test_features_at_an_earlier_date_do_not_depend_on_future_bars():
    """Changing later prices leaves all features at earlier dates unchanged."""
    windows = FeatureWindows(
        sma=(3,),
        ema=(3,),
        macd_fast=2,
        macd_slow=4,
        macd_signal=2,
        rsi=3,
        volatility=3,
        momentum=(1, 2),
    )
    original = _ohlcv()
    changed = original.copy()
    changed.loc[8:, "Close"] *= 3
    changed.loc[8:, "Volume"] *= 4

    original_features = engineer_technical_features(original, windows)
    changed_features = engineer_technical_features(changed, windows)
    feature_columns = original_features.columns.difference(original.columns)

    pd.testing.assert_frame_equal(
        original_features.loc[:7, feature_columns],
        changed_features.loc[:7, feature_columns],
    )


def test_invalid_macd_order_is_rejected():
    """The MACD fast period must precede the slow period."""
    with pytest.raises(ValueError, match="macd_fast must be smaller"):
        FeatureWindows(macd_fast=26, macd_slow=12)

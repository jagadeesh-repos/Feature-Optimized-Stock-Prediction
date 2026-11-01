"""Build causal technical indicators from cleaned OHLCV history.

All indicators use the current and prior observations only. The default windows
are conventional implementation defaults, not verified settings from the cited
paper. Warm-up values remain missing so callers can drop them for training
without introducing backward-looking or forward-looking imputations.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureWindows:
    """Window configuration for technical features (not paper-derived)."""

    sma: tuple[int, ...] = (5, 20, 50)
    ema: tuple[int, ...] = (5, 20, 50)
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    rsi: int = 14
    volatility: int = 20
    momentum: tuple[int, ...] = (1, 5, 21)

    def __post_init__(self) -> None:
        """Reject invalid windows at configuration construction time."""
        windows = (
            *self.sma,
            *self.ema,
            self.macd_fast,
            self.macd_slow,
            self.macd_signal,
            self.rsi,
            self.volatility,
            *self.momentum,
        )
        if any(
            isinstance(window, bool) or not isinstance(window, int) or window < 1
            for window in windows
        ):
            raise ValueError("all feature windows must be positive integers")
        if self.macd_fast >= self.macd_slow:
            raise ValueError("macd_fast must be smaller than macd_slow")


def engineer_technical_features(
    data: pd.DataFrame,
    windows: FeatureWindows = FeatureWindows(),
) -> pd.DataFrame:
    """Append moving averages, MACD, RSI, volatility, volume and return features.

    The input is copied and returned with original columns intact. A target
    column may be present; this function never reads it when computing features.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    if data.empty:
        raise ValueError("data must contain at least one observation")
    missing_columns = {"Close", "Volume"}.difference(data.columns)
    if missing_columns:
        raise ValueError(
            "data is missing required columns: " + ", ".join(sorted(missing_columns))
        )

    close = pd.to_numeric(data["Close"], errors="coerce")
    volume = pd.to_numeric(data["Volume"], errors="coerce")
    if close.isna().any() or volume.isna().any() or (close <= 0).any():
        raise ValueError("Close and Volume must contain valid numeric values")
    if not close.index.is_monotonic_increasing:
        raise ValueError("data must be in chronological index order")

    result = data.copy()
    returns = close.pct_change(fill_method=None)

    for window in windows.sma:
        result[f"sma_{window}"] = close.rolling(
            window=window, min_periods=window
        ).mean()
    for window in windows.ema:
        result[f"ema_{window}"] = close.ewm(
            span=window, adjust=False, min_periods=window
        ).mean()

    fast_ema = close.ewm(
        span=windows.macd_fast, adjust=False, min_periods=windows.macd_fast
    ).mean()
    slow_ema = close.ewm(
        span=windows.macd_slow, adjust=False, min_periods=windows.macd_slow
    ).mean()
    result["macd"] = fast_ema - slow_ema
    result["macd_signal"] = (
        result["macd"]
        .ewm(
            span=windows.macd_signal,
            adjust=False,
            min_periods=windows.macd_signal,
        )
        .mean()
    )
    result["macd_histogram"] = result["macd"] - result["macd_signal"]

    price_change = close.diff()
    gains = price_change.clip(lower=0)
    losses = -price_change.clip(upper=0)
    average_gain = gains.ewm(
        alpha=1 / windows.rsi, adjust=False, min_periods=windows.rsi
    ).mean()
    average_loss = losses.ewm(
        alpha=1 / windows.rsi, adjust=False, min_periods=windows.rsi
    ).mean()
    relative_strength = average_gain / average_loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + relative_strength)
    rsi = rsi.mask((average_loss == 0) & (average_gain > 0), 100.0)
    rsi = rsi.mask((average_loss == 0) & (average_gain == 0), 50.0)
    result[f"rsi_{windows.rsi}"] = rsi

    result[f"volatility_{windows.volatility}"] = returns.rolling(
        window=windows.volatility, min_periods=windows.volatility
    ).std(ddof=1)
    result["volume_change_1"] = volume.pct_change(fill_method=None).replace(
        [np.inf, -np.inf], np.nan
    )
    for period in windows.momentum:
        result[f"return_{period}"] = close.pct_change(periods=period, fill_method=None)

    return result

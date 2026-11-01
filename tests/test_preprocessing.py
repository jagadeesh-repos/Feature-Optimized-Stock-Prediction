"""Unit tests for chronology and data-quality handling before modeling."""

import pandas as pd
import pytest

from app.services.data_collection import MarketDataError
from app.services.preprocessing import preprocess_ohlcv


def _frame() -> pd.DataFrame:
    """Create a small canonical OHLCV fixture with plausible price bars."""
    return pd.DataFrame(
        {
            "Date": ["2024-01-03", "2024-01-02", "2024-01-02"],
            "Open": [10.0, 9.0, 9.5],
            "High": [12.0, 11.0, 11.5],
            "Low": [8.0, 7.0, 7.5],
            "Close": [11.0, 10.0, 10.5],
            "Volume": [100.0, 90.0, 95.0],
        }
    )


def test_preprocess_sorts_and_keeps_last_duplicate_date():
    """Output is chronological and duplicate records resolve deterministically."""
    result = preprocess_ohlcv(_frame())

    assert result["Date"].dt.strftime("%Y-%m-%d").tolist() == [
        "2024-01-02",
        "2024-01-03",
    ]
    assert result.iloc[0]["Open"] == 9.5


def test_preprocess_drops_incomplete_rows_without_filling_gaps():
    """Incomplete records are omitted; missing sessions remain missing."""
    frame = _frame().iloc[[0, 1]].copy()
    frame.loc[0, "Close"] = None

    result = preprocess_ohlcv(frame)

    assert result["Date"].dt.strftime("%Y-%m-%d").tolist() == ["2024-01-02"]


def test_preprocess_rejects_missing_columns():
    """A mislabeled or partial source frame cannot enter the model pipeline."""
    with pytest.raises(MarketDataError, match="missing columns"):
        preprocess_ohlcv(pd.DataFrame({"Date": ["2024-01-02"], "Close": [1.0]}))


@pytest.mark.parametrize("field,value", [("Open", 0.0), ("Volume", -1.0)])
def test_preprocess_rejects_invalid_numeric_values(field, value):
    """Non-positive prices and negative volume are not treated as valid bars."""
    frame = _frame().iloc[[0]].copy()
    frame.loc[frame.index[0], field] = value

    with pytest.raises(MarketDataError):
        preprocess_ohlcv(frame)


def test_preprocess_rejects_inconsistent_price_range():
    """Each bar must satisfy Low <= Open/Close <= High."""
    frame = _frame().iloc[[0]].copy()
    frame.loc[frame.index[0], "High"] = 9.0

    with pytest.raises(MarketDataError, match="inconsistent high"):
        preprocess_ohlcv(frame)

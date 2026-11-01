"""Offline unit tests for historical OHLCV collection and validation."""

from datetime import date

import pandas as pd
import pytest

from app.services.data_collection import MarketDataError, collect_historical_data


class FakeProvider:
    """Capture provider arguments while returning a deterministic fixture."""

    def __init__(self, data: pd.DataFrame | Exception):
        self.data = data
        self.calls = []

    def download(self, ticker: str, **kwargs):
        self.calls.append((ticker, kwargs))
        if isinstance(self.data, Exception):
            raise self.data
        return self.data.copy()


def _ohlcv(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Build plausible synthetic bars without relying on a live market feed."""
    return pd.DataFrame(
        {
            ("Open", "RELIANCE.NS"): [100.0] * len(index),
            ("High", "RELIANCE.NS"): [105.0] * len(index),
            ("Low", "RELIANCE.NS"): [95.0] * len(index),
            ("Close", "RELIANCE.NS"): [102.0] * len(index),
            ("Volume", "RELIANCE.NS"): [1000.0] * len(index),
        },
        index=index,
    )


def test_collect_normalizes_multiindex_and_includes_end_date():
    """Provider dates are filtered inclusively and returned in stable order."""
    index = pd.to_datetime(["2024-01-03", "2024-01-02", "2024-01-02"])
    provider = FakeProvider(_ohlcv(index))

    result = collect_historical_data(
        " RELIANCE.NS ", "2024-01-02", date(2024, 1, 3), provider
    )

    assert result.columns.tolist() == ["Date", "Open", "High", "Low", "Close", "Volume"]
    assert result["Date"].dt.strftime("%Y-%m-%d").tolist() == [
        "2024-01-02",
        "2024-01-03",
    ]
    assert provider.calls[0][0] == "RELIANCE.NS"
    assert provider.calls[0][1]["end"] == "2024-01-04"
    assert result.iloc[0]["Close"] == 102.0


def test_incomplete_rows_are_dropped_without_filling_missing_dates():
    """A broken session is omitted, not synthesized or forward-filled."""
    frame = _ohlcv(pd.to_datetime(["2024-01-02", "2024-01-04"]))
    frame.iloc[1, frame.columns.get_loc(("Close", "RELIANCE.NS"))] = None

    result = collect_historical_data(
        "RELIANCE.NS", "2024-01-02", "2024-01-04", FakeProvider(frame)
    )

    assert result["Date"].dt.strftime("%Y-%m-%d").tolist() == ["2024-01-02"]


def test_empty_provider_result_is_an_error_not_fabricated_data():
    """Unavailable provider data must never become sample prices."""
    provider = FakeProvider(pd.DataFrame())

    with pytest.raises(MarketDataError, match="No historical data"):
        collect_historical_data("RELIANCE.NS", "2024-01-01", "2024-01-05", provider)


def test_provider_failure_is_reported_without_leaking_provider_details():
    """Network/provider errors are wrapped in a stable application exception."""
    provider = FakeProvider(ConnectionError("private provider detail"))

    with pytest.raises(MarketDataError, match="request failed") as error:
        collect_historical_data("RELIANCE.NS", "2024-01-01", "2024-01-05", provider)
    assert "private provider detail" not in str(error.value)


@pytest.mark.parametrize(
    ("ticker", "start_date", "end_date"),
    [
        ("", "2024-01-01", "2024-01-02"),
        ("BAD TICKER", "2024-01-01", "2024-01-02"),
        ("RELIANCE.NS", "2024-01-03", "2024-01-02"),
        ("RELIANCE.NS", "not-a-date", "2024-01-02"),
    ],
)
def test_invalid_inputs_are_rejected_before_provider_call(ticker, start_date, end_date):
    """Bad identifiers and date ranges never trigger external work."""
    provider = FakeProvider(pd.DataFrame())

    with pytest.raises(ValueError):
        collect_historical_data(ticker, start_date, end_date, provider)
    assert provider.calls == []


def test_invalid_market_bars_are_rejected():
    """Impossible OHLC relationships are treated as invalid source data."""
    frame = _ohlcv(pd.to_datetime(["2024-01-02"]))
    frame.iloc[0, frame.columns.get_loc(("High", "RELIANCE.NS"))] = 90.0

    with pytest.raises(MarketDataError, match="inconsistent high"):
        collect_historical_data(
            "RELIANCE.NS", "2024-01-02", "2024-01-02", FakeProvider(frame)
        )

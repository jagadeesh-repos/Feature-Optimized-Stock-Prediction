"""Download and validate historical OHLCV data from a configurable provider.

The provider boundary keeps network access out of unit tests and allows a
different market-data source to be configured without changing downstream code.
Missing observations are not forward-filled: only returned trading sessions are
kept, and rows with incomplete OHLCV values are discarded rather than invented.
"""

from datetime import date, datetime, timedelta
from typing import Any

import pandas as pd

OHLCV_FIELDS = ("Open", "High", "Low", "Close", "Volume")
PRICE_FIELDS = ("Open", "High", "Low", "Close")


class MarketDataError(RuntimeError):
    """Raised when market data cannot be retrieved or safely validated."""


def _as_date(value: date | str, name: str) -> date:
    """Parse an ISO date or date object and report invalid inputs clearly."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError as error:
            raise ValueError(f"{name} must use YYYY-MM-DD format") from error
    raise ValueError(f"{name} must be a date or YYYY-MM-DD string")


def _matching_column(columns: pd.Index, field: str, ticker: str) -> Any:
    """Find an OHLCV column in either flat or yfinance MultiIndex output."""
    matches = []
    for column in columns:
        labels = column if isinstance(column, tuple) else (column,)
        normalized = tuple(str(label).strip() for label in labels)
        if any(label.casefold() == field.casefold() for label in normalized):
            matches.append((column, normalized))

    if not matches:
        raise MarketDataError(f"Provider response is missing the {field} column")
    if len(matches) > 1:
        ticker_matches = [
            item
            for item in matches
            if any(label.casefold() == ticker.casefold() for label in item[1])
        ]
        matches = ticker_matches or matches
    if len(matches) != 1:
        raise MarketDataError(f"Provider response has ambiguous {field} columns")
    return matches[0][0]


def _date_column(columns: pd.Index) -> Any | None:
    """Locate a date column when a provider returns dates as data, not index."""
    for column in columns:
        labels = column if isinstance(column, tuple) else (column,)
        if any(
            str(label).strip().casefold() in {"date", "datetime"} for label in labels
        ):
            return column
    return None


def _normalize_ohlcv(raw_data: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Return a flat Date/Open/High/Low/Close/Volume frame from provider output."""
    normalized = pd.DataFrame(index=raw_data.index)
    for field in OHLCV_FIELDS:
        column = _matching_column(raw_data.columns, field, ticker)
        values = raw_data.loc[:, column]
        if isinstance(values, pd.DataFrame):
            values = values.iloc[:, 0]
        normalized[field] = pd.to_numeric(values, errors="coerce")

    date_column = _date_column(raw_data.columns)
    date_values = (
        raw_data.loc[:, date_column] if date_column is not None else raw_data.index
    )
    dates = pd.to_datetime(
        pd.Series(date_values, index=raw_data.index), errors="coerce"
    )
    if isinstance(dates.dtype, pd.DatetimeTZDtype):
        dates = dates.dt.tz_localize(None)
    normalized.insert(0, "Date", dates)
    return normalized.reset_index(drop=True)


def _yfinance_provider() -> Any:
    """Load yfinance only when a live provider is actually requested."""
    try:
        import yfinance
    except ImportError as error:
        raise MarketDataError(
            "yfinance is required for the default market-data provider"
        ) from error
    return yfinance


def collect_historical_data(
    ticker: str,
    start_date: date | str,
    end_date: date | str,
    provider: Any | None = None,
) -> pd.DataFrame:
    """Fetch and validate daily OHLCV rows for an inclusive date range.

    Yahoo Finance treats its ``end`` argument as exclusive, so the provider
    request advances it by one calendar day and this function filters the result
    back to the caller's inclusive range. ``provider`` may be any object with a
    compatible ``download`` method; tests use a deterministic fake provider.
    """
    ticker = ticker.strip() if isinstance(ticker, str) else ""
    if (
        not ticker
        or len(ticker) > 32
        or any(character.isspace() for character in ticker)
    ):
        raise ValueError("ticker must be a non-empty symbol of at most 32 characters")

    start = _as_date(start_date, "start_date")
    end = _as_date(end_date, "end_date")
    if start > end:
        raise ValueError("start_date must be on or before end_date")

    source = provider if provider is not None else _yfinance_provider()
    download = getattr(source, "download", source)
    if not callable(download):
        raise TypeError("provider must be callable or expose a download method")

    try:
        raw_data = download(
            ticker,
            start=start.isoformat(),
            end=(end + timedelta(days=1)).isoformat(),
            auto_adjust=False,
            progress=False,
            threads=False,
        )
    except Exception as error:
        raise MarketDataError("Market data provider request failed") from error

    if not isinstance(raw_data, pd.DataFrame) or raw_data.empty:
        raise MarketDataError(f"No historical data was returned for {ticker}")

    data = _normalize_ohlcv(raw_data, ticker)
    data = data.dropna(subset=["Date", *OHLCV_FIELDS])
    start_timestamp = pd.Timestamp(start)
    end_timestamp = pd.Timestamp(end)
    data = data.loc[data["Date"].between(start_timestamp, end_timestamp)].copy()
    data = data.drop_duplicates(subset="Date", keep="last").sort_values("Date")

    if data.empty:
        raise MarketDataError(f"No valid observations were returned for {ticker}")
    if not (data.loc[:, PRICE_FIELDS] > 0).all().all():
        raise MarketDataError("Provider response contains non-positive prices")
    if (data["Volume"] < 0).any():
        raise MarketDataError("Provider response contains negative volume")
    if (data["High"] < data[["Open", "Close", "Low"]].max(axis=1)).any():
        raise MarketDataError("Provider response contains inconsistent high prices")
    if (data["Low"] > data[["Open", "Close", "High"]].min(axis=1)).any():
        raise MarketDataError("Provider response contains inconsistent low prices")

    return data.loc[:, ["Date", *OHLCV_FIELDS]].reset_index(drop=True)

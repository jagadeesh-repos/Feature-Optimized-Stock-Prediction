"""Validate and prepare canonical OHLCV frames for downstream modeling.

This module does not fill non-trading days or impute prices. It removes rows
whose dates or OHLCV values are incomplete, then checks the remaining bars for
basic market-data consistency while retaining chronological order.
"""

import pandas as pd

from app.services.data_collection import OHLCV_FIELDS, PRICE_FIELDS, MarketDataError

REQUIRED_COLUMNS = ("Date", *OHLCV_FIELDS)


def preprocess_ohlcv(data: pd.DataFrame) -> pd.DataFrame:
    """Return a validated, sorted OHLCV frame with one row per trading date.

    Duplicate dates keep the last supplied observation. Missing and
    non-numeric fields are not synthesized; incomplete rows are dropped. A
    frame with no valid observations or impossible price bars is rejected.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    if data.empty:
        raise MarketDataError("OHLCV data is empty")

    missing_columns = set(REQUIRED_COLUMNS).difference(data.columns)
    if missing_columns:
        missing_list = ", ".join(sorted(missing_columns))
        raise MarketDataError(f"OHLCV data is missing columns: {missing_list}")

    prepared = data.loc[:, REQUIRED_COLUMNS].copy()
    prepared["Date"] = pd.to_datetime(prepared["Date"], errors="coerce")
    if isinstance(prepared["Date"].dtype, pd.DatetimeTZDtype):
        prepared["Date"] = prepared["Date"].dt.tz_localize(None)

    for field in OHLCV_FIELDS:
        prepared[field] = pd.to_numeric(prepared[field], errors="coerce")
    prepared.loc[:, OHLCV_FIELDS] = prepared.loc[:, OHLCV_FIELDS].replace(
        [float("inf"), float("-inf")], pd.NA
    )
    prepared = prepared.dropna(subset=REQUIRED_COLUMNS)
    prepared = prepared.drop_duplicates(subset="Date", keep="last")
    prepared = prepared.sort_values("Date").reset_index(drop=True)

    if prepared.empty:
        raise MarketDataError("OHLCV data contains no complete observations")
    if not (prepared.loc[:, PRICE_FIELDS] > 0).all().all():
        raise MarketDataError("OHLCV data contains non-positive prices")
    if (prepared["Volume"] < 0).any():
        raise MarketDataError("OHLCV data contains negative volume")
    if (prepared["High"] < prepared[["Open", "Close", "Low"]].max(axis=1)).any():
        raise MarketDataError("OHLCV data contains inconsistent high prices")
    if (prepared["Low"] > prepared[["Open", "Close", "High"]].min(axis=1)).any():
        raise MarketDataError("OHLCV data contains inconsistent low prices")

    return prepared.loc[:, REQUIRED_COLUMNS]

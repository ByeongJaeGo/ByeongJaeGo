"""Market data helpers (Yahoo Finance)."""

from __future__ import annotations

import pandas as pd
import yfinance as yf


def normalize_kr_ticker(ticker: str) -> str:
    """
    Accept 005930, 005930.KS, 035420.KQ etc.
    Bare 6-digit codes default to .KS (KOSPI). Use .KQ for KOSDAQ.
    """
    t = ticker.strip().upper()
    if t.endswith(".KS") or t.endswith(".KQ"):
        return t
    if t.isdigit() and len(t) == 6:
        return f"{t}.KS"
    return t


def download_ohlcv(
    ticker: str,
    start: str | None = "2018-01-01",
    end: str | None = None,
) -> pd.DataFrame:
    """Download daily OHLCV and flatten MultiIndex columns if present."""
    symbol = normalize_kr_ticker(ticker)
    raw = yf.download(
        symbol,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if raw is None or raw.empty:
        raise ValueError(f"No data for {symbol}")

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [c[0] for c in raw.columns]

    needed = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in needed if c not in raw.columns]
    if missing:
        raise ValueError(f"{symbol} missing columns: {missing}")

    df = raw[needed].dropna(how="any").copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df.sort_index()
    return df

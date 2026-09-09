"""Technical indicators: multi-period Bollinger %b and MFI."""

from __future__ import annotations

import numpy as np
import pandas as pd


def bollinger_percent_b(
    close: pd.Series,
    period: int = 20,
    num_std: float = 2.0,
) -> pd.Series:
    middle = close.rolling(period, min_periods=period).mean()
    std = close.rolling(period, min_periods=period).std(ddof=0)
    upper = middle + num_std * std
    lower = middle - num_std * std
    width = upper - lower
    pb = np.where(width == 0, np.nan, (close - lower) / width)
    return pd.Series(pb, index=close.index, name=f"percent_b_{period}")


def bollinger_bands(
    close: pd.Series,
    period: int = 20,
    num_std: float = 2.0,
) -> pd.DataFrame:
    middle = close.rolling(period, min_periods=period).mean()
    std = close.rolling(period, min_periods=period).std(ddof=0)
    upper = middle + num_std * std
    lower = middle - num_std * std
    width = upper - lower
    percent_b = np.where(width == 0, np.nan, (close - lower) / width)
    return pd.DataFrame(
        {
            "bb_middle": middle,
            "bb_upper": upper,
            "bb_lower": lower,
            "percent_b": percent_b,
        },
        index=close.index,
    )


def money_flow_index(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    period: int = 14,
) -> pd.Series:
    """Volume-weighted RSI (Money Flow Index), range 0–100."""
    typical_price = (high + low + close) / 3.0
    raw_money_flow = typical_price * volume

    delta = typical_price.diff()
    positive_flow = raw_money_flow.where(delta > 0, 0.0)
    negative_flow = raw_money_flow.where(delta < 0, 0.0)
    positive_flow = positive_flow.where(delta != 0, 0.0)
    negative_flow = negative_flow.where(delta != 0, 0.0)

    pos_sum = positive_flow.rolling(period, min_periods=period).sum()
    neg_sum = negative_flow.rolling(period, min_periods=period).sum()

    mfr = np.where(neg_sum == 0, np.inf, pos_sum / neg_sum)
    mfi = 100.0 - (100.0 / (1.0 + mfr))
    mfi = np.where(np.isinf(mfr), 100.0, mfi)
    return pd.Series(mfi, index=close.index, name=f"mfi_{period}")


def weekly_mfi(df: pd.DataFrame, period: int = 6) -> pd.Series:
    """Compute MFI on weekly (Fri) bars and forward-fill onto daily index."""
    weekly = (
        df.resample("W-FRI")
        .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})
        .dropna(how="any")
    )
    w = money_flow_index(
        weekly["High"], weekly["Low"], weekly["Close"], weekly["Volume"], period=period
    )
    w.name = f"weekly_mfi_{period}"
    return w.reindex(df.index, method="ffill")


def add_indicators(
    df: pd.DataFrame,
    bb_period: int = 20,
    bb_std: float = 2.0,
    mfi_period: int = 14,
    sma_fast: int = 50,
    sma_slow: int = 200,
) -> pd.DataFrame:
    """Legacy helper used by the original single-%b strategy."""
    out = df.copy()
    bands = bollinger_bands(out["Close"], period=bb_period, num_std=bb_std)
    out = pd.concat([out, bands], axis=1)
    out["mfi"] = money_flow_index(
        out["High"], out["Low"], out["Close"], out["Volume"], period=mfi_period
    )
    out["sma_fast"] = out["Close"].rolling(sma_fast, min_periods=sma_fast).mean()
    out["sma_slow"] = out["Close"].rolling(sma_slow, min_periods=sma_slow).mean()
    return out


def add_v2_indicators(df: pd.DataFrame, bb_std: float = 2.0) -> pd.DataFrame:
    """
    Indicators for the top100 rule set:
      - daily MFI(6), MFI(13)
      - weekly MFI(6)
      - %b(8), %b(14), %b(20)
    """
    out = df.copy()
    out["mfi_6"] = money_flow_index(out["High"], out["Low"], out["Close"], out["Volume"], 6)
    out["mfi_13"] = money_flow_index(out["High"], out["Low"], out["Close"], out["Volume"], 13)
    out["weekly_mfi_6"] = weekly_mfi(out, period=6)
    for period in (8, 14, 20):
        out[f"percent_b_{period}"] = bollinger_percent_b(out["Close"], period=period, num_std=bb_std)
    out["pb_lt0_count"] = (
        (out["percent_b_8"] < 0).astype(int)
        + (out["percent_b_14"] < 0).astype(int)
        + (out["percent_b_20"] < 0).astype(int)
    )
    # Keep a representative %b/mfi for exit heuristics
    out["percent_b"] = out["percent_b_20"]
    out["mfi"] = out["mfi_13"]
    return out

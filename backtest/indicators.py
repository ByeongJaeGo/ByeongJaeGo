"""Technical indicators: Bollinger %b and Money Flow Index (MFI)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def bollinger_bands(
    close: pd.Series,
    period: int = 20,
    num_std: float = 2.0,
) -> pd.DataFrame:
    """Return middle / upper / lower Bollinger Bands and %b."""
    middle = close.rolling(period, min_periods=period).mean()
    std = close.rolling(period, min_periods=period).std(ddof=0)
    upper = middle + num_std * std
    lower = middle - num_std * std
    band_width = upper - lower
    percent_b = np.where(band_width == 0, np.nan, (close - lower) / band_width)
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
    """Volume-weighted RSI (Money Flow Index)."""
    typical_price = (high + low + close) / 3.0
    raw_money_flow = typical_price * volume

    delta = typical_price.diff()
    positive_flow = raw_money_flow.where(delta > 0, 0.0)
    negative_flow = raw_money_flow.where(delta < 0, 0.0)
    # Unchanged TP: exclude from both sides
    positive_flow = positive_flow.where(delta != 0, 0.0)
    negative_flow = negative_flow.where(delta != 0, 0.0)

    pos_sum = positive_flow.rolling(period, min_periods=period).sum()
    neg_sum = negative_flow.rolling(period, min_periods=period).sum()

    mfr = np.where(neg_sum == 0, np.inf, pos_sum / neg_sum)
    mfi = 100.0 - (100.0 / (1.0 + mfr))
    mfi = np.where(np.isinf(mfr), 100.0, mfi)
    return pd.Series(mfi, index=close.index, name="mfi")


def add_indicators(
    df: pd.DataFrame,
    bb_period: int = 20,
    bb_std: float = 2.0,
    mfi_period: int = 14,
    sma_fast: int = 50,
    sma_slow: int = 200,
) -> pd.DataFrame:
    """Attach %b, MFI, and trend SMAs to OHLCV data."""
    out = df.copy()
    bands = bollinger_bands(out["Close"], period=bb_period, num_std=bb_std)
    out = pd.concat([out, bands], axis=1)
    out["mfi"] = money_flow_index(
        out["High"], out["Low"], out["Close"], out["Volume"], period=mfi_period
    )
    out["sma_fast"] = out["Close"].rolling(sma_fast, min_periods=sma_fast).mean()
    out["sma_slow"] = out["Close"].rolling(sma_slow, min_periods=sma_slow).mean()
    return out

"""Unit tests for indicators (no network)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from backtest.indicators import add_indicators, bollinger_bands, money_flow_index
from backtest.strategy import StrategyParams, generate_signals


def _synthetic_ohlcv(n: int = 300, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2019-01-01", periods=n)
    close = 100 + np.cumsum(rng.normal(0, 1.2, size=n))
    high = close + rng.uniform(0.2, 1.5, size=n)
    low = close - rng.uniform(0.2, 1.5, size=n)
    open_ = close + rng.normal(0, 0.4, size=n)
    volume = rng.integers(100_000, 500_000, size=n).astype(float)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=idx,
    )


def test_percent_b_bounds_typical():
    df = _synthetic_ohlcv()
    bands = bollinger_bands(df["Close"])
    valid = bands["percent_b"].dropna()
    assert len(valid) > 200
    # Most values should sit roughly in [-0.5, 1.5]
    assert valid.between(-1, 2).mean() > 0.95


def test_mfi_range():
    df = _synthetic_ohlcv()
    mfi = money_flow_index(df["High"], df["Low"], df["Close"], df["Volume"])
    valid = mfi.dropna()
    assert valid.min() >= 0
    assert valid.max() <= 100


def test_signals_columns_exist():
    df = add_indicators(_synthetic_ohlcv())
    sig = generate_signals(df, StrategyParams(require_trend_filter=False))
    for col in ("buy", "sell", "entry_ready", "sell_zone", "percent_b", "mfi"):
        assert col in sig.columns

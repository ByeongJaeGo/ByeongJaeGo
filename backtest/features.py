"""Bollinger bandwidth and multi-feature indicator pack."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .indicators import bollinger_percent_b, money_flow_index, weekly_mfi


def add_swing_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Features for swing optimization:
      MFI(6/13/14), %b(8/14/20), BB(20,2), bandwidth, squeeze flags.
    """
    out = df.copy()
    close = out["Close"]

    # Bollinger 20,2
    mid = close.rolling(20, min_periods=20).mean()
    std = close.rolling(20, min_periods=20).std(ddof=0)
    upper = mid + 2.0 * std
    lower = mid - 2.0 * std
    width = upper - lower
    bw = np.where(mid == 0, np.nan, width / mid)

    out["bb_mid"] = mid
    out["bb_upper"] = upper
    out["bb_lower"] = lower
    out["bandwidth"] = bw
    # Percentile rank of bandwidth over trailing 120 sessions (vectorized)
    bw_s = pd.Series(bw, index=out.index)
    out["bw_pct"] = bw_s.rolling(120, min_periods=40).apply(
        lambda x: np.mean(x <= x[-1]), raw=True
    )
    out["bw_low20"] = pd.Series(bw, index=out.index).rolling(20, min_periods=20).min()
    out["squeeze"] = out["bandwidth"] <= out["bw_low20"] * 1.05
    # bandwidth expanding vs prior bar
    out["bw_expand"] = out["bandwidth"] > out["bandwidth"].shift(1)

    for p in (8, 14, 20):
        out[f"pb_{p}"] = bollinger_percent_b(close, period=p, num_std=2.0)
    out["pb"] = out["pb_20"]
    out["pb_up"] = out["pb"] > out["pb"].shift(1)
    out["pb_dn"] = out["pb"] < out["pb"].shift(1)

    for p in (6, 13, 14):
        out[f"mfi_{p}"] = money_flow_index(
            out["High"], out["Low"], out["Close"], out["Volume"], period=p
        )
    out["mfi"] = out["mfi_14"]
    out["mfi_up"] = out["mfi_14"] > out["mfi_14"].shift(1)
    out["weekly_mfi_6"] = weekly_mfi(out, period=6)

    out["above_mid"] = close > mid
    out["break_upper"] = close > upper
    out["break_lower"] = close < lower
    out["ret1"] = close.pct_change()
    return out

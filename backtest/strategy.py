"""MFI + Bollinger %b mean-reversion swing signals (daily)."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class StrategyParams:
    """Entry/exit thresholds for undervalued → overheated swing."""

    buy_percent_b: float = 0.0
    buy_mfi: float = 20.0
    sell_percent_b: float = 0.85
    sell_mfi: float = 80.0
    exit_percent_b: float = 0.80
    require_trend_filter: bool = True
    # If True, only buy when price is not in a hard downtrend
    # (Close >= SMA50 OR SMA50 >= SMA200). Skip when both fail.
    stop_below_entry_low: bool = True


def _bullish_reversal(row: pd.Series, prev: pd.Series) -> bool:
    """Confirm bounce: %b turns up or MFI crosses back above 20."""
    pb_turn = pd.notna(row["percent_b"]) and pd.notna(prev["percent_b"])
    pb_turn = pb_turn and row["percent_b"] > prev["percent_b"]

    mfi_cross = (
        pd.notna(row["mfi"])
        and pd.notna(prev["mfi"])
        and prev["mfi"] <= 20.0
        and row["mfi"] > 20.0
    )
    return bool(pb_turn or mfi_cross)


def _trend_ok(row: pd.Series) -> bool:
    """Avoid hard downtrends: allow if above SMA50 or SMA50 >= SMA200."""
    close = row["Close"]
    sma_fast = row.get("sma_fast")
    sma_slow = row.get("sma_slow")

    above_fast = pd.notna(sma_fast) and close >= sma_fast
    if above_fast:
        return True
    if pd.notna(sma_fast) and pd.notna(sma_slow) and sma_fast >= sma_slow:
        return True
    # Not enough SMA history yet — allow (warmup handled elsewhere)
    if pd.isna(sma_fast) or pd.isna(sma_slow):
        return True
    return False


def _mfi_bullish_divergence(window: pd.DataFrame) -> bool:
    """Price lower low, MFI higher low over last ~10 bars with two troughs."""
    if len(window) < 8:
        return False
    closes = window["Close"]
    mfis = window["mfi"]
    if mfis.isna().any() or closes.isna().any():
        return False

    mid = len(window) // 2
    left_price_low = closes.iloc[:mid].min()
    right_price_low = closes.iloc[mid:].min()
    left_mfi_low = mfis.iloc[:mid].min()
    right_mfi_low = mfis.iloc[mid:].min()
    return right_price_low < left_price_low and right_mfi_low > left_mfi_low


def _mfi_bearish_divergence(window: pd.DataFrame) -> bool:
    if len(window) < 8:
        return False
    closes = window["Close"]
    mfis = window["mfi"]
    if mfis.isna().any() or closes.isna().any():
        return False

    mid = len(window) // 2
    left_price_high = closes.iloc[:mid].max()
    right_price_high = closes.iloc[mid:].max()
    left_mfi_high = mfis.iloc[:mid].max()
    right_mfi_high = mfis.iloc[mid:].max()
    return right_price_high > left_price_high and right_mfi_high < left_mfi_high


def generate_signals(df: pd.DataFrame, params: StrategyParams | None = None) -> pd.DataFrame:
    """
    Add signal columns:
      - entry_ready: undervalued zone (%b + MFI)
      - buy: entry_ready + reversal confirm (+ optional trend)
      - sell_zone: overheated zone
      - sell: sell_zone + rollover / %b drop
    """
    params = params or StrategyParams()
    out = df.copy()
    out["entry_ready"] = False
    out["buy"] = False
    out["sell_zone"] = False
    out["sell"] = False
    out["bull_div"] = False
    out["bear_div"] = False

    for i in range(1, len(out)):
        row = out.iloc[i]
        prev = out.iloc[i - 1]

        if pd.isna(row["percent_b"]) or pd.isna(row["mfi"]):
            continue

        entry_ready = row["percent_b"] <= params.buy_percent_b and row["mfi"] <= params.buy_mfi
        out.iat[i, out.columns.get_loc("entry_ready")] = entry_ready

        lookback = out.iloc[max(0, i - 9) : i + 1]
        bull_div = _mfi_bullish_divergence(lookback)
        bear_div = _mfi_bearish_divergence(lookback)
        out.iat[i, out.columns.get_loc("bull_div")] = bull_div
        out.iat[i, out.columns.get_loc("bear_div")] = bear_div

        trend_ok = (not params.require_trend_filter) or _trend_ok(row)
        if entry_ready and _bullish_reversal(row, prev) and trend_ok:
            out.iat[i, out.columns.get_loc("buy")] = True

        sell_zone = row["percent_b"] >= params.sell_percent_b and row["mfi"] >= params.sell_mfi
        out.iat[i, out.columns.get_loc("sell_zone")] = sell_zone

        mfi_roll = (
            pd.notna(prev["mfi"])
            and prev["mfi"] >= params.sell_mfi
            and row["mfi"] < params.sell_mfi
        )
        pb_drop = (
            pd.notna(prev["percent_b"])
            and prev["percent_b"] >= 1.0
            and row["percent_b"] < params.exit_percent_b
        )
        # Sell when overheated zone rolls over, or %b falls off the upper band
        recent_hot = out.iloc[max(0, i - 5) : i]["sell_zone"].any()
        if (sell_zone and (mfi_roll or bear_div)) or (recent_hot and mfi_roll) or pb_drop:
            out.iat[i, out.columns.get_loc("sell")] = True

    return out

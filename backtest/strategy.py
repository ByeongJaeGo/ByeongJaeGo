"""Signal rules: legacy mean-reversion + top100 MFI/%b confluence."""

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
    stop_below_entry_low: bool = True


@dataclass(frozen=True)
class Top100Params:
    """
    Daily: MFI(6) and MFI(13) below `daily_mfi_max`.
      Request text was "<0", but standard MFI is [0,100] so literal <0 never fires.
      Default 20 = near-zero oversold zone (operable).
    Weekly: MFI(6) has not reached 60 (weekly_mfi_6 < weekly_mfi_max).
    %b: at least `min_pb_lt0` of periods {8,14,20} have %b < 0.
    """

    daily_mfi_max: float = 20.0
    weekly_mfi_max: float = 60.0
    min_pb_lt0: int = 2
    # Exit when weekly MFI becomes overheated or daily %b recovers hard
    sell_weekly_mfi: float = 60.0
    sell_pb_ge: float = 0.85
    sell_mfi_ge: float = 80.0
    stop_below_entry_low: bool = True


def _bullish_reversal(row: pd.Series, prev: pd.Series) -> bool:
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
    close = row["Close"]
    sma_fast = row.get("sma_fast")
    sma_slow = row.get("sma_slow")

    above_fast = pd.notna(sma_fast) and close >= sma_fast
    if above_fast:
        return True
    if pd.notna(sma_fast) and pd.notna(sma_slow) and sma_fast >= sma_slow:
        return True
    if pd.isna(sma_fast) or pd.isna(sma_slow):
        return True
    return False


def _mfi_bullish_divergence(window: pd.DataFrame) -> bool:
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
        recent_hot = out.iloc[max(0, i - 5) : i]["sell_zone"].any()
        if (sell_zone and (mfi_roll or bear_div)) or (recent_hot and mfi_roll) or pb_drop:
            out.iat[i, out.columns.get_loc("sell")] = True

    return out


def generate_top100_signals(df: pd.DataFrame, params: Top100Params | None = None) -> pd.DataFrame:
    """
    Buy when ALL hold on daily bar:
      - mfi_6 < daily_mfi_max AND mfi_13 < daily_mfi_max  (user: < 0)
      - weekly_mfi_6 < weekly_mfi_max                      (user: not ≥ 60)
      - pb_lt0_count >= min_pb_lt0                         (≥2 of %b 8/14/20 < 0)

    Sell when weekly MFI overheats, or daily MFI(13)+%b20 overheated rollover.
    """
    params = params or Top100Params()
    out = df.copy()
    out["entry_ready"] = False
    out["buy"] = False
    out["sell_zone"] = False
    out["sell"] = False

    needed = ["mfi_6", "mfi_13", "weekly_mfi_6", "pb_lt0_count", "percent_b_20", "mfi_13"]
    for col in needed:
        if col not in out.columns:
            raise KeyError(f"missing indicator column: {col}")

    for i in range(1, len(out)):
        row = out.iloc[i]
        prev = out.iloc[i - 1]

        if (
            pd.isna(row["mfi_6"])
            or pd.isna(row["mfi_13"])
            or pd.isna(row["weekly_mfi_6"])
            or pd.isna(row["pb_lt0_count"])
        ):
            continue

        # Standard MFI is [0,100]. User wrote "<0"; threshold 0 uses floor (<=0).
        # Operable oversold default is daily_mfi_max=20 (near-zero zone).
        if params.daily_mfi_max <= 0:
            daily_ok = row["mfi_6"] <= 0 and row["mfi_13"] <= 0
        else:
            daily_ok = (
                row["mfi_6"] < params.daily_mfi_max and row["mfi_13"] < params.daily_mfi_max
            )

        weekly_ok = row["weekly_mfi_6"] < params.weekly_mfi_max
        pb_ok = int(row["pb_lt0_count"]) >= params.min_pb_lt0
        entry_ready = bool(daily_ok and weekly_ok and pb_ok)
        out.iat[i, out.columns.get_loc("entry_ready")] = entry_ready

        # Enter on the setup day (confluence already is the filter)
        if entry_ready:
            out.iat[i, out.columns.get_loc("buy")] = True

        sell_zone = (
            pd.notna(row["percent_b_20"])
            and pd.notna(row["mfi_13"])
            and row["percent_b_20"] >= params.sell_pb_ge
            and row["mfi_13"] >= params.sell_mfi_ge
        )
        out.iat[i, out.columns.get_loc("sell_zone")] = sell_zone

        weekly_hot = row["weekly_mfi_6"] >= params.sell_weekly_mfi
        mfi_roll = (
            pd.notna(prev["mfi_13"])
            and prev["mfi_13"] >= params.sell_mfi_ge
            and row["mfi_13"] < params.sell_mfi_ge
        )
        pb_recover = (
            pd.notna(row["percent_b_20"])
            and pd.notna(prev["percent_b_20"])
            and prev["percent_b_20"] >= 1.0
            and row["percent_b_20"] < 0.8
        )
        if weekly_hot or (sell_zone and mfi_roll) or pb_recover or mfi_roll:
            # Avoid selling on the same bar we just bought unless weekly already hot
            if weekly_hot or sell_zone or pb_recover:
                out.iat[i, out.columns.get_loc("sell")] = True

    # Prevent buy+sell same bar: prefer buy only if not already hot weekly
    same = out["buy"] & out["sell"]
    out.loc[same, "sell"] = False
    return out

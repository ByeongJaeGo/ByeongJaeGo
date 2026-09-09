"""Rule definitions and fast trade simulation for swing optimization."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RuleSet:
    name: str
    # entry style: mean_rev | squeeze_break | band_walk
    style: str
    mfi_period: int = 14
    mfi_buy_max: float = 20.0
    mfi_buy_min: float = 0.0
    pb_buy_max: float = 0.0
    pb_periods_lt0_min: int = 0  # of {8,14,20}
    require_squeeze: bool = False
    bw_pct_max: float = 1.0  # enter only if bandwidth percentile <= this
    require_pb_up: bool = False
    require_mfi_up: bool = False
    require_break_upper: bool = False
    weekly_mfi_max: float = 100.0
    # exits
    pb_sell_min: float = 0.85
    mfi_sell_min: float = 80.0
    exit_to_mid: bool = False
    stop_pct: float = 0.07
    max_hold: int = 20
    trail_pb_drop: float = 0.15  # from peak %b


def entry_mask(df: pd.DataFrame, rule: RuleSet) -> pd.Series:
    mfi = df[f"mfi_{rule.mfi_period}"]
    pb = df["pb_20"]
    ok = pd.Series(True, index=df.index)

    if rule.style == "mean_rev":
        ok &= mfi <= rule.mfi_buy_max
        ok &= mfi >= rule.mfi_buy_min
        ok &= pb <= rule.pb_buy_max
        if rule.pb_periods_lt0_min > 0:
            cnt = (
                (df["pb_8"] < 0).astype(int)
                + (df["pb_14"] < 0).astype(int)
                + (df["pb_20"] < 0).astype(int)
            )
            ok &= cnt >= rule.pb_periods_lt0_min
        if rule.require_squeeze:
            ok &= df["squeeze"].fillna(False)
        if rule.bw_pct_max < 1.0:
            ok &= df["bw_pct"] <= rule.bw_pct_max
        if rule.require_pb_up:
            ok &= df["pb_up"].fillna(False)
        if rule.require_mfi_up:
            ok &= df["mfi_up"].fillna(False)
        ok &= df["weekly_mfi_6"] < rule.weekly_mfi_max

    elif rule.style == "squeeze_break":
        # squeeze recently, then expansion + break upper / %b>0.8 + mfi strength
        recent_sq = df["squeeze"].rolling(5, min_periods=1).max().astype(bool)
        ok &= recent_sq
        ok &= df["bw_expand"].fillna(False)
        if rule.require_break_upper:
            ok &= df["break_upper"].fillna(False)
        else:
            ok &= pb >= 0.8
        ok &= mfi >= rule.mfi_buy_min
        if rule.require_mfi_up:
            ok &= df["mfi_up"].fillna(False)
        if rule.bw_pct_max < 1.0:
            # after squeeze, bandwidth still not extremely wide
            ok &= df["bw_pct"] <= rule.bw_pct_max

    elif rule.style == "band_walk":
        ok &= pb >= rule.pb_buy_max  # here pb_buy_max used as minimum %b
        ok &= mfi >= rule.mfi_buy_min
        ok &= df["above_mid"].fillna(False)
        ok &= df["bw_expand"].fillna(False)
        if rule.require_mfi_up:
            ok &= df["mfi_up"].fillna(False)

    else:
        raise ValueError(rule.style)

    return ok.fillna(False)


def simulate_trades(df: pd.DataFrame, rule: RuleSet) -> list[dict[str, Any]]:
    """Long-only next-open entries; exit next open after signal/stop/timeout."""
    buys = entry_mask(df, rule).to_numpy()
    close = df["Close"].to_numpy(dtype=float)
    open_ = df["Open"].to_numpy(dtype=float)
    low = df["Low"].to_numpy(dtype=float)
    high = df["High"].to_numpy(dtype=float)
    pb = df["pb_20"].to_numpy(dtype=float)
    mfi = df[f"mfi_{rule.mfi_period}"].to_numpy(dtype=float)
    mid = df["bb_mid"].to_numpy(dtype=float)
    idx = df.index

    trades: list[dict[str, Any]] = []
    i = 1
    n = len(df)
    while i < n - 1:
        if not buys[i]:
            i += 1
            continue
        entry_i = i + 1
        entry_px = open_[entry_i] if np.isfinite(open_[entry_i]) else close[entry_i]
        if not np.isfinite(entry_px) or entry_px <= 0:
            i += 1
            continue
        stop_px = entry_px * (1.0 - rule.stop_pct)
        peak_pb = pb[entry_i] if np.isfinite(pb[entry_i]) else 0.0
        exit_i = None
        reason = "eod"
        for j in range(entry_i, min(entry_i + rule.max_hold, n - 1)):
            if np.isfinite(pb[j]):
                peak_pb = max(peak_pb, pb[j])
            # stop on close
            if close[j] < stop_px:
                exit_i = j + 1
                reason = "stop"
                break
            sell = False
            if np.isfinite(pb[j]) and np.isfinite(mfi[j]):
                if pb[j] >= rule.pb_sell_min and mfi[j] >= rule.mfi_sell_min:
                    sell = True
                    reason = "overheat"
                if rule.exit_to_mid and np.isfinite(mid[j]) and close[j] >= mid[j] and j > entry_i:
                    sell = True
                    reason = "mid"
                if peak_pb - pb[j] >= rule.trail_pb_drop and peak_pb >= 0.7:
                    sell = True
                    reason = "trail"
            if sell:
                exit_i = j + 1
                break
        if exit_i is None:
            exit_i = min(entry_i + rule.max_hold, n - 1)
            reason = "timeout" if exit_i < n - 1 or entry_i + rule.max_hold <= n - 1 else "eod"
            if exit_i == n - 1:
                reason = "eod"

        exit_px = open_[exit_i] if np.isfinite(open_[exit_i]) else close[exit_i]
        if exit_i == n - 1:
            exit_px = close[exit_i]
        ret = exit_px / entry_px - 1.0
        trades.append(
            {
                "entry_date": idx[entry_i],
                "exit_date": idx[exit_i],
                "entry_px": float(entry_px),
                "exit_px": float(exit_px),
                "return": float(ret),
                "bars": int(exit_i - entry_i),
                "reason": reason,
            }
        )
        i = exit_i  # no overlapping trades per name
    return trades


def portfolio_monthly_returns(
    all_trades: list[dict[str, Any]],
    *,
    max_positions: int = 5,
    start: str | None = None,
    end: str | None = None,
) -> pd.Series:
    """
    Capital split equally across up to `max_positions` concurrent trades.
    Approximate daily PnL from trade span using linear contribution of trade return
    over holding days, then compound to monthly.
    """
    if not all_trades:
        return pd.Series(dtype=float)

    # Build event list
    rows = []
    for t in all_trades:
        rows.append(t)
    tdf = pd.DataFrame(rows).sort_values("entry_date")
    if start:
        tdf = tdf[tdf["entry_date"] >= pd.Timestamp(start)]
    if end:
        tdf = tdf[tdf["exit_date"] <= pd.Timestamp(end)]
    if tdf.empty:
        return pd.Series(dtype=float)

    # Greedy portfolio: accept trade if open positions < max
    open_pos: list[dict[str, Any]] = []
    accepted: list[dict[str, Any]] = []
    for _, t in tdf.iterrows():
        open_pos = [p for p in open_pos if p["exit_date"] > t["entry_date"]]
        if len(open_pos) >= max_positions:
            continue
        accepted.append(t.to_dict())
        open_pos.append(t.to_dict())

    if not accepted:
        return pd.Series(dtype=float)

    # Daily equity: each day, each open trade contributes (total_return / bars) for that day
    # More accurate: mark equal weight of capital to each slot
    day_index = pd.bdate_range(
        min(a["entry_date"] for a in accepted),
        max(a["exit_date"] for a in accepted),
    )
    daily_ret = pd.Series(0.0, index=day_index)
    slot_w = 1.0 / max_positions

    for a in accepted:
        bars = max(int(a["bars"]), 1)
        # distribute arithmetic return across bars (approx)
        per_day = (1.0 + float(a["return"])) ** (1.0 / bars) - 1.0
        span = pd.bdate_range(a["entry_date"], a["exit_date"])
        # exclude exit day overnight? include until day before exit
        span = span[:-1] if len(span) > 1 else span
        for d in span:
            if d in daily_ret.index:
                daily_ret.loc[d] += slot_w * per_day
        # idle capital in empty slots earns 0

    equity = (1.0 + daily_ret).cumprod()
    monthly = equity.resample("ME").last().pct_change().dropna()
    return monthly


def trade_stats(trades: list[dict[str, Any]]) -> dict[str, Any]:
    if not trades:
        return {
            "trades": 0,
            "win_rate": None,
            "avg_return": None,
            "median_return": None,
            "total_compound": 0.0,
        }
    rets = np.array([t["return"] for t in trades], dtype=float)
    eq = np.cumprod(1.0 + rets)
    return {
        "trades": int(len(rets)),
        "win_rate": float((rets > 0).mean() * 100.0),
        "avg_return": float(rets.mean() * 100.0),
        "median_return": float(np.median(rets) * 100.0),
        "total_compound": float((eq[-1] - 1.0) * 100.0),
        "best": float(rets.max() * 100.0),
        "worst": float(rets.min() * 100.0),
    }

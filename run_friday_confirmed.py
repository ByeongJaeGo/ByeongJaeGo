#!/usr/bin/env python3
"""Friday daily-chart weekly trading with confirmation grades A/B/C."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tabulate import tabulate

sys.path.insert(0, str(Path(__file__).resolve().parent))

from backtest.features import add_swing_features
from backtest.indicators import weekly_mfi
from backtest.optimize_rules import trade_stats
from run_optimize_monthly import monthly_stats
from run_weekly_best import WeeklyRule, entry_mask


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    dn = (-delta).clip(lower=0.0)
    au = up.ewm(alpha=1 / n, min_periods=n, adjust=False).mean()
    ad = dn.ewm(alpha=1 / n, min_periods=n, adjust=False).mean()
    return 100.0 - (100.0 / (1.0 + au / ad.replace(0, np.nan)))


def friday_enriched(daily: pd.DataFrame) -> pd.DataFrame:
    feat = add_swing_features(daily)
    feat["rsi14"] = rsi(feat["Close"], 14)
    feat["vol_ratio"] = feat["Volume"] / feat["Volume"].rolling(20).mean()
    feat["sma50"] = feat["Close"].rolling(50).mean()
    feat["sma200"] = feat["Close"].rolling(200).mean()
    feat["weekly_mfi6"] = weekly_mfi(feat, 6)
    feat["hard_down"] = (feat["Close"] < feat["sma50"]) & (feat["sma50"] < feat["sma200"])
    fri = feat.loc[feat.index.dayofweek == 4].copy()
    fri["pb"] = feat.loc[fri.index, "pb_20"].astype(float)
    fri["mfi_14"] = feat.loc[fri.index, "mfi_14"].astype(float)
    fri["pb_up"] = fri["pb"] > fri["pb"].shift(1)
    fri["mfi_up"] = fri["mfi_14"] > fri["mfi_14"].shift(1)
    return fri


RULE = WeeklyRule(
    name="confirmed",
    style="mean_rev",
    mfi_period=14,
    mfi_buy_max=30.0,
    pb_buy_max=0.15,
    stop_pct=0.10,
    max_hold=2,
    exit_to_mid=False,
    pb_sell_min=0.85,
    mfi_sell_min=70.0,
)


def grade_mask(df: pd.DataFrame, grade: str) -> pd.Series:
    base = entry_mask(df, RULE)
    if grade == "C":
        return base
    vol = df["vol_ratio"] >= 1.5
    if grade == "B":
        return base & vol.fillna(False)
    # A: full confirmation
    return (
        base
        & vol.fillna(False)
        & (df["rsi14"] <= 30).fillna(False)
        & (df["weekly_mfi6"] < 60).fillna(False)
        & (~df["hard_down"].fillna(False))
    )


def simulate(df: pd.DataFrame, buys: pd.Series) -> list[dict]:
    close = df["Close"].to_numpy(float)
    open_ = df["Open"].to_numpy(float)
    pb = df["pb"].to_numpy(float)
    mfi = df["mfi_14"].to_numpy(float)
    idx = df.index
    n = len(df)
    buy = buys.fillna(False).to_numpy()
    trades: list[dict] = []
    i = 1
    while i < n - 1:
        if not buy[i]:
            i += 1
            continue
        entry_i = i + 1
        entry_px = open_[entry_i] if np.isfinite(open_[entry_i]) else close[entry_i]
        if not np.isfinite(entry_px) or entry_px <= 0:
            i += 1
            continue
        stop_px = entry_px * 0.90
        exit_i = None
        reason = "eod"
        for j in range(entry_i, min(entry_i + 2, n - 1)):
            if close[j] < stop_px:
                exit_i, reason = j + 1, "stop"
                break
            if np.isfinite(pb[j]) and np.isfinite(mfi[j]) and pb[j] >= 0.85 and mfi[j] >= 70:
                exit_i, reason = j + 1, "overheat"
                break
        if exit_i is None:
            exit_i = min(entry_i + 2, n - 1)
            reason = "timeout" if exit_i < n - 1 else "eod"
        exit_px = open_[exit_i] if np.isfinite(open_[exit_i]) else close[exit_i]
        if exit_i == n - 1:
            exit_px = close[exit_i]
        trades.append(
            {
                "entry_date": idx[entry_i],
                "exit_date": idx[exit_i],
                "return": float(min(max(exit_px / entry_px - 1.0, -0.5), 1.0)),
                "bars": int(exit_i - entry_i),
                "reason": reason,
            }
        )
        i = exit_i
    return trades


def portfolio_monthly(trades: list[dict], positions: int = 1) -> pd.Series:
    if not trades:
        return pd.Series(dtype=float)
    tdf = pd.DataFrame(trades).sort_values("entry_date")
    open_pos: list[dict] = []
    accepted: list[dict] = []
    for _, t in tdf.iterrows():
        open_pos = [p for p in open_pos if p["exit_date"] > t["entry_date"]]
        if len(open_pos) >= positions:
            continue
        accepted.append(t.to_dict())
        open_pos.append(t.to_dict())
    if not accepted:
        return pd.Series(dtype=float)
    weeks = pd.date_range(
        min(a["entry_date"] for a in accepted),
        max(a["exit_date"] for a in accepted),
        freq="W-FRI",
    )
    wr = pd.Series(0.0, index=weeks)
    slot = 1.0 / positions
    for a in accepted:
        bars = max(int(a["bars"]), 1)
        per = (1.0 + float(a["return"])) ** (1.0 / bars) - 1.0
        for d in pd.date_range(a["entry_date"], periods=bars, freq="W-FRI"):
            if d in wr.index:
                wr.loc[d] += slot * per
    return (1.0 + wr).cumprod().resample("ME").last().pct_change().dropna()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--grade", choices=["A", "B", "C"], default="A")
    p.add_argument("--positions", type=int, default=1)
    p.add_argument("--start", default="2019-01-01")
    p.add_argument("--cache", default="cache_ohlcv_v2")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--compare", action="store_true", help="Print A/B/C side by side")
    args = p.parse_args()

    universe = pd.read_csv("out_top100_universe.csv")
    if args.limit:
        universe = universe.head(args.limit)
    cache = Path(args.cache)

    packs: dict[str, pd.DataFrame] = {}
    for _, row in universe.iterrows():
        path = cache / f"{row['yahoo']}.pkl"
        if not path.exists():
            continue
        daily = pd.read_pickle(path)
        daily = daily[daily.index >= pd.Timestamp(args.start)]
        if len(daily) < 220:
            continue
        packs[row["yahoo"]] = friday_enriched(daily[["Open", "High", "Low", "Close", "Volume"]])

    grades = ["A", "B", "C"] if args.compare else [args.grade]
    for g in grades:
        trades: list[dict] = []
        for ticker, df in packs.items():
            for t in simulate(df, grade_mask(df, g)):
                t["ticker"] = ticker
                trades.append(t)
        ts = trade_stats(trades)
        ms = monthly_stats(portfolio_monthly(trades, args.positions))
        std = float(portfolio_monthly(trades, args.positions).std() * 100) if trades else None
        print(f"\n=== GRADE {g} ===")
        print(tabulate(ts.items(), headers=["trade", "value"], tablefmt="github"))
        print(tabulate(ms.items(), headers=["month", "value"], tablefmt="github"))
        if std is not None:
            print(f"monthly std%: {std:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

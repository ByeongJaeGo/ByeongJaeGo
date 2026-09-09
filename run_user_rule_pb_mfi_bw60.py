#!/usr/bin/env python3
"""
User-specified entry rule backtest (full return report).

BUY (all must hold):
  - %b(20) < 0
  - MFI(6)  <= 20
  - MFI(16) <= 20
  - Bandwidth == 60-day low (at lowest of prior 60 sessions)

Exits: several common BB/MFI exits so returns are fully reported.
Cadence: daily next-open, and Friday-only weekly.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from tabulate import tabulate

sys.path.insert(0, str(Path(__file__).resolve().parent))

from backtest.indicators import bollinger_percent_b, money_flow_index
from backtest.optimize_rules import trade_stats
from run_friday_daily_best import weekly_portfolio_returns
from backtest.optimize_rules import portfolio_monthly_returns
from run_optimize_monthly import monthly_stats


@dataclass(frozen=True)
class ExitCfg:
    name: str
    stop_pct: float
    max_hold: int
    pb_sell_min: float
    mfi_sell_min: float
    exit_to_mid: bool
    trail_pb_drop: float = 9.0  # off if large


def add_user_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    close = out["Close"]
    mid = close.rolling(20, min_periods=20).mean()
    std = close.rolling(20, min_periods=20).std(ddof=0)
    upper = mid + 2.0 * std
    lower = mid - 2.0 * std
    width = upper - lower
    bw = np.where(mid == 0, np.nan, width / mid)
    bw_s = pd.Series(bw, index=out.index)

    out["bb_mid"] = mid
    out["bb_upper"] = upper
    out["bb_lower"] = lower
    out["bandwidth"] = bw_s
    # 이전 60일 최저 (당일 포함 60봉 최저와 같으면 True)
    out["bw_low60"] = bw_s.rolling(60, min_periods=60).min()
    out["bw_at_60low"] = bw_s <= out["bw_low60"] * 1.0001

    out["pb"] = bollinger_percent_b(close, period=20, num_std=2.0)
    out["mfi_6"] = money_flow_index(out["High"], out["Low"], out["Close"], out["Volume"], 6)
    out["mfi_16"] = money_flow_index(out["High"], out["Low"], out["Close"], out["Volume"], 16)
    out["mfi_14"] = money_flow_index(out["High"], out["Low"], out["Close"], out["Volume"], 14)
    return out


def entry_mask(df: pd.DataFrame) -> pd.Series:
    return (
        (df["pb"] < 0)
        & (df["mfi_6"] <= 20)
        & (df["mfi_16"] <= 20)
        & df["bw_at_60low"].fillna(False)
    ).fillna(False)


def simulate(df: pd.DataFrame, buys: pd.Series, ex: ExitCfg) -> list[dict]:
    buy = buys.to_numpy()
    close = df["Close"].to_numpy(float)
    open_ = df["Open"].to_numpy(float)
    pb = df["pb"].to_numpy(float)
    mfi = df["mfi_14"].to_numpy(float)  # exit MFI uses 14 (standard)
    mid = df["bb_mid"].to_numpy(float)
    idx = df.index
    n = len(df)
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
        stop_px = entry_px * (1.0 - ex.stop_pct)
        peak_pb = pb[entry_i] if np.isfinite(pb[entry_i]) else 0.0
        exit_i = None
        reason = "eod"
        for j in range(entry_i, min(entry_i + ex.max_hold, n - 1)):
            if np.isfinite(pb[j]):
                peak_pb = max(peak_pb, float(pb[j]))
            if close[j] < stop_px:
                exit_i, reason = j + 1, "stop"
                break
            sell = False
            if np.isfinite(pb[j]) and np.isfinite(mfi[j]):
                if pb[j] >= ex.pb_sell_min and mfi[j] >= ex.mfi_sell_min:
                    sell, reason = True, "overheat"
                if ex.exit_to_mid and np.isfinite(mid[j]) and close[j] >= mid[j] and j > entry_i:
                    sell, reason = True, "mid"
                if ex.trail_pb_drop < 5 and peak_pb - pb[j] >= ex.trail_pb_drop and peak_pb >= 0.7:
                    sell, reason = True, "trail"
            if sell:
                exit_i = j + 1
                break
        if exit_i is None:
            exit_i = min(entry_i + ex.max_hold, n - 1)
            reason = "timeout" if exit_i < n - 1 else "eod"
        exit_px = open_[exit_i] if np.isfinite(open_[exit_i]) else close[exit_i]
        if exit_i == n - 1:
            exit_px = close[exit_i]
        ret = float(exit_px / entry_px - 1.0)
        ret = float(min(max(ret, -0.5), 1.0))
        trades.append(
            {
                "entry_date": idx[entry_i],
                "exit_date": idx[exit_i],
                "entry_px": float(entry_px),
                "exit_px": float(exit_px),
                "return": ret,
                "bars": int(exit_i - entry_i),
                "reason": reason,
            }
        )
        i = exit_i
    return trades


EXITS = [
    ExitCfg("mid_h10_s7", 0.07, 10, 0.85, 70.0, True),
    ExitCfg("mid_h20_s7", 0.07, 20, 0.85, 70.0, True),
    ExitCfg("oh_h10_s10", 0.10, 10, 0.85, 70.0, False),
    ExitCfg("oh_h20_s10", 0.10, 20, 0.85, 80.0, False),
    ExitCfg("oh80_h15_s10", 0.10, 15, 0.85, 80.0, False),
    ExitCfg("mid_oh_h15_s8", 0.08, 15, 0.85, 70.0, True),
    ExitCfg("timeout_h5_s10", 0.10, 5, 9.0, 200.0, False),  # hold only
    ExitCfg("timeout_h10_s10", 0.10, 10, 9.0, 200.0, False),
    ExitCfg("timeout_h20_s10", 0.10, 20, 9.0, 200.0, False),
    ExitCfg("fri_mid_h2_s10", 0.10, 2, 0.85, 70.0, True),  # weekly bars equiv
    ExitCfg("fri_oh_h2_s10", 0.10, 2, 0.85, 70.0, False),
    ExitCfg("fri_oh_h3_s10", 0.10, 3, 0.85, 80.0, False),
    ExitCfg("fri_mid_h3_s8", 0.08, 3, 0.85, 70.0, True),
]


def load_frames(cache: Path, universe: pd.DataFrame, start: str, limit: int | None):
    rows = universe if limit is None else universe.head(limit)
    daily_frames = []
    friday_frames = []
    for _, r in rows.iterrows():
        path = cache / f"{r['yahoo']}.pkl"
        if not path.exists():
            continue
        raw = pd.read_pickle(path)
        raw = raw[raw.index >= pd.Timestamp(start)]
        if len(raw) < 120:
            continue
        feat = add_user_features(raw[["Open", "High", "Low", "Close", "Volume"]])
        daily_frames.append((r["yahoo"], r["name"], feat))
        fri = feat.loc[feat.index.dayofweek == 4].copy()
        if len(fri) >= 40:
            friday_frames.append((r["yahoo"], r["name"], fri))
    return daily_frames, friday_frames


def run_universe(frames, exits: list[ExitCfg], mode: str, positions: int) -> pd.DataFrame:
    rows = []
    for ex in exits:
        # skip friday-named exits on daily and vice versa lightly
        if mode == "daily" and ex.name.startswith("fri_"):
            continue
        if mode == "friday" and not ex.name.startswith("fri_") and ex.max_hold > 4:
            # still allow short daily-style holds measured in weeks
            pass
        trades: list[dict] = []
        n_signals = 0
        for yahoo, name, df in frames:
            buys = entry_mask(df)
            n_signals += int(buys.sum())
            for t in simulate(df, buys, ex):
                t = dict(t)
                t["ticker"] = yahoo
                t["name"] = name
                trades.append(t)
        ts = trade_stats(trades)
        if mode == "friday":
            monthly = weekly_portfolio_returns(trades, positions)
        else:
            monthly = portfolio_monthly_returns(trades, max_positions=positions)
        ms = monthly_stats(monthly)
        # walk-forward
        split = pd.Timestamp("2022-01-01")
        train = [t for t in trades if t["entry_date"] < split]
        test = [t for t in trades if t["entry_date"] >= split]
        tr, te = trade_stats(train), trade_stats(test)
        rows.append(
            {
                "mode": mode,
                "exit": ex.name,
                "signals_raw": n_signals,
                "trades": ts["trades"],
                "win_rate": ts["win_rate"],
                "avg_return": ts["avg_return"],
                "median_return": ts["median_return"],
                "best_trade": ts.get("best"),
                "worst_trade": ts.get("worst"),
                "total_compound": ts.get("total_compound"),
                "avg_month_pct": ms.get("avg_month_pct"),
                "median_month_pct": ms.get("median_month_pct"),
                "pct_months_gt_0": ms.get("pct_months_gt_0"),
                "best_month_pct": ms.get("best_month_pct"),
                "worst_month_pct": ms.get("worst_month_pct"),
                "train_n": tr["trades"],
                "train_wr": tr["win_rate"],
                "train_avg": tr["avg_return"],
                "test_n": te["trades"],
                "test_wr": te["win_rate"],
                "test_avg": te["avg_return"],
                "avg_bars": float(np.mean([t["bars"] for t in trades])) if trades else None,
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2019-01-01")
    ap.add_argument("--cache", default="cache_ohlcv_v2")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--positions", type=int, default=1)
    ap.add_argument("--out", default="out_user_rule_pb_mfi_bw60.csv")
    args = ap.parse_args()

    universe = pd.read_csv("out_top100_universe.csv")
    daily, friday = load_frames(Path(args.cache), universe, args.start, args.limit)
    print(f"loaded daily={len(daily)} friday={len(friday)}")
    print("ENTRY: %b<0 & MFI6<=20 & MFI16<=20 & Bandwidth=60d low")

    ddf = run_universe(daily, EXITS, "daily", args.positions)
    fdf = run_universe(friday, EXITS, "friday", args.positions)
    out = pd.concat([ddf, fdf], ignore_index=True)
    out = out.sort_values(["mode", "avg_month_pct"], ascending=[True, False])
    out.to_csv(args.out, index=False)

    cols = [
        "mode", "exit", "trades", "win_rate", "avg_return", "median_return",
        "avg_month_pct", "median_month_pct", "pct_months_gt_0",
        "best_month_pct", "worst_month_pct",
        "train_wr", "train_avg", "test_wr", "test_avg", "avg_bars",
    ]
    print("\n=== DAILY (next open) ===")
    print(tabulate(out[out["mode"] == "daily"][cols], headers="keys", tablefmt="github", showindex=False, floatfmt=".2f"))
    print("\n=== FRIDAY-ONLY (weekly decision) ===")
    print(tabulate(out[out["mode"] == "friday"][cols], headers="keys", tablefmt="github", showindex=False, floatfmt=".2f"))

    best = out.dropna(subset=["avg_month_pct"]).sort_values("avg_month_pct", ascending=False).iloc[0]
    print("\n=== BEST BY AVG MONTHLY RETURN ===")
    print(best[cols].to_string())
    print(f"\nsaved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

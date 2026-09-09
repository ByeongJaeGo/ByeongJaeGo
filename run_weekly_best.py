#!/usr/bin/env python3
"""
Best once-per-week swing method using Bollinger + %b + Bandwidth + MFI.

Two decision modes:
  A) weekly bars (native weekly chart)
  B) Friday-only decisions on daily indicators

Example:
  python3 run_weekly_best.py
  python3 run_weekly_best.py --limit 50
"""

from __future__ import annotations

import argparse
import itertools
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from tabulate import tabulate

sys.path.insert(0, str(Path(__file__).resolve().parent))

from backtest.indicators import bollinger_percent_b, money_flow_index
from backtest.optimize_rules import portfolio_monthly_returns, trade_stats
from run_optimize_monthly import monthly_stats


@dataclass(frozen=True)
class WeeklyRule:
    name: str
    style: str  # mean_rev | squeeze_break | band_walk
    mfi_period: int = 6
    mfi_buy_max: float = 20.0
    mfi_buy_min: float = 50.0
    pb_buy_max: float = 0.15
    pb_buy_min: float = 0.8
    require_squeeze: bool = False
    bw_pct_max: float = 1.0
    require_mfi_up: bool = False
    require_pb_up: bool = False
    require_break_upper: bool = False
    pb_sell_min: float = 0.85
    mfi_sell_min: float = 70.0
    exit_to_mid: bool = True
    stop_pct: float = 0.08
    max_hold: int = 8  # weeks
    trail_pb_drop: float = 0.2


def to_weekly(df: pd.DataFrame) -> pd.DataFrame:
    w = (
        df.resample("W-FRI")
        .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})
        .dropna(how="any")
    )
    return w


def add_weekly_features(df: pd.DataFrame, bb_period: int = 20) -> pd.DataFrame:
    out = df.copy()
    close = out["Close"]
    mid = close.rolling(bb_period, min_periods=bb_period).mean()
    std = close.rolling(bb_period, min_periods=bb_period).std(ddof=0)
    upper = mid + 2.0 * std
    lower = mid - 2.0 * std
    width = upper - lower
    bw = np.where(mid == 0, np.nan, width / mid)
    bw_s = pd.Series(bw, index=out.index)

    out["bb_mid"] = mid
    out["bb_upper"] = upper
    out["bb_lower"] = lower
    out["bandwidth"] = bw
    out["bw_pct"] = bw_s.rolling(52, min_periods=20).apply(lambda x: np.mean(x <= x[-1]), raw=True)
    out["bw_low20"] = bw_s.rolling(20, min_periods=10).min()
    out["squeeze"] = out["bandwidth"] <= out["bw_low20"] * 1.05
    out["bw_expand"] = out["bandwidth"] > out["bandwidth"].shift(1)

    out["pb"] = bollinger_percent_b(close, period=bb_period, num_std=2.0)
    out["pb_up"] = out["pb"] > out["pb"].shift(1)
    for p in (6, 9, 14):
        out[f"mfi_{p}"] = money_flow_index(
            out["High"], out["Low"], out["Close"], out["Volume"], period=p
        )
    out["mfi_up"] = out["mfi_6"] > out["mfi_6"].shift(1)
    out["above_mid"] = close > mid
    out["break_upper"] = close > upper
    out["break_lower"] = close < lower
    return out


def friday_daily_features(daily: pd.DataFrame) -> pd.DataFrame:
    """Keep only Friday rows from daily feature pack (once-a-week decisions)."""
    # Expect columns from add_swing_features; compute minimal set if missing
    from backtest.features import add_swing_features

    feat = add_swing_features(daily)
    fri = feat[feat.index.dayofweek == 4].copy()
    # map names to weekly schema
    fri = fri.rename(columns={"pb_20": "pb", "bb_mid": "bb_mid", "mfi_6": "mfi_6", "mfi_14": "mfi_14"})
    if "mfi_9" not in fri.columns:
        fri["mfi_9"] = money_flow_index(fri["High"], fri["Low"], fri["Close"], fri["Volume"], 9)
    fri["mfi_up"] = fri["mfi_6"] > fri["mfi_6"].shift(1)
    fri["pb_up"] = fri["pb"] > fri["pb"].shift(1)
    return fri


def entry_mask(df: pd.DataFrame, rule: WeeklyRule) -> pd.Series:
    mfi = df[f"mfi_{rule.mfi_period}"]
    pb = df["pb"]
    ok = pd.Series(True, index=df.index)

    if rule.style == "mean_rev":
        ok &= mfi <= rule.mfi_buy_max
        ok &= pb <= rule.pb_buy_max
        if rule.require_squeeze:
            ok &= df["squeeze"].fillna(False)
        if rule.bw_pct_max < 1.0:
            ok &= df["bw_pct"] <= rule.bw_pct_max
        if rule.require_pb_up:
            ok &= df["pb_up"].fillna(False)
        if rule.require_mfi_up:
            ok &= df["mfi_up"].fillna(False)
    elif rule.style == "squeeze_break":
        recent_sq = df["squeeze"].rolling(3, min_periods=1).max().astype(bool)
        ok &= recent_sq
        ok &= df["bw_expand"].fillna(False)
        if rule.require_break_upper:
            ok &= df["break_upper"].fillna(False)
        else:
            ok &= pb >= rule.pb_buy_min
        ok &= mfi >= rule.mfi_buy_min
        if rule.require_mfi_up:
            ok &= df["mfi_up"].fillna(False)
    elif rule.style == "band_walk":
        ok &= pb >= rule.pb_buy_min
        ok &= mfi >= rule.mfi_buy_min
        ok &= df["above_mid"].fillna(False)
        ok &= df["bw_expand"].fillna(False)
        if rule.require_mfi_up:
            ok &= df["mfi_up"].fillna(False)
    else:
        raise ValueError(rule.style)
    return ok.fillna(False)


def simulate_weekly(df: pd.DataFrame, rule: WeeklyRule) -> list[dict]:
    """Enter next weekly open after signal week; one decision per week."""
    buys = entry_mask(df, rule).to_numpy()
    close = df["Close"].to_numpy(float)
    open_ = df["Open"].to_numpy(float)
    pb = df["pb"].to_numpy(float)
    mfi = df[f"mfi_{rule.mfi_period}"].to_numpy(float)
    mid = df["bb_mid"].to_numpy(float)
    idx = df.index
    n = len(df)
    trades: list[dict] = []
    i = 1
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
            if close[j] < stop_px:
                exit_i, reason = j + 1, "stop"
                break
            sell = False
            if np.isfinite(pb[j]) and np.isfinite(mfi[j]):
                if pb[j] >= rule.pb_sell_min and mfi[j] >= rule.mfi_sell_min:
                    sell, reason = True, "overheat"
                if rule.exit_to_mid and np.isfinite(mid[j]) and close[j] >= mid[j] and j > entry_i:
                    sell, reason = True, "mid"
                if peak_pb - pb[j] >= rule.trail_pb_drop and peak_pb >= 0.7:
                    sell, reason = True, "trail"
            if sell:
                exit_i = j + 1
                break
        if exit_i is None:
            exit_i = min(entry_i + rule.max_hold, n - 1)
            reason = "timeout" if exit_i < n - 1 else "eod"
        exit_px = open_[exit_i] if np.isfinite(open_[exit_i]) else close[exit_i]
        if exit_i == n - 1:
            exit_px = close[exit_i]
        trades.append(
            {
                "entry_date": idx[entry_i],
                "exit_date": idx[exit_i],
                "entry_px": float(entry_px),
                "exit_px": float(exit_px),
                "return": float(exit_px / entry_px - 1.0),
                "bars": int(exit_i - entry_i),
                "reason": reason,
            }
        )
        i = exit_i
    return trades


def build_weekly_grid() -> list[WeeklyRule]:
    rules: list[WeeklyRule] = []
    for mfi_p, mfi_max, pb_max, filt, confirm, hold, mid in itertools.product(
        (6, 9, 14),
        (20, 25, 30),
        (0.0, 0.15, 0.25),
        ("none", "squeeze", "bw40"),
        ("none", "pb_up", "both"),
        (4, 8),
        (True, False),
    ):
        sq = filt == "squeeze"
        bw = 0.40 if filt == "bw40" else 1.0
        rules.append(
            WeeklyRule(
                name=f"WMR_mfi{mfi_p}<{mfi_max}_pb<={pb_max}_{filt}_{confirm}_h{hold}_mid{int(mid)}",
                style="mean_rev",
                mfi_period=mfi_p,
                mfi_buy_max=float(mfi_max),
                pb_buy_max=float(pb_max),
                require_squeeze=sq,
                bw_pct_max=float(bw),
                require_pb_up=confirm in ("pb_up", "both"),
                require_mfi_up=confirm == "both",
                stop_pct=0.08,
                max_hold=int(hold),
                exit_to_mid=bool(mid),
                pb_sell_min=0.85,
                mfi_sell_min=70.0,
            )
        )

    for mfi_p, mfi_min, brk, mup, hold in itertools.product(
        (6, 14), (50, 55), (True, False), (True, False), (4, 8)
    ):
        rules.append(
            WeeklyRule(
                name=f"WSQ_mfi{mfi_p}>={mfi_min}_brk{int(brk)}_mup{int(mup)}_h{hold}",
                style="squeeze_break",
                mfi_period=mfi_p,
                mfi_buy_min=float(mfi_min),
                pb_buy_min=0.8,
                require_break_upper=bool(brk),
                require_mfi_up=bool(mup),
                stop_pct=0.10,
                max_hold=int(hold),
                exit_to_mid=False,
                pb_sell_min=1.05,
                mfi_sell_min=85.0,
            )
        )

    for mfi_p, mfi_min, pb_min, mup, hold in itertools.product(
        (6, 14), (55, 60), (0.75, 0.85), (True, False), (4, 8)
    ):
        rules.append(
            WeeklyRule(
                name=f"WWALK_mfi{mfi_p}>={mfi_min}_pb>={pb_min}_mup{int(mup)}_h{hold}",
                style="band_walk",
                mfi_period=mfi_p,
                mfi_buy_min=float(mfi_min),
                pb_buy_min=float(pb_min),
                require_mfi_up=bool(mup),
                stop_pct=0.12,
                max_hold=int(hold),
                exit_to_mid=False,
                pb_sell_min=1.1,
                mfi_sell_min=90.0,
                trail_pb_drop=0.25,
            )
        )
    return rules


def score_rule(ts: dict, ms: dict) -> float:
    if not ts["trades"] or ts["trades"] < 20 or not ms["months"] or ms["months"] < 12:
        return -1e9
    if ms["avg_month_pct"] is None:
        return -1e9
    s = (
        ms["avg_month_pct"] * 2.0
        + (ms["median_month_pct"] or 0) * 1.5
        + (ms["pct_months_ge_20"] or 0) * 0.5
        + (ts["win_rate"] or 0) * 0.08
        + min(ts["trades"], 150) * 0.02
    )
    if ms["worst_month_pct"] is not None and ms["worst_month_pct"] < -35:
        s -= 20
    return s


def evaluate_mode(
    datasets: dict[str, pd.DataFrame],
    rules: list[WeeklyRule],
    *,
    mode: str,
    max_positions: list[int],
) -> pd.DataFrame:
    rows = []
    for ri, rule in enumerate(rules):
        all_trades: list[dict] = []
        for ticker, df in datasets.items():
            for t in simulate_weekly(df, rule):
                t = dict(t)
                t["ticker"] = ticker
                all_trades.append(t)
        ts = trade_stats(all_trades)
        best_score, best_ms, best_pos = -1e9, monthly_stats(pd.Series(dtype=float)), max_positions[0]
        for pos in max_positions:
            monthly = portfolio_monthly_returns(all_trades, max_positions=pos)
            ms = monthly_stats(monthly)
            sc = score_rule(ts, ms)
            if sc > best_score:
                best_score, best_ms, best_pos = sc, ms, pos
        rows.append(
            {
                "mode": mode,
                "rule": rule.name,
                "style": rule.style,
                "max_positions": best_pos,
                "score": round(best_score, 2),
                **{f"t_{k}": v for k, v in ts.items()},
                **{f"m_{k}": v for k, v in best_ms.items()},
            }
        )
        if (ri + 1) % 100 == 0:
            print(f"  [{mode}] {ri+1}/{len(rules)} best={max(r['score'] for r in rows):.1f}")
    return pd.DataFrame(rows).sort_values("score", ascending=False)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--start", default="2019-01-01")
    p.add_argument("--cache", default="cache_ohlcv_v2")
    p.add_argument("--max-rules", type=int, default=None)
    p.add_argument("--out", default="out_weekly_optimize.csv")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    universe = pd.read_csv("out_top100_universe.csv")
    if args.limit:
        universe = universe.head(args.limit)

    cache = Path(args.cache)
    weekly_sets: dict[str, pd.DataFrame] = {}
    friday_sets: dict[str, pd.DataFrame] = {}
    for i, row in universe.iterrows():
        path = cache / f"{row['yahoo']}.pkl"
        if not path.exists():
            continue
        daily = pd.read_pickle(path)
        daily = daily[daily.index >= pd.Timestamp(args.start)]
        if len(daily) < 120:
            continue
        w = add_weekly_features(to_weekly(daily))
        if len(w) >= 40:
            weekly_sets[row["yahoo"]] = w
        try:
            fri = friday_daily_features(daily[["Open", "High", "Low", "Close", "Volume"]])
            fri = fri[fri.index >= pd.Timestamp(args.start)]
            if len(fri) >= 40:
                friday_sets[row["yahoo"]] = fri
        except Exception as exc:  # noqa: BLE001
            print(f"friday skip {row['yahoo']}: {exc}")
        if (i + 1) % 20 == 0:
            print(f"features {i+1}/{len(universe)}")

    print(f"Weekly datasets: {len(weekly_sets)} | Friday-daily: {len(friday_sets)}")
    rules = build_weekly_grid()
    if args.max_rules:
        rules = rules[: args.max_rules]
    print(f"Rules: {len(rules)}")

    pos = [1, 2, 3]
    res_w = evaluate_mode(weekly_sets, rules, mode="weekly_bars", max_positions=pos)
    res_f = evaluate_mode(friday_sets, rules, mode="friday_daily", max_positions=pos)
    res = pd.concat([res_w, res_f], ignore_index=True).sort_values("score", ascending=False)
    res.to_csv(args.out, index=False)

    cols = [
        "mode",
        "style",
        "max_positions",
        "t_trades",
        "t_win_rate",
        "t_avg_return",
        "m_avg_month_pct",
        "m_median_month_pct",
        "m_pct_months_ge_20",
        "m_best_month_pct",
        "m_worst_month_pct",
        "score",
        "rule",
    ]
    print("\n=== TOP 20 WEEKLY METHODS ===")
    print(tabulate(res.head(20)[cols].values, headers=cols, tablefmt="github", floatfmt=".2f"))

    best = res.iloc[0]
    print("\n=== BEST ===")
    print(best[cols].to_string())

    # Compare best of each mode
    for mode in ("weekly_bars", "friday_daily"):
        sub = res[res["mode"] == mode]
        if len(sub):
            b = sub.iloc[0]
            print(
                f"\nBest {mode}: avg_month={b['m_avg_month_pct']}% win={b['t_win_rate']:.1f}% "
                f"pos={b['max_positions']} rule={b['rule']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

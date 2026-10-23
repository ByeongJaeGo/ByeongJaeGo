"""Reproduce sequential equity-max backtest on cached KOSDAQ OHLCV.

Expects:
  /tmp/krx_wide_universe.csv
  /tmp/ohlcv_krx_wide_from2020/*.pkl
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.equity_max_rules import EQUITY_MAX, EquityMaxParams
from backtest.indicators import bollinger_bands, bollinger_percent_b, money_flow_index


def _simulate_exit(
    entry: float,
    hi: np.ndarray,
    lo: np.ndarray,
    cl: np.ndarray,
    dates: np.ndarray,
    p: EquityMaxParams,
) -> tuple[float, np.datetime64, str, int]:
    stop_px = entry * (1 - p.stop_pct)
    tp_px = entry * (1 + p.take_profit_pct)
    hold = p.max_hold_days
    for k in range(hold):
        if lo[k] <= stop_px:
            return -p.stop_pct, dates[k], "stop", k + 1
        if hi[k] >= tp_px:
            return p.take_profit_pct, dates[k], "tp", k + 1
        if p.day3_no_up_cut and (k + 1) == 3 and cl[k] <= entry:
            return cl[k] / entry - 1, dates[k], "d3", k + 1
    return cl[hold - 1] / entry - 1, dates[hold - 1], "time", hold


def collect_signals(
    universe_csv: Path,
    cache_dir: Path,
    p: EquityMaxParams,
    start: str,
    end: str,
) -> pd.DataFrame:
    uni = pd.read_csv(universe_csv)
    meta = {r.yahoo: (r["name"], r.market) for _, r in uni.iterrows()}
    files = [
        x
        for x in cache_dir.glob("*.pkl")
        if x.stem in meta and meta[x.stem][1] == p.market
    ]
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    rows: list[dict] = []
    for path in files:
        name = meta[path.stem][0]
        df = pd.read_pickle(path).copy()
        if len(df) < 260:
            continue
        bands = bollinger_bands(df["Close"], 20, 2.0)
        df["percent_b"] = bands["percent_b"]
        df["pb_prev"] = df["percent_b"].shift(1)
        df["bb_width"] = (bands["bb_upper"] - bands["bb_lower"]) / bands["bb_middle"]
        df["close_loc"] = (df["Close"] - df["Low"]) / (df["High"] - df["Low"]).replace(
            0, np.nan
        )
        df["ret1"] = df["Close"].pct_change()
        weekly = (
            df.resample("W-FRI")
            .agg(
                {
                    "Open": "first",
                    "High": "max",
                    "Low": "min",
                    "Close": "last",
                    "Volume": "sum",
                }
            )
            .dropna(how="any")
        )
        if len(weekly) < 25:
            continue
        wmfi10 = money_flow_index(
            weekly["High"], weekly["Low"], weekly["Close"], weekly["Volume"], 10
        )
        wmfi6 = money_flow_index(
            weekly["High"], weekly["Low"], weekly["Close"], weekly["Volume"], 6
        )
        wpb = bollinger_percent_b(weekly["Close"], 20, 2.0)
        wfeat = pd.DataFrame(
            {
                "w_mfi10": wmfi10,
                "w_mfi10_prev": wmfi10.shift(1),
                "w_mfi6": wmfi6,
                "w_pb20": wpb,
            }
        ).dropna()
        wfeat = wfeat.reset_index()
        wfeat.columns = ["wdate", "w_mfi10", "w_mfi10_prev", "w_mfi6", "w_pb20"]
        daily = df.reset_index().rename(columns={df.reset_index().columns[0]: "date"})
        m = pd.merge_asof(
            daily.sort_values("date"),
            wfeat.sort_values("wdate"),
            left_on="date",
            right_on="wdate",
            direction="backward",
        ).set_index("date")
        cross = (m["pb_prev"] <= p.pb_cross) & (m["percent_b"] > p.pb_cross)
        mask = (m.index >= start_ts) & (m.index <= end_ts) & cross
        for i in np.where(mask.values)[0]:
            if i + p.max_hold_days >= len(m):
                continue
            r = m.iloc[i]
            if pd.isna(r.w_mfi10) or pd.isna(r.w_pb20) or pd.isna(r.w_mfi6):
                continue
            if p.require_weekly_mfi10_up and not (
                pd.notna(r.w_mfi10_prev) and r.w_mfi10 > r.w_mfi10_prev
            ):
                continue
            if p.weekly_pb_lo is not None and not (
                p.weekly_pb_lo <= float(r.w_pb20) <= float(p.weekly_pb_hi)
            ):
                continue
            if p.bb_width_min is not None and (
                pd.isna(r.bb_width) or float(r.bb_width) < p.bb_width_min
            ):
                continue
            if p.weekly_mfi6_max is not None and float(r.w_mfi6) > p.weekly_mfi6_max:
                continue
            if p.close_loc_min is not None and (
                pd.isna(r.close_loc) or float(r.close_loc) < p.close_loc_min
            ):
                continue
            if p.ret1_min is not None and (pd.isna(r.ret1) or float(r.ret1) < p.ret1_min):
                continue
            entry = float(r.Close)
            hi = m["High"].iloc[i + 1 : i + 1 + p.max_hold_days].to_numpy(float)
            lo = m["Low"].iloc[i + 1 : i + 1 + p.max_hold_days].to_numpy(float)
            cl = m["Close"].iloc[i + 1 : i + 1 + p.max_hold_days].to_numpy(float)
            dts = m.index[i + 1 : i + 1 + p.max_hold_days].to_numpy(dtype="datetime64[ns]")
            ret, sell, reason, hold = _simulate_exit(entry, hi, lo, cl, dts, p)
            rows.append(
                dict(
                    name=name,
                    buy=m.index[i],
                    sell=pd.Timestamp(sell),
                    buy_px=entry,
                    ret=ret * 100,
                    hold=hold,
                    reason=reason,
                )
            )
    return pd.DataFrame(rows)


def sequential_equity(signals: pd.DataFrame, initial: float = 100_000_000) -> tuple[float, pd.DataFrame]:
    if signals.empty:
        return initial, signals
    T = signals.sort_values(["buy", "name"]).reset_index(drop=True)
    cash = initial
    busy = None
    taken = []
    for _, r in T.iterrows():
        if busy is not None and r.buy <= busy:
            continue
        cash *= 1 + r.ret / 100
        taken.append({**r.to_dict(), "capital": cash})
        busy = r.sell
    return cash, pd.DataFrame(taken)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2020-01-01")
    ap.add_argument("--end", default="2026-10-22")
    ap.add_argument("--universe", default="/tmp/krx_wide_universe.csv")
    ap.add_argument("--cache", default="/tmp/ohlcv_krx_wide_from2020")
    ap.add_argument("--initial", type=float, default=100_000_000)
    args = ap.parse_args()
    p = EQUITY_MAX
    sig = collect_signals(Path(args.universe), Path(args.cache), p, args.start, args.end)
    final, taken = sequential_equity(sig, args.initial)
    print(f"signals {len(sig)} | sequential trades {len(taken)}")
    print(f"final capital {final:,.0f} ({final / args.initial:.2f}x)")
    if len(taken):
        print(
            f"wr {(taken.ret > 0).mean() * 100:.1f}% avg {taken.ret.mean():+.2f}% "
            f"mean hold {taken.hold.mean():.1f}d"
        )


if __name__ == "__main__":
    main()

"""FORWARD_MAX 2026 sequential simulation (final ops rule included).

Buy (signal day close):
  daily %b crosses above 0.3
  weekly MFI(10) rising
  close location >= 0.8
  ret1 >= +5%

Sell:
  stop -10% (intraday low)
  take-profit +20% (intraday high)
  max hold 20 trading days
  ops: if NEXT day's close <= -3% from entry, cut at that close (d1)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.forward_max_rules import FORWARD_MAX, ForwardMaxParams
from backtest.indicators import bollinger_bands, bollinger_percent_b, money_flow_index


def _simulate_exit(
    entry: float,
    hi: np.ndarray,
    lo: np.ndarray,
    cl: np.ndarray,
    dates: np.ndarray,
    p: ForwardMaxParams,
    day1_cut_pct: float | None,
) -> tuple[float, np.datetime64, str, int]:
    stop_px = entry * (1 - p.stop_pct)
    tp_px = entry * (1 + p.take_profit_pct)
    hold = p.max_hold_days
    n = min(hold, len(hi))
    for k in range(n):
        # Day-1 early cut: first post-entry close <= -day1_cut_pct
        if day1_cut_pct is not None and k == 0 and cl[k] <= entry * (1 - day1_cut_pct):
            return float(cl[k] / entry - 1), dates[k], "d1", 1
        if lo[k] <= stop_px:
            return -p.stop_pct, dates[k], "stop", k + 1
        if hi[k] >= tp_px:
            return p.take_profit_pct, dates[k], "tp", k + 1
        if p.day3_no_up_cut and (k + 1) == 3 and cl[k] <= entry:
            return float(cl[k] / entry - 1), dates[k], "d3", k + 1
    return float(cl[n - 1] / entry - 1), dates[n - 1], "time", n


def collect_signals(
    universe_csv: Path,
    cache_dir: Path,
    p: ForwardMaxParams,
    start: str,
    end: str,
    day1_cut_pct: float | None = 0.03,
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
            # Need at least one post-entry bar; allow short windows near sample end.
            if i + 1 >= len(m):
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
            j1 = i + 1
            j2 = min(len(m), i + 1 + p.max_hold_days)
            hi = m["High"].iloc[j1:j2].to_numpy(float)
            lo = m["Low"].iloc[j1:j2].to_numpy(float)
            cl = m["Close"].iloc[j1:j2].to_numpy(float)
            dts = m.index[j1:j2].to_numpy(dtype="datetime64[ns]")
            if len(hi) < 1:
                continue
            ret, sell, reason, hold = _simulate_exit(
                entry, hi, lo, cl, dts, p, day1_cut_pct
            )
            # Incomplete hold window at data end → mark eod if neither stop/tp/d1 fired early
            if reason == "time" and len(hi) < p.max_hold_days:
                reason = "eod"
            rows.append(
                dict(
                    name=name,
                    buy=m.index[i],
                    sell=pd.Timestamp(sell),
                    buy_px=entry,
                    ret=ret * 100,
                    hold=hold,
                    reason=reason,
                    sig_ret1=float(r.ret1) if pd.notna(r.ret1) else 0.0,
                    sig_close_loc=float(r.close_loc) if pd.notna(r.close_loc) else 0.0,
                )
            )
    return pd.DataFrame(rows)


def sequential_equity(
    signals: pd.DataFrame,
    initial: float = 100_000_000,
    same_day: str = "ret1",
) -> tuple[float, pd.DataFrame]:
    """All-in sequential. When several signals share a free day, pick one.

    same_day:
      - name: alphabetical (legacy / fragile)
      - ret1: strongest same-day momentum (practical default)
    """
    if signals.empty:
        return initial, signals
    S = signals.copy()
    S["buy"] = pd.to_datetime(S["buy"])
    S["sell"] = pd.to_datetime(S["sell"])
    if same_day == "ret1" and "sig_ret1" in S.columns:
        S = S.sort_values(["buy", "sig_ret1", "name"], ascending=[True, False, True])
    else:
        S = S.sort_values(["buy", "name"])
    S = S.reset_index(drop=True)

    cash = initial
    busy = None
    taken = []
    for _, r in S.iterrows():
        if busy is not None and r.buy <= busy:
            continue
        capital_in = cash
        cash *= 1 + r.ret / 100
        taken.append({**r.to_dict(), "capital_in": capital_in, "capital": cash})
        busy = r.sell
    out = pd.DataFrame(taken)
    if len(out):
        out["ym"] = pd.to_datetime(out["buy"]).dt.strftime("%Y-%m")
    return cash, out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2026-01-01")
    ap.add_argument("--end", default="2026-10-24")
    ap.add_argument("--universe", default="/tmp/krx_wide_universe.csv")
    ap.add_argument("--cache", default="/tmp/ohlcv_krx_wide_from2020")
    ap.add_argument("--initial", type=float, default=100_000_000)
    ap.add_argument("--day1-cut", type=float, default=None)
    ap.add_argument("--same-day", choices=["ret1", "name"], default="ret1")
    ap.add_argument("--out", default="backtest/results/forward_max_final_2026_seq.csv")
    args = ap.parse_args()

    p = FORWARD_MAX
    day1 = args.day1_cut
    sig = collect_signals(
        Path(args.universe),
        Path(args.cache),
        p,
        args.start,
        args.end,
        day1_cut_pct=day1,
    )
    final, taken = sequential_equity(sig, args.initial, same_day=args.same_day)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if len(taken):
        taken.to_csv(out, index=False)

    label = "no d1-cut" if day1 is None else f"d1-cut -{day1*100:.0f}%"
    print(f"FORWARD_MAX ({args.start}~{args.end}) [{label}] same-day={args.same_day}")
    print(f"signals {len(sig)} | sequential trades {len(taken)}")
    print(f"final {final:,.0f} ({final / args.initial:.2f}x)")
    if len(taken):
        wr = (taken.ret > 0).mean() * 100
        print(
            f"wr {wr:.1f}% avg {taken.ret.mean():+.2f}% "
            f"mean hold {taken.hold.mean():.1f}d"
        )
        print(f"reasons:\n{taken.reason.value_counts().to_string()}")
        print(taken[["name", "buy", "sell", "buy_px", "ret", "hold", "reason", "capital"]].to_string(index=False))
        print(f"wrote {out}")


if __name__ == "__main__":
    main()

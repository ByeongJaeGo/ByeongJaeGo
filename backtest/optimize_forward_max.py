"""Forward-looking rule search on user's MFI/%b neighborhood.

Goal: rules that can keep making money going forward — not a 2020-only jackpot.

Selection
---------
1) Sequential all-in multiple > 1.0x in BOTH 2010-2019 and 2020-present
2) Among those, maximize geo-mean(mult_2010, mult_2020)
3) Tie-break: higher worst-period multiple, then more trades

Period caches are scored separately (Naver 2010s / Yahoo 2020s).
"""

from __future__ import annotations

import itertools
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.indicators import bollinger_bands, bollinger_percent_b, money_flow_index


@dataclass(frozen=True)
class BuyRule:
    pb_cross: float
    weekly_pb_lo: float | None
    weekly_pb_hi: float | None
    bb_width_min: float | None
    weekly_mfi6_max: float | None
    close_loc_min: float | None
    ret1_min: float | None


EXIT_CFG = [
    # stop, tp, hold, day3
    (0.10, 0.20, 5, False),
    (0.10, 0.20, 10, False),
    (0.10, 0.20, 20, False),
    (0.10, 0.20, 20, True),
    (0.10, 0.30, 5, False),
    (0.10, 0.30, 10, False),
    (0.15, 0.20, 5, False),
    (0.15, 0.20, 10, False),
    (0.15, 0.20, 20, False),
    (0.15, 0.20, 20, True),
    (0.15, 0.30, 5, False),
    (0.15, 0.30, 10, False),
    (0.15, 0.15, 10, False),
    (0.15, 0.15, 20, False),
]


def _exit_key(stop: float, tp: float, hold: int, day3: bool) -> str:
    return f"r_{int(stop*100)}_{int(tp*100)}_{hold}_{int(day3)}"


def _exit_ret(
    entry: float,
    hi: np.ndarray,
    lo: np.ndarray,
    cl: np.ndarray,
    stop: float,
    tp: float,
    hold: int,
    day3: bool,
) -> tuple[float, int]:
    stop_px = entry * (1 - stop)
    tp_px = entry * (1 + tp)
    n = min(hold, len(hi))
    for k in range(n):
        if lo[k] <= stop_px:
            return -stop, k + 1
        if hi[k] >= tp_px:
            return tp, k + 1
        if day3 and (k + 1) == 3 and cl[k] <= entry:
            return float(cl[k] / entry - 1), k + 1
    return float(cl[n - 1] / entry - 1), n


def _build_stock_table(path: Path, name: str) -> pd.DataFrame | None:
    df = pd.read_pickle(path).copy()
    if len(df) < 260:
        return None
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
        return None
    wmfi10 = money_flow_index(
        weekly["High"], weekly["Low"], weekly["Close"], weekly["Volume"], 10
    )
    wmfi6 = money_flow_index(
        weekly["High"], weekly["Low"], weekly["Close"], weekly["Volume"], 6
    )
    wpb = bollinger_percent_b(weekly["Close"], 20, 2.0)
    wfeat = (
        pd.DataFrame(
            {
                "w_mfi10": wmfi10,
                "w_mfi10_prev": wmfi10.shift(1),
                "w_mfi6": wmfi6,
                "w_pb20": wpb,
            }
        )
        .dropna()
        .reset_index()
    )
    wfeat.columns = ["wdate", "w_mfi10", "w_mfi10_prev", "w_mfi6", "w_pb20"]
    daily = df.reset_index().rename(columns={df.reset_index().columns[0]: "date"})
    m = pd.merge_asof(
        daily.sort_values("date"),
        wfeat.sort_values("wdate"),
        left_on="date",
        right_on="wdate",
        direction="backward",
    ).set_index("date")
    m["name"] = name
    return m


def precompute_candidates(
    universe_csv: Path,
    cache_dir: Path,
    start: str,
    end: str,
    market: str = "KOSDAQ",
    max_hold: int = 20,
) -> pd.DataFrame:
    uni = pd.read_csv(universe_csv)
    meta = {r.yahoo: (r["name"], r.market) for _, r in uni.iterrows()}
    files = [
        x
        for x in cache_dir.glob("*.pkl")
        if x.stem in meta and meta[x.stem][1] == market
    ]
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    rows: list[dict] = []
    for path in files:
        name = meta[path.stem][0]
        m = _build_stock_table(path, name)
        if m is None:
            continue
        # union of crosses at 0.2/0.3/0.4
        cross = (
            ((m["pb_prev"] <= 0.2) & (m["percent_b"] > 0.2))
            | ((m["pb_prev"] <= 0.3) & (m["percent_b"] > 0.3))
            | ((m["pb_prev"] <= 0.4) & (m["percent_b"] > 0.4))
        )
        mask = (
            cross
            & m["w_mfi10"].notna()
            & m["w_mfi10_prev"].notna()
            & (m.index >= start_ts)
            & (m.index <= end_ts)
        )
        idxs = np.where(mask.values)[0]
        for i in idxs:
            if i + max_hold >= len(m):
                continue
            r = m.iloc[i]
            hi = m["High"].iloc[i + 1 : i + 1 + max_hold].to_numpy(float)
            lo = m["Low"].iloc[i + 1 : i + 1 + max_hold].to_numpy(float)
            cl = m["Close"].iloc[i + 1 : i + 1 + max_hold].to_numpy(float)
            sdates = m.index[i + 1 : i + 1 + max_hold]
            entry = float(r.Close)
            rec = dict(
                name=name,
                buy=m.index[i],
                close=entry,
                percent_b=float(r.percent_b),
                pb_prev=float(r.pb_prev),
                bb_width=float(r.bb_width) if pd.notna(r.bb_width) else np.nan,
                close_loc=float(r.close_loc) if pd.notna(r.close_loc) else np.nan,
                ret1=float(r.ret1) if pd.notna(r.ret1) else np.nan,
                w_mfi10=float(r.w_mfi10),
                w_mfi10_prev=float(r.w_mfi10_prev),
                w_mfi6=float(r.w_mfi6) if pd.notna(r.w_mfi6) else np.nan,
                w_pb20=float(r.w_pb20) if pd.notna(r.w_pb20) else np.nan,
                cross_02=bool(r.pb_prev <= 0.2 and r.percent_b > 0.2),
                cross_03=bool(r.pb_prev <= 0.3 and r.percent_b > 0.3),
                cross_04=bool(r.pb_prev <= 0.4 and r.percent_b > 0.4),
            )
            for stop, tp, hold, day3 in EXIT_CFG:
                ret, h = _exit_ret(entry, hi, lo, cl, stop, tp, hold, day3)
                key = _exit_key(stop, tp, hold, day3)
                rec[key] = ret * 100
                rec[key + "_sell"] = sdates[h - 1]
            rows.append(rec)
    return pd.DataFrame(rows)


def sequential_mult(df: pd.DataFrame, ret_col: str, sell_col: str) -> tuple[float, int, float]:
    if df.empty:
        return 1.0, 0, 0.0
    T = df.sort_values(["buy", "name"])[["buy", "name", ret_col, sell_col]].copy()
    T.columns = ["buy", "name", "ret", "sell"]
    cash = 1.0
    busy = None
    n = 0
    wins = 0
    for buy, _name, ret, sell in T.itertuples(index=False, name=None):
        if busy is not None and buy <= busy:
            continue
        cash *= 1 + ret / 100
        n += 1
        wins += ret > 0
        busy = sell
    wr = wins / n if n else 0.0
    return cash, n, wr


def buy_mask(df: pd.DataFrame, rule: BuyRule) -> pd.Series:
    if rule.pb_cross == 0.2:
        m = df["cross_02"]
    elif rule.pb_cross == 0.3:
        m = df["cross_03"]
    else:
        m = df["cross_04"]
    m = m & (df["w_mfi10"] > df["w_mfi10_prev"])
    if rule.weekly_pb_lo is not None:
        m = m & (df["w_pb20"] >= rule.weekly_pb_lo) & (df["w_pb20"] <= rule.weekly_pb_hi)
    if rule.bb_width_min is not None:
        m = m & (df["bb_width"] >= rule.bb_width_min)
    if rule.weekly_mfi6_max is not None:
        m = m & (df["w_mfi6"] <= rule.weekly_mfi6_max)
    if rule.close_loc_min is not None:
        m = m & (df["close_loc"] >= rule.close_loc_min)
    if rule.ret1_min is not None:
        m = m & (df["ret1"] >= rule.ret1_min)
    return m


def buy_grid() -> list[BuyRule]:
    rules: list[BuyRule] = []
    for vals in itertools.product(
        [0.2, 0.3, 0.4],
        [None, (0.15, 0.30), (0.10, 0.40)],
        [None, 0.15, 0.20, 0.25],
        [None, 40.0, 45.0, 50.0],
        [0.6, 0.7, 0.8],
        [0.03, 0.05, 0.07],
    ):
        pb, wpb, bw, m6, loc, ret1 = vals
        rules.append(
            BuyRule(
                pb_cross=pb,
                weekly_pb_lo=None if wpb is None else wpb[0],
                weekly_pb_hi=None if wpb is None else wpb[1],
                bb_width_min=bw,
                weekly_mfi6_max=m6,
                close_loc_min=loc,
                ret1_min=ret1,
            )
        )
    return rules


def main() -> None:
    uni = Path("/tmp/krx_wide_universe.csv")
    out_dir = Path("backtest/results")
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print("precompute 2010-2019...", flush=True)
    a = precompute_candidates(
        uni, Path("/tmp/ohlcv_krx_from2010"), "2010-01-01", "2019-12-31"
    )
    print(f"  rows={len(a)} ({time.time()-t0:.0f}s)", flush=True)
    t1 = time.time()
    print("precompute 2020-2026...", flush=True)
    b = precompute_candidates(
        uni, Path("/tmp/ohlcv_krx_wide_from2020"), "2020-01-01", "2026-10-22"
    )
    print(f"  rows={len(b)} ({time.time()-t1:.0f}s)", flush=True)

    buys = buy_grid()
    print(f"buy rules={len(buys)} exit cfgs={len(EXIT_CFG)} total={len(buys)*len(EXIT_CFG)}", flush=True)

    rows = []
    best = None
    i = 0
    for br in buys:
        ma_mask = buy_mask(a, br) if len(a) else pd.Series(dtype=bool)
        mb_mask = buy_mask(b, br) if len(b) else pd.Series(dtype=bool)
        a_sub = a.loc[ma_mask] if len(a) else a
        b_sub = b.loc[mb_mask] if len(b) else b
        for stop, tp, hold, day3 in EXIT_CFG:
            i += 1
            key = _exit_key(stop, tp, hold, day3)
            sell_key = key + "_sell"
            ma, na, wra = sequential_mult(a_sub, key, sell_key)
            mb, nb, wrb = sequential_mult(b_sub, key, sell_key)
            both_ok = ma > 1.0 and mb > 1.0 and na >= 15 and nb >= 15
            geo = math.sqrt(max(ma, 1e-12) * max(mb, 1e-12))
            worst = min(ma, mb)
            row = dict(
                pb_cross=br.pb_cross,
                weekly_pb_lo=br.weekly_pb_lo,
                weekly_pb_hi=br.weekly_pb_hi,
                bb_width_min=br.bb_width_min,
                weekly_mfi6_max=br.weekly_mfi6_max,
                close_loc_min=br.close_loc_min,
                ret1_min=br.ret1_min,
                stop_pct=stop,
                take_profit_pct=tp,
                max_hold_days=hold,
                day3_no_up_cut=day3,
                mult_2010=ma,
                n_2010=na,
                wr_2010=wra,
                mult_2020=mb,
                n_2020=nb,
                wr_2020=wrb,
                both_ok=both_ok,
                geo=geo,
                worst=worst,
            )
            rows.append(row)
            score = (geo, worst, na + nb)
            if both_ok and (best is None or score > best[0]):
                best = (score, row)
            if i % 1000 == 0:
                g = best[1]["geo"] if best else None
                print(f"[{i}] best_geo={g}", flush=True)

    df = pd.DataFrame(rows)
    df = df.sort_values(["both_ok", "geo", "worst"], ascending=[False, False, False])
    df.to_csv(out_dir / "forward_max_grid.csv", index=False)
    ok = df[df.both_ok]
    print(f"\nboth_ok count={len(ok)} / {len(df)}")
    cols = [
        "geo",
        "worst",
        "mult_2010",
        "mult_2020",
        "n_2010",
        "n_2020",
        "wr_2010",
        "wr_2020",
        "pb_cross",
        "weekly_pb_lo",
        "weekly_pb_hi",
        "bb_width_min",
        "weekly_mfi6_max",
        "close_loc_min",
        "ret1_min",
        "stop_pct",
        "take_profit_pct",
        "max_hold_days",
        "day3_no_up_cut",
    ]
    show = ok if len(ok) else df
    print(show[cols].head(20).to_string(index=False))
    if len(ok):
        ok.head(1).to_csv(out_dir / "forward_max_best.csv", index=False)
        # also compare legacy A/B/C markers if present in grid
    print(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()

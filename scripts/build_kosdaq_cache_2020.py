"""Build KOSDAQ universe + Yahoo OHLCV cache from 2019 (warmup) through now."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import FinanceDataReader as fdr
import pandas as pd
import yfinance as yf

OUT_UNI = Path("/tmp/krx_wide_universe.csv")
OUT_CACHE = Path("/tmp/ohlcv_krx_wide_from2020")
START = "2019-01-01"


def build_universe() -> pd.DataFrame:
    kq = fdr.StockListing("KOSDAQ").copy()
    kq = kq.rename(columns={"Code": "code", "Name": "name"})
    kq["code"] = kq["code"].astype(str).str.zfill(6)
    kq["market"] = "KOSDAQ"
    kq["yahoo"] = kq["code"] + ".KQ"
    # Keep common stocks; drop obvious ETFs/SPACs if name hints
    bad = kq["name"].astype(str).str.contains(
        "스팩|ETF|ETN|인버스|레버리지|KODEX|TIGER|ACE|KBSTAR", regex=True
    )
    kq = kq.loc[~bad, ["code", "name", "market", "yahoo"]].drop_duplicates("code")
    return kq.reset_index(drop=True)


def _download_one(yahoo: str, out_dir: Path) -> tuple[str, str]:
    dest = out_dir / f"{yahoo}.pkl"
    if dest.exists() and dest.stat().st_size > 1000:
        return yahoo, "skip"
    try:
        raw = yf.download(
            yahoo,
            start=START,
            auto_adjust=True,
            progress=False,
            threads=False,
        )
        if raw is None or raw.empty:
            return yahoo, "empty"
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = [c[0] for c in raw.columns]
        needed = ["Open", "High", "Low", "Close", "Volume"]
        if any(c not in raw.columns for c in needed):
            return yahoo, "cols"
        df = raw[needed].dropna(how="any").copy()
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df = df.sort_index()
        if len(df) < 100:
            return yahoo, "short"
        df.to_pickle(dest)
        return yahoo, "ok"
    except Exception as e:  # noqa: BLE001
        return yahoo, f"err:{type(e).__name__}"


def main() -> None:
    OUT_CACHE.mkdir(parents=True, exist_ok=True)
    uni = build_universe()
    uni.to_csv(OUT_UNI, index=False)
    print(f"universe {len(uni)} -> {OUT_UNI}")

    tickers = uni["yahoo"].tolist()
    ok = skip = fail = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=12) as ex:
        futs = [ex.submit(_download_one, y, OUT_CACHE) for y in tickers]
        for i, fut in enumerate(as_completed(futs), 1):
            y, st = fut.result()
            if st == "ok":
                ok += 1
            elif st == "skip":
                skip += 1
            else:
                fail += 1
            if i % 100 == 0 or i == len(tickers):
                elapsed = time.time() - t0
                print(
                    f"[{i}/{len(tickers)}] ok={ok} skip={skip} fail={fail} "
                    f"{elapsed:.0f}s last={y}:{st}"
                )
    print(f"done cache={OUT_CACHE} files={len(list(OUT_CACHE.glob('*.pkl')))}")


if __name__ == "__main__":
    main()

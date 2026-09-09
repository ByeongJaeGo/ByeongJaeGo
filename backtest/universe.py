"""Universe helpers: KOSPI/KOSDAQ top names by trading value (Naver)."""

from __future__ import annotations

import re
import time
from io import StringIO

import pandas as pd
import urllib.request

ETF_KW = (
    "KODEX",
    "TIGER",
    "ACE",
    "KBSTAR",
    "SOL",
    "PLUS",
    "HANARO",
    "TIMEFOLIO",
    "KOSEF",
    "TREX",
    "FOCUS",
    "ETN",
    "인버스",
    "레버리지",
    "선물",
    "RISE",
    "PARPLE",
    "WON",
    "UNICORN",
    "히어로",
    "KoAct",
    "ETF",
)


def _fetch_market_sum(sosok: int, max_pages: int = 60) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    first_code: str | None = None
    for page in range(1, max_pages + 1):
        url = f"https://finance.naver.com/sise/sise_market_sum.naver?sosok={sosok}&page={page}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            html = resp.read().decode("euc-kr", errors="replace")

        name_to_code = {
            name: code
            for code, name in re.findall(r"code=(\d{6})[^>]*>([^<]+)</a>", html)
        }
        tables = pd.read_html(StringIO(html))
        found = None
        for df in tables:
            if "종목명" in df.columns and "거래량" in df.columns and "현재가" in df.columns:
                found = df.dropna(subset=["종목명"]).copy()
                break
        if found is None or found.empty:
            break

        found["name"] = found["종목명"].astype(str)
        found["code"] = found["name"].map(name_to_code)
        found = found[found["code"].notna()]
        if found.empty:
            break

        if page == 1:
            first_code = str(found.iloc[0]["code"])
        elif str(found.iloc[0]["code"]) == first_code:
            break

        frames.append(found[["code", "name", "현재가", "거래량"]])
        if len(found) < 10:
            break
        time.sleep(0.05)

    if not frames:
        return pd.DataFrame(columns=["code", "name", "현재가", "거래량"])
    return pd.concat(frames, ignore_index=True).drop_duplicates("code")


def _is_etf_or_preferred(name: str) -> bool:
    if any(k in name for k in ETF_KW):
        return True
    if re.search(r"우$|우B$|우\(N\)", name):
        return True
    return False


def fetch_top_trading_stocks(n: int = 100) -> pd.DataFrame:
    """
    Approximate current top-N by trading value = price * volume
    across KOSPI (sosok=0) and KOSDAQ (sosok=1), excluding ETF/ETN/preferred.
    """
    kospi = _fetch_market_sum(0)
    kosdaq = _fetch_market_sum(1)
    kospi["market"] = "KOSPI"
    kosdaq["market"] = "KOSDAQ"
    all_df = pd.concat([kospi, kosdaq], ignore_index=True)
    all_df["price"] = pd.to_numeric(all_df["현재가"], errors="coerce")
    all_df["volume"] = pd.to_numeric(all_df["거래량"], errors="coerce")
    all_df["trade_value"] = all_df["price"] * all_df["volume"]
    all_df = all_df.dropna(subset=["trade_value", "code"])
    all_df = all_df[~all_df["name"].map(_is_etf_or_preferred)]
    all_df = all_df.sort_values("trade_value", ascending=False).drop_duplicates("code")
    top = all_df.head(n).reset_index(drop=True)
    top["yahoo"] = top.apply(
        lambda r: f"{r['code']}.KS" if r["market"] == "KOSPI" else f"{r['code']}.KQ",
        axis=1,
    )
    return top[["code", "name", "market", "trade_value", "yahoo"]]

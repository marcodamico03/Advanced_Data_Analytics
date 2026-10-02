"""
Stage 1: download the raw data from FRED/ALFRED and Yahoo Finance into data/raw/.
Run from the repository root: python code/01_get_data.py
Needs a free FRED API key in .env (see README).
"""

import os
import sys
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from dotenv import load_dotenv

from config import (DATA_RAW, ETF_TICKERS, FREEZE_DATE, MACRO_SERIES,
                    MARKET_SERIES, OBS_START, ROOT)

API_URL = "https://api.stlouisfed.org/fred/series/observations"


def fred_observations(series_id, api_key, realtime_start):
    """One FRED series, every value published between realtime_start and FREEZE_DATE."""
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": OBS_START,
        "observation_end": FREEZE_DATE,
        "realtime_start": realtime_start,
        "realtime_end": FREEZE_DATE,
        "limit": 100000,
    }
    rows, offset = [], 0
    while True:
        params["offset"] = offset
        for _ in range(3):
            resp = requests.get(API_URL, params=params, timeout=60)
            if resp.status_code != 429:
                break
            time.sleep(20)  # too many requests: wait and retry
        if resp.status_code != 200:
            msg = f"{series_id}: HTTP {resp.status_code} {resp.text[:200]}"
            raise RuntimeError(msg.replace(api_key, "***"))

        data = resp.json()
        rows += data["observations"]
        offset += len(data["observations"])
        if not data["observations"] or offset >= data["count"]:
            break

    if not rows:
        raise RuntimeError(f"{series_id}: no data")

    df = pd.DataFrame(rows)
    df["value"] = pd.to_numeric(df["value"].replace(".", np.nan), errors="coerce")  # FRED writes missing as "."
    df.insert(0, "series_id", series_id)
    return df[["series_id", "date", "realtime_start", "realtime_end", "value"]]


def download_fred(series, api_key, all_vintages):
    """All series in one long table. all_vintages=False keeps only the version valid on FREEZE_DATE."""
    start = "1776-07-04" if all_vintages else FREEZE_DATE  # 1776-07-04 = earliest date the API accepts
    frames = []
    for sid in series:
        df = fred_observations(sid, api_key, start)
        frames.append(df)
        print(f"  {sid:<9} {len(df):>6} rows, {df['date'].min()} -> {df['date'].max()}")
        time.sleep(0.6)  # API limit is 120 requests per minute
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["series_id", "date", "realtime_start"]).reset_index(drop=True)


def latest_values(vintages):
    """From the vintage table, keep the value valid on FREEZE_DATE for each month."""
    live = vintages[(vintages["realtime_start"] <= FREEZE_DATE)
                    & (vintages["realtime_end"] >= FREEZE_DATE)]
    assert not live.duplicated(["series_id", "date"]).any()
    return live[["series_id", "date", "value"]].reset_index(drop=True)


def download_etfs(tickers):
    """Daily prices from Yahoo Finance up to FREEZE_DATE, long table."""
    end = (pd.Timestamp(FREEZE_DATE) + timedelta(days=1)).strftime("%Y-%m-%d")  # end date is excluded
    raw = yf.download(tickers, start=OBS_START, end=end, interval="1d",
                      auto_adjust=False, actions=False, progress=False)
    raw.columns.names = ["field", "ticker"]
    df = (raw[["Adj Close", "Close", "Volume"]]
          .stack("ticker", future_stack=True)
          .reset_index()
          .rename(columns={"Date": "date"})
          .dropna(subset=["Adj Close"])
          .sort_values(["ticker", "date"])
          .reset_index(drop=True))

    for t in tickers:
        g = df[df["ticker"] == t]
        if g.empty:
            raise RuntimeError(f"no prices for {t}")
        print(f"  {t:<4} {len(g):>5} days, {g['date'].min().date()} -> {g['date'].max().date()}")
    return df


def write_sources():
    """Table with source and licence of every series."""
    rows = []
    for role, series in [("regime", MACRO_SERIES), ("descriptive", MARKET_SERIES)]:
        for sid, (block, desc, licence, shareable) in series.items():
            rows.append({"source": "FRED/ALFRED", "series_id": sid, "role": role, "block": block,
                         "description": desc, "licence": licence, "redistributable": shareable,
                         "url": f"https://fred.stlouisfed.org/series/{sid}"})
    for t in ETF_TICKERS:
        rows.append({"source": "Yahoo Finance (yfinance)", "series_id": t, "role": "asset",
                     "block": "asset", "description": f"{t} daily prices",
                     "licence": "Yahoo terms of service - do not redistribute; regenerate with this script",
                     "redistributable": False, "url": f"https://finance.yahoo.com/quote/{t}"})
    pd.DataFrame(rows).to_csv(DATA_RAW / "sources.csv", index=False)


def main():
    # a freeze date in the future could still change
    if pd.Timestamp(FREEZE_DATE) >= pd.Timestamp(datetime.now(timezone.utc).date()):
        sys.exit(f"FREEZE_DATE {FREEZE_DATE} must be in the past")

    load_dotenv(ROOT / ".env")
    api_key = os.environ.get("FRED_API_KEY")
    if not api_key:
        sys.exit("FRED_API_KEY not found: put it in a file .env in the repository root (see README)")

    DATA_RAW.mkdir(parents=True, exist_ok=True)

    print("1. Macro series, all vintages (ALFRED)")
    vintages = download_fred(MACRO_SERIES, api_key, all_vintages=True)
    vintages.to_csv(DATA_RAW / "alfred_vintages.csv", index=False)

    print("2. Macro series, latest values")
    latest_values(vintages).to_csv(DATA_RAW / "fred_revised.csv", index=False)

    print("3. Market series")
    market = download_fred(MARKET_SERIES, api_key, all_vintages=False)
    market = market[["series_id", "date", "value"]]
    # split by licence: only the public file goes on GitHub
    public = market["series_id"].isin([s for s, v in MARKET_SERIES.items() if v[3]])
    market[public].to_csv(DATA_RAW / "fred_market_public.csv", index=False)
    market[~public].to_csv(DATA_RAW / "fred_market_thirdparty.csv", index=False)

    print("4. ETF prices (Yahoo Finance)")
    download_etfs(ETF_TICKERS).to_csv(DATA_RAW / "etf_daily.csv", index=False)

    write_sources()
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    (DATA_RAW / "download_log.txt").write_text(
        f"Freeze date: {FREEZE_DATE}\nDownloaded: {now}\n"
        f"pandas {pd.__version__}, numpy {np.__version__}, yfinance {yf.__version__}\n")
    print(f"Done, files in {DATA_RAW}")


if __name__ == "__main__":
    main()
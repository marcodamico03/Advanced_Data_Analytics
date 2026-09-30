"""
Stage 1 - Download raw data (no cleaning, no transformations).

ADA 2026 - Advanced Data Analytics, HEC Lausanne
Author: Marco D'Amico

Reproducibility: the freeze date
--------------------------------
Every FRED/ALFRED series is requested "as it was known on FREEZE_DATE"
(realtime_end = FREEZE_DATE, observation_end = FREEZE_DATE). Past vintages
never change, so anyone running this script at any later moment, with any API
key, receives byte-identical FRED files. ETF prices from Yahoo cannot be
frozen through the API; they are cut at FREEZE_DATE and checked against the
return statistics stored in data/manifest.json (see below).

Output (data/raw/)
------------------
alfred_vintages.csv       macro series, every vintage known on FREEZE_DATE
fred_revised.csv          macro series, the vintage in force on FREEZE_DATE
fred_market_public.csv    T10Y2Y as of FREEZE_DATE          (descriptive only)
fred_market_thirdparty.csv BAA10Y, NFCI as of FREEZE_DATE   (not redistributable)
etf_daily.csv             SPY, TLT, GLD, DBC daily prices    (not redistributable)
sources.csv               source, role and licence of every series
download_log.txt          freeze date, download time, package versions

Manifest (data/manifest.json, committed)
----------------------------------------
SHA-256 of every FRED file and return statistics of every ETF, as obtained by
the author. Each run compares the new download with it and reports MATCH or
MISMATCH. It contains no data, only fingerprints, so it can be public.

Usage (from the repository root)
--------------------------------
Put a free FRED API key (https://fred.stlouisfed.org/docs/api/api_key.html)
in a file named .env, one line:  FRED_API_KEY=your_key_here
    python 01_get_data.py                     # download and verify
    python 01_get_data.py --write-manifest    # author only: (re)create manifest
"""

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from dotenv import load_dotenv

# Reads FRED_API_KEY from a local .env file (listed in .gitignore, never committed).
load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
FREEZE_DATE = "2026-09-28"  # must be in the past: all data "as known on" this date
OBS_START = "2000-01-01"    # history before the 2006 sample, for 12-month changes

RAW_DIR = Path("data/raw")
MANIFEST = Path("data/manifest.json")
API_URL = "https://api.stlouisfed.org/fred/series/observations"

PUBLIC = "US government source - public domain (verify on the FRED series page)"
THIRD_PARTY = "Third-party copyright - do not redistribute; regenerate with this script"

# Macro variables used to DEFINE the regimes (growth / labour / inflation).
MACRO_SERIES = {
    # series_id: (block, description, licence, redistributable)
    "INDPRO":   ("growth",    "Industrial Production Index",                 PUBLIC, True),
    "TCU":      ("growth",    "Capacity Utilization: Total Industry",        PUBLIC, True),
    "RRSFS":    ("growth",    "Advance Real Retail and Food Services Sales", PUBLIC, True),
    "HOUST":    ("growth",    "Housing Starts: Total",                       PUBLIC, True),
    "PAYEMS":   ("labour",    "All Employees, Total Nonfarm",                PUBLIC, True),
    "UNRATE":   ("labour",    "Unemployment Rate",                           PUBLIC, True),
    "ICSA":     ("labour",    "Initial Claims (weekly; vintages only from 2009)", PUBLIC, True),
    "CPIAUCSL": ("inflation", "CPI, All Items",                              PUBLIC, True),
    "CPILFESL": ("inflation", "CPI, Less Food and Energy (core)",            PUBLIC, True),
    "PPIACO":   ("inflation", "PPI, All Commodities",                        PUBLIC, True),
}

# Market-based variables: NOT used to define regimes (circularity), descriptive only.
MARKET_SERIES = {
    "T10Y2Y": ("market", "10Y minus 2Y Treasury spread",           PUBLIC, True),
    "BAA10Y": ("market", "Moody's Baa yield minus 10Y Treasury",   THIRD_PARTY, False),
    "NFCI":   ("market", "Chicago Fed National Financial Conditions Index",
               "Federal Reserve Bank of Chicago - verify terms before redistributing", False),
}

# SPY equities, TLT long Treasuries, GLD gold, DBC broad commodities.
ETF_TICKERS = ["SPY", "TLT", "GLD", "DBC"]

FRED_FILES = ["alfred_vintages.csv", "fred_revised.csv",
              "fred_market_public.csv", "fred_market_thirdparty.csv"]


# ---------------------------------------------------------------------------
# FRED / ALFRED
# ---------------------------------------------------------------------------
def fred_observations(series_id, api_key, realtime_start):
    """
    Download one series as known between realtime_start and FREEZE_DATE.

    realtime_start = "1776-07-04" -> every vintage (ALFRED real-time data).
    realtime_start = FREEZE_DATE  -> only the vintage in force on FREEZE_DATE.
    Each row is a value valid between realtime_start and realtime_end; rows
    still valid on FREEZE_DATE carry realtime_end = FREEZE_DATE.
    Handles pagination and rate limiting; raises on any other error.
    """
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
            if resp.status_code == 429:          # too many requests: wait and retry
                time.sleep(20)
                continue
            if resp.status_code != 200:
                # The API explains the problem in the body; show it (never the key).
                raise RuntimeError(f"{series_id}: HTTP {resp.status_code} - "
                                   f"{resp.text[:300]}".replace(api_key, "***"))
            break
        else:
            raise RuntimeError(f"{series_id}: rate limit not cleared after 3 attempts")

        payload = resp.json()
        batch = payload["observations"]
        rows.extend(batch)
        offset += len(batch)
        if not batch or offset >= payload["count"]:
            break

    if not rows:
        raise RuntimeError(f"{series_id}: no observations known on {FREEZE_DATE}")

    df = pd.DataFrame(rows)
    df["value"] = pd.to_numeric(df["value"].replace(".", np.nan), errors="coerce")  # "." = missing
    df.insert(0, "series_id", series_id)
    # Dates stay ISO strings: they sort correctly and avoid out-of-range datetimes.
    return df[["series_id", "date", "realtime_start", "realtime_end", "value"]]


def download_fred_block(series_map, api_key, all_vintages):
    """Download every series in series_map, stacked in long format, sorted."""
    frames = []
    start = "1776-07-04" if all_vintages else FREEZE_DATE
    for sid in series_map:
        df = fred_observations(sid, api_key, realtime_start=start)
        frames.append(df)
        extra = (f", {df['realtime_start'].nunique():>4} vintages from {df['realtime_start'].min()}"
                 if all_vintages else "")
        print(f"  [OK] {sid:<9} {len(df):>7} rows, "
              f"{df['date'].min()} -> {df['date'].max()}{extra}")
        time.sleep(0.6)  # stay well below the API limit of 120 requests/minute
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["series_id", "date", "realtime_start"]).reset_index(drop=True)


def as_of_freeze(vintages):
    """The vintage in force on FREEZE_DATE, taken from the full vintage table."""
    live = vintages[(vintages["realtime_start"] <= FREEZE_DATE)
                    & (vintages["realtime_end"] >= FREEZE_DATE)]
    if live.duplicated(["series_id", "date"]).any():
        raise RuntimeError("More than one value in force on FREEZE_DATE for some dates")
    return live[["series_id", "date", "value"]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# ETFs (Yahoo Finance)
# ---------------------------------------------------------------------------
def download_etfs(tickers):
    """Daily Adj Close, Close and Volume up to FREEZE_DATE, long format."""
    end = (pd.Timestamp(FREEZE_DATE) + timedelta(days=1)).strftime("%Y-%m-%d")  # end is exclusive
    raw = yf.download(tickers, start=OBS_START, end=end, interval="1d",
                      auto_adjust=False, actions=False, progress=False)
    if raw.empty:
        raise RuntimeError("yfinance returned no data")
    raw.columns.names = ["field", "ticker"]
    df = (raw[["Adj Close", "Close", "Volume"]]
          .stack("ticker", future_stack=True)
          .reset_index()
          .rename(columns={"Date": "date"})
          .dropna(subset=["Adj Close"])
          .sort_values(["ticker", "date"])
          .reset_index(drop=True))

    missing = set(tickers) - set(df["ticker"])
    if missing:
        raise RuntimeError(f"No price data for: {sorted(missing)}")
    for t, g in df.groupby("ticker"):
        print(f"  [OK] {t:<4} {len(g):>5} days, {g['date'].min().date()} -> {g['date'].max().date()}")
    return df


def etf_fingerprint(etf):
    """
    Return statistics per ETF. Adjusted prices are rescaled after every dividend,
    so prices are not comparable across downloads; daily log returns are.
    """
    out = {}
    for t, g in etf.groupby("ticker"):
        r = np.log(g["Adj Close"]).diff().dropna()
        out[t] = {"n_days": int(len(g)),
                  "first": str(g["date"].min().date()),
                  "last": str(g["date"].max().date()),
                  "sum_log_ret": float(r.sum()),
                  "sum_abs_log_ret": float(r.abs().sum()),
                  "sum_sq_log_ret": float((r ** 2).sum())}
    return out


# ---------------------------------------------------------------------------
# Manifest: write (author) or verify (everyone else)
# ---------------------------------------------------------------------------
def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_manifest(etf):
    return {"freeze_date": FREEZE_DATE,
            "fred_files_sha256": {f: sha256(RAW_DIR / f) for f in FRED_FILES},
            "etf_returns": etf_fingerprint(etf)}


def verify(current, reference):
    """Compare this download with the committed manifest. Returns True if all match."""
    ok = True
    if current["freeze_date"] != reference["freeze_date"]:
        print(f"  [MISMATCH] freeze date {current['freeze_date']} vs "
              f"manifest {reference['freeze_date']}")
        return False

    for f, h in reference["fred_files_sha256"].items():
        same = current["fred_files_sha256"].get(f) == h
        ok &= same
        print(f"  [{'MATCH' if same else 'MISMATCH'}] {f} (SHA-256)")

    for t, ref in reference["etf_returns"].items():
        cur = current["etf_returns"].get(t)
        same = (cur is not None
                and all(cur[k] == ref[k] for k in ("n_days", "first", "last"))
                and all(np.isclose(cur[k], ref[k], rtol=1e-6, atol=1e-4)
                        for k in ("sum_log_ret", "sum_abs_log_ret", "sum_sq_log_ret")))
        ok &= same
        print(f"  [{'MATCH' if same else 'MISMATCH'}] {t} daily returns")
        if not same and cur is not None:
            print(f"      now:      {cur}\n      manifest: {ref}")
    return ok


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------
def write_sources():
    rows = []
    for role, mapping in [("regime", MACRO_SERIES), ("descriptive", MARKET_SERIES)]:
        for sid, (block, desc, licence, redist) in mapping.items():
            rows.append({"source": "FRED/ALFRED", "series_id": sid, "role": role, "block": block,
                         "description": desc, "licence": licence, "redistributable": redist,
                         "url": f"https://fred.stlouisfed.org/series/{sid}"})
    for t in ETF_TICKERS:
        rows.append({"source": "Yahoo Finance (yfinance)", "series_id": t, "role": "asset",
                     "block": "asset", "description": f"{t} daily prices",
                     "licence": "Yahoo terms of service - do not redistribute; regenerate with this script",
                     "redistributable": False, "url": f"https://finance.yahoo.com/quote/{t}"})
    pd.DataFrame(rows).to_csv(RAW_DIR / "sources.csv", index=False)


# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Stage 1: download raw data.")
    parser.add_argument("--write-manifest", action="store_true",
                        help="author only: overwrite data/manifest.json with this download")
    args = parser.parse_args()

    if pd.Timestamp(FREEZE_DATE) >= pd.Timestamp(datetime.now(timezone.utc).date()):
        sys.exit(f"FREEZE_DATE {FREEZE_DATE} is not in the past: the data could still change.")

    api_key = os.environ.get("FRED_API_KEY")
    if not api_key:
        sys.exit("FRED_API_KEY is not set. Get a free key at "
                 "https://fred.stlouisfed.org/docs/api/api_key.html and create a file\n"
                 ".env in the repository root containing one line:\n"
                 "    FRED_API_KEY=your_key_here")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Freeze date: {FREEZE_DATE} (all data as known on this date)\n")

    print("1. Macro series - all vintages (ALFRED real-time data)")
    vintages = download_fred_block(MACRO_SERIES, api_key, all_vintages=True)
    vintages.to_csv(RAW_DIR / "alfred_vintages.csv", index=False)

    print("\n2. Macro series - vintage in force on the freeze date (revised data)")
    revised = as_of_freeze(vintages)
    revised.to_csv(RAW_DIR / "fred_revised.csv", index=False)
    for sid, g in revised.groupby("series_id"):
        print(f"  [OK] {sid:<9} {len(g):>7} rows, {g['date'].min()} -> {g['date'].max()}")

    print("\n3. Market-based series (descriptive only)")
    market = download_fred_block(MARKET_SERIES, api_key, all_vintages=False)
    market = market[["series_id", "date", "value"]]
    redist = {sid for sid, v in MARKET_SERIES.items() if v[3]}
    market[market["series_id"].isin(redist)].to_csv(RAW_DIR / "fred_market_public.csv", index=False)
    market[~market["series_id"].isin(redist)].to_csv(RAW_DIR / "fred_market_thirdparty.csv", index=False)

    print("\n4. ETF daily prices (Yahoo Finance)")
    etf = download_etfs(ETF_TICKERS)
    etf.to_csv(RAW_DIR / "etf_daily.csv", index=False)

    write_sources()
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    (RAW_DIR / "download_log.txt").write_text(
        f"Freeze date: {FREEZE_DATE}\nDownloaded: {stamp}\n"
        f"pandas {pd.__version__}, numpy {np.__version__}, yfinance {yf.__version__}, "
        f"requests {requests.__version__}\n"
    )

    print("\n5. Reproducibility check")
    current = build_manifest(etf)
    if args.write_manifest or not MANIFEST.exists():
        MANIFEST.write_text(json.dumps(current, indent=2) + "\n")
        print(f"  Manifest written to {MANIFEST}. Commit it: it holds fingerprints, not data.")
    else:
        if not verify(current, json.loads(MANIFEST.read_text())):
            sys.exit("\nThe download differs from the author's (see MISMATCH above). "
                     "Results may not reproduce exactly.")
        print("  All files match the author's download.")

    print(f"\nDone. Raw files in {RAW_DIR}/ (downloaded {stamp})")


if __name__ == "__main__":
    main()
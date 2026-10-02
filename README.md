# Macro regimes in real time and risk parity

Advanced Data Analytics, HEC Lausanne, autumn 2026 — Marco D'Amico

The project asks whether US macroeconomic data, **as it was known at the time**, can identify the regimes in which the stock–bond correlation turns positive, and whether a risk-parity portfolio that adapts to those regimes beats a static one. Regimes are found with unsupervised learning (PCA, K-Means, Gaussian mixtures), built twice: from real-time data (ALFRED vintages) and from today's revised data.

## Data

| File in `data/raw/` | Source | In this repository? |
|---|---|---|
| `alfred_vintages.csv`, `fred_revised.csv` | FRED / ALFRED, St. Louis Fed (US government data) | Yes, public domain |
| `fred_market_public.csv` (T10Y2Y) | FRED (US Treasury data) | Yes, public domain |
| `fred_market_thirdparty.csv` (BAA10Y, NFCI) | FRED (Moody's; Chicago Fed) | No — regenerate with step 1 |
| `etf_daily.csv` (SPY, TLT, GLD, DBC) | Yahoo Finance via `yfinance` | No — Yahoo terms forbid redistribution; regenerate with step 1 |

`data/raw/sources.csv` lists every series with its licence. Because all FRED data is requested as known on the freeze date, a new download is identical to the author's; step 1 prints `MATCH` for every file when it is.



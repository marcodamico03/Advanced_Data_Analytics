"""
Functions used by more than one stage (Requirement 4.9).

Only loading and saving for now; functions are added here when a second
notebook needs them, never copied between notebooks.
"""

import pandas as pd

import config


# ---------------------------------------------------------------------------
# Loading raw data (output of 01_get_data.py)
# ---------------------------------------------------------------------------
def load_alfred_vintages():
    """
    Every vintage of the macro series, long format.

    One row = the value of `series_id` for month `date`, as published between
    `realtime_start` and `realtime_end`. Dates stay ISO strings, which sort and
    compare correctly.
    """
    return pd.read_csv(config.DATA_RAW / "alfred_vintages.csv")


def load_revised_macro():
    """
    Macro series as known on the freeze date (latest revision), long format:
    columns series_id, date, value. Kept long because frequencies differ
    (ICSA is weekly, the rest monthly); aligning them is stage 2's job.
    """
    return pd.read_csv(config.DATA_RAW / "fred_revised.csv", parse_dates=["date"])


def load_market_series():
    """
    T10Y2Y, BAA10Y (daily) and NFCI (weekly) as known on the freeze date,
    long format: columns series_id, date, value.
    """
    files = ["fred_market_public.csv", "fred_market_thirdparty.csv"]
    return pd.concat([pd.read_csv(config.DATA_RAW / f, parse_dates=["date"]) for f in files],
                     ignore_index=True)


def load_etf_prices(field="Adj Close"):
    """
    Daily ETF prices, wide: one column per ticker.

    Use "Adj Close" (dividends reinvested) for returns, "Close" only to see
    the price actually traded.
    """
    df = pd.read_csv(config.DATA_RAW / "etf_daily.csv", parse_dates=["date"])
    return df.pivot(index="date", columns="ticker", values=field)


# ---------------------------------------------------------------------------
# Saving outputs
# ---------------------------------------------------------------------------
def save_processed(df, name):
    """Save an intermediate dataset to data/processed/<name>.csv for the next stage."""
    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    path = config.DATA_PROCESSED / f"{name}.csv"
    df.to_csv(path)
    return path


def load_processed(name, **kwargs):
    """Load a dataset saved by save_processed; extra arguments go to pd.read_csv."""
    return pd.read_csv(config.DATA_PROCESSED / f"{name}.csv", **kwargs)


def save_figure(fig, name):
    """Save a matplotlib figure to figures/<name>.pdf, the format the paper includes."""
    config.FIGURES.mkdir(parents=True, exist_ok=True)
    path = config.FIGURES / f"{name}.pdf"
    fig.savefig(path, bbox_inches="tight")
    return path


def save_table(df, name, float_format="%.3f"):
    """Save a table to tables/<name>.csv."""
    config.TABLES.mkdir(parents=True, exist_ok=True)
    path = config.TABLES / f"{name}.csv"
    df.to_csv(path, float_format=float_format)
    return path

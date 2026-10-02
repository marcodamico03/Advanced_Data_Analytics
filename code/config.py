"""
Settings shared by all stages: paths, freeze date, series.
"""

from pathlib import Path

# paths are built from this file's location, so they work from any folder
ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "figures"
TABLES = ROOT / "tables"

FREEZE_DATE = "2026-09-28"  # all FRED data as known on this date
OBS_START = "2000-01-01"    # earlier than the sample, needed for 12-month changes
RANDOM_SEED = 42

PUBLIC = "US government source - public domain (verify on the FRED series page)"
THIRD_PARTY = "Third-party copyright - do not redistribute; regenerate with this script"

# macro series used to build the regimes: id -> (block, description, licence, can be shared)
MACRO_SERIES = {
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

# market series, only to describe the regimes (not used to build them)
MARKET_SERIES = {
    "T10Y2Y": ("market", "10Y minus 2Y Treasury spread",           PUBLIC, True),
    "BAA10Y": ("market", "Moody's Baa yield minus 10Y Treasury",   THIRD_PARTY, False),
    "NFCI":   ("market", "Chicago Fed National Financial Conditions Index",
               "Federal Reserve Bank of Chicago - verify terms before redistributing", False),
}

# stocks, long Treasuries, gold, commodities
ETF_TICKERS = ["SPY", "TLT", "GLD", "DBC"]
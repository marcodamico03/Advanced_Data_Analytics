"""
Functions used by more than one stage (Requirement 4.9).

A function is moved here as soon as a second notebook needs it, never copied
between notebooks.
"""

import math

import numpy as np
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
# Statistics
# ---------------------------------------------------------------------------
def ols_hac(y, X, lags=None):
    """
    OLS with Newey-West (1987) standard errors, robust to autocorrelation and
    heteroskedasticity. y: Series, X: DataFrame of regressors (a constant is added).
    lags=None uses the usual rule floor(4 * (T/100)^(2/9)).
    Returns a table (coef, se, t, p) and a dict with R2 and number of observations.
    """
    data = pd.concat([y, X], axis=1).dropna()
    yv = data.iloc[:, 0].to_numpy()
    Xv = np.column_stack([np.ones(len(data)), data.iloc[:, 1:].to_numpy()])
    names = ["const"] + list(X.columns)
    T = len(yv)
    if lags is None:
        lags = int(np.floor(4 * (T / 100) ** (2 / 9)))

    XtX_inv = np.linalg.inv(Xv.T @ Xv)
    beta = XtX_inv @ Xv.T @ yv
    u = yv - Xv @ beta

    # long-run covariance of x_t * u_t, Bartlett weights
    g = Xv * u[:, None]
    S = g.T @ g
    for j in range(1, lags + 1):
        w = 1 - j / (lags + 1)
        G = g[j:].T @ g[:-j]
        S += w * (G + G.T)
    V = XtX_inv @ S @ XtX_inv

    se = np.sqrt(np.diag(V))
    t = beta / se
    p = np.array([math.erfc(abs(x) / math.sqrt(2)) for x in t])   # two-sided, normal approximation
    table = pd.DataFrame({"coef": beta, "se": se, "t": t, "p": p}, index=names)
    r2 = 1 - (u @ u) / ((yv - yv.mean()) @ (yv - yv.mean()))
    return table, {"r2": r2, "nobs": T, "lags": lags, "cov": pd.DataFrame(V, index=names, columns=names)}


def wald_test(table, info, variables):
    """Joint test that the coefficients of `variables` are all zero, using the Newey-West covariance."""
    from scipy.stats import chi2
    b = table.loc[variables, "coef"].to_numpy()
    V = info["cov"].loc[variables, variables].to_numpy()
    stat = float(b @ np.linalg.solve(V, b))
    return stat, float(chi2.sf(stat, len(variables)))


def sharpe_diff_test(a, b, lags=None):
    """
    Annualised Sharpe ratio of a minus that of b (monthly excess returns), with the
    standard error of Ledoit and Wolf (2008): delta method, Newey-West covariance.
    Returns a dict: sharpe difference, se, p (two-sided).
    """
    from scipy.stats import norm
    x = pd.concat([a, b], axis=1).dropna().to_numpy()
    T = len(x)
    y = np.column_stack([x, x ** 2])
    m1, m2, g1, g2 = y.mean(axis=0)
    s1, s2 = np.sqrt(g1 - m1 ** 2), np.sqrt(g2 - m2 ** 2)
    diff = m1 / s1 - m2 / s2
    grad = np.array([g1 / s1 ** 3, -g2 / s2 ** 3, -m1 / (2 * s1 ** 3), m2 / (2 * s2 ** 3)])
    u = y - y.mean(axis=0)
    if lags is None:
        lags = int(np.floor(4 * (T / 100) ** (2 / 9)))
    S = u.T @ u / T
    for j in range(1, lags + 1):
        G = u[j:].T @ u[:-j] / T
        S += (1 - j / (lags + 1)) * (G + G.T)
    se = np.sqrt(grad @ S @ grad / T)
    return {"sharpe difference": np.sqrt(12) * diff, "se": np.sqrt(12) * se, "p": 2 * norm.sf(abs(diff / se))}


# ---------------------------------------------------------------------------
# Portfolios (stages 4 and 5)
# ---------------------------------------------------------------------------
def monthly_correlations(daily_returns, min_days=15):
    """
    Correlation matrix of the daily returns of each calendar month, using days
    where every asset has a return, if there are at least min_days of them.
    Returns (month-end dates, array months x assets x assets).
    """
    months, mats = [], []
    for m, g in daily_returns.resample("ME"):
        g = g.dropna()
        if len(g) >= min_days:
            months.append(m)
            mats.append(g.corr().to_numpy())
    return pd.DatetimeIndex(months), np.stack(mats)


def erc_weights(cov, tol=1e-12, max_iter=10_000):
    """Equal risk contribution weights, by cyclical coordinate descent (Griveau-Billion, Richard and Roncalli, 2013)."""
    n = len(cov)
    x = 1 / np.sqrt(np.diag(cov))
    for _ in range(max_iter):
        x_old = x.copy()
        for i in range(n):
            a = cov[i] @ x - cov[i, i] * x[i]
            x[i] = (-a + np.sqrt(a * a + 4 * cov[i, i] / n)) / (2 * cov[i, i])
        if np.abs(x - x_old).max() < tol:
            break
    return x / x.sum()


def backtest(positions, asset_returns, rf, cost_bps):
    """
    Monthly net returns of one strategy.

    positions: DataFrame, one row per decision date t (a month-end), one column per asset,
    held during month t+1; what is not invested earns rf, what is borrowed costs rf.
    During the month positions drift with prices; the next trade is the gap between the
    drifted and the new positions, and pays cost_bps per dollar traded.
    Returns a DataFrame indexed by the month held: net, turnover, exposure.
    """
    assets = list(positions.columns)
    out, drifted = [], np.zeros(len(assets))
    for t in positions.index:
        nxt = t + pd.offsets.MonthEnd(1)
        w = positions.loc[t].to_numpy(dtype=float)
        turnover = np.abs(w - drifted).sum()
        r = asset_returns.loc[nxt, assets].to_numpy()
        gross = w @ r + (1 - w.sum()) * rf[nxt]
        out.append({"date": nxt, "net": gross - cost_bps / 10_000 * turnover, "turnover": turnover,
                    "exposure": w.sum()})
        drifted = w * (1 + r) / (1 + gross)
    return pd.DataFrame(out).set_index("date")


def max_drawdown(r):
    """Largest fall from a previous peak of the wealth built from returns r."""
    wealth = (1 + r).cumprod()
    return (wealth / wealth.cummax() - 1).min()


def performance(bt, rf):
    """Summary of a backtest (output of backtest): excess returns over rf, annualised."""
    ex = bt["net"] - rf.reindex(bt.index)
    return {"excess return": 12 * ex.mean(), "volatility": np.sqrt(12) * ex.std(),
            "sharpe": np.sqrt(12) * ex.mean() / ex.std(), "max drawdown": max_drawdown(bt["net"]),
            "worst month": bt["net"].min(), "turnover per year": 12 * bt["turnover"].mean(),
            "average exposure": bt["exposure"].mean(), "months": len(bt)}


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
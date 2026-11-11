"""Price data loaders: synthetic (offline), CSV, and Yahoo Finance."""
from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_TICKERS = ["SPY", "QQQ", "TLT", "GLD", "EEM"]

_PRESET_MU = {"SPY": 0.09, "QQQ": 0.11, "TLT": 0.03, "GLD": 0.05, "EEM": 0.07}
_PRESET_VOL = {"SPY": 0.16, "QQQ": 0.21, "TLT": 0.14, "GLD": 0.15, "EEM": 0.20}
_PRESET_CORR = np.array([
    [1.00, 0.92, -0.30, 0.05, 0.75],
    [0.92, 1.00, -0.25, 0.05, 0.70],
    [-0.30, -0.25, 1.00, 0.20, -0.15],
    [0.05, 0.05, 0.20, 1.00, 0.10],
    [0.75, 0.70, -0.15, 0.10, 1.00],
])


def synthetic_prices(tickers=None, n_days: int = 1500, seed: int = 42) -> pd.DataFrame:
    """Generate realistic-looking correlated prices with fat tails and
    volatility clustering (2-state calm/turbulent regime, Student-t shocks).

    Useful for offline demos and for showing that fat-tailed data makes a
    Gaussian model under-estimate risk.
    """
    tickers = list(tickers or DEFAULT_TICKERS)
    n = len(tickers)
    rng = np.random.default_rng(seed)

    if tickers == DEFAULT_TICKERS[:n]:
        mu = np.array([_PRESET_MU[t] for t in tickers])
        vol = np.array([_PRESET_VOL[t] for t in tickers])
        corr = _PRESET_CORR[:n, :n]
    else:
        mu = rng.uniform(0.03, 0.12, n)
        vol = rng.uniform(0.12, 0.30, n)
        a = rng.normal(size=(n, n + 3))
        cov = a @ a.T
        d = np.sqrt(np.diag(cov))
        corr = cov / np.outer(d, d)

    # regime: 0 = calm, 1 = turbulent (shared across assets -> vol clustering)
    p_stay = np.array([0.985, 0.94])
    regime = np.zeros(n_days, dtype=int)
    for t in range(1, n_days):
        regime[t] = regime[t - 1] if rng.random() < p_stay[regime[t - 1]] else 1 - regime[t - 1]
    vol_mult = np.where(regime == 1, 2.2, 0.85)

    df = 5.0
    L = np.linalg.cholesky(corr)
    z = rng.standard_normal((n_days, n)) @ L.T
    w = rng.chisquare(df, size=(n_days, 1)) / df
    shocks = z / np.sqrt(w) * np.sqrt((df - 2) / df)

    daily_vol = vol / np.sqrt(252)
    rets = (mu / 252 - 0.5 * daily_vol**2) + shocks * daily_vol * vol_mult[:, None]
    idx = pd.bdate_range("2019-01-01", periods=n_days)
    prices = 100 * np.exp(np.cumsum(rets, axis=0))
    return pd.DataFrame(prices, index=idx, columns=tickers)


def load_prices_csv(path: str) -> pd.DataFrame:
    """CSV with a date index in column 0 and one price column per asset."""
    df = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    return df.dropna(how="all").ffill().dropna()


def load_prices_yfinance(tickers, start="2018-01-01", end=None) -> pd.DataFrame:
    """Download adjusted close prices (requires `pip install yfinance`)."""
    try:
        import yfinance as yf
    except ImportError as e:  # pragma: no cover
        raise ImportError("pip install yfinance to use --source yfinance") from e
    px = yf.download(list(tickers), start=start, end=end, auto_adjust=True, progress=False)["Close"]
    if isinstance(px, pd.Series):
        px = px.to_frame(list(tickers)[0])
    return px[list(tickers)].dropna()

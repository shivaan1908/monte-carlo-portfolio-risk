"""Parameter estimation: mean, shrunk covariance, tail-fatness (t d.o.f.)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices / prices.shift(1)).dropna()


def ledoit_wolf_corr(X: np.ndarray) -> tuple[np.ndarray, float]:
    """Ledoit-Wolf shrinkage of a correlation matrix toward the identity.

    X: standardised returns (n_obs, n_assets). Returns (corr, shrinkage).
    Shrinking stabilises the Cholesky factor and reduces estimation noise,
    which matters when n_assets is large relative to n_obs.
    """
    n, p = X.shape
    X = X - X.mean(0)
    S = X.T @ X / n
    m = np.trace(S) / p
    d2 = ((S - m * np.eye(p)) ** 2).sum() / p
    b2 = ((X**2).sum(1) ** 2).sum() - n * (S**2).sum()
    b2 = min(b2 / (n**2 * p), d2)
    k = b2 / d2 if d2 > 0 else 0.0
    corr = k * m * np.eye(p) + (1 - k) * S
    d = np.sqrt(np.diag(corr))
    return corr / np.outer(d, d), float(k)


def fit_dof(portfolio_returns: np.ndarray, lo: float = 3.5, hi: float = 30.0) -> float:
    """MLE Student-t degrees of freedom of portfolio returns, clipped to a sane range."""
    df, _, _ = stats.t.fit(portfolio_returns)
    return float(np.clip(df, lo, hi))


@dataclass
class Estimates:
    tickers: list
    mu: np.ndarray          # daily mean log return
    cov: np.ndarray         # daily covariance of log returns
    df: float               # Student-t d.o.f.
    shrinkage: float
    hist: np.ndarray        # raw historical log returns (for bootstrap)


def build_estimates(returns: pd.DataFrame, weights: np.ndarray, shrink: bool = True) -> Estimates:
    R = returns.values
    mu = R.mean(0)
    vols = R.std(0, ddof=1)
    Z = (R - mu) / vols
    if shrink:
        corr, k = ledoit_wolf_corr(Z)
    else:
        corr, k = np.corrcoef(Z, rowvar=False), 0.0
    cov = corr * np.outer(vols, vols)
    port = (np.exp(R) - 1) @ weights
    return Estimates(list(returns.columns), mu, cov, fit_dof(port), k, R)

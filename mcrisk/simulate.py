"""Monte Carlo engines. All return daily log-return paths of shape (n_sims, horizon, n_assets)."""
from __future__ import annotations

import numpy as np

METHODS = ("gaussian", "student_t", "bootstrap")


def simulate_log_returns(mu, cov, n_sims, horizon, method="gaussian", df=6.0, hist=None, seed=None):
    """
    gaussian   : r ~ N(mu, cov), i.i.d. daily. Cholesky-correlated normals.
    student_t  : multivariate t with `df` d.o.f., rescaled so Cov(r) == cov. Fat tails +
                 tail dependence (one chi-square mixing draw per sim-day scales ALL assets,
                 so assets crash together).
    bootstrap  : resample whole historical days (preserves empirical skew, kurtosis and
                 cross-asset dependence; ignores volatility clustering).
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    rng = np.random.default_rng(seed)
    n = len(mu)

    if method == "bootstrap":
        if hist is None:
            raise ValueError("bootstrap needs `hist`")
        idx = rng.integers(0, len(hist), size=(n_sims, horizon))
        return hist[idx]

    L = np.linalg.cholesky(cov + 1e-12 * np.eye(n))
    z = rng.standard_normal((n_sims, horizon, n)) @ L.T
    if method == "student_t":
        if df <= 2:
            raise ValueError("df must be > 2 for finite variance")
        w = rng.chisquare(df, size=(n_sims, horizon, 1)) / df
        z = z / np.sqrt(w) * np.sqrt((df - 2) / df)
    return mu + z


def portfolio_paths(log_ret, weights, value):
    """Buy-and-hold portfolio (weights fixed at t0, no rebalancing).

    Returns
    -------
    values : (n_sims, horizon)  portfolio value path
    growth_T : (n_sims, n_assets) terminal gross return of each asset
    """
    growth = np.exp(np.cumsum(log_ret, axis=1))
    values = value * (growth * weights).sum(axis=2)
    return values, growth[:, -1, :]

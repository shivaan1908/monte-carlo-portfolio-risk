"""High-level entry point: estimates + weights -> simulated risk."""
from __future__ import annotations

import numpy as np

from .estimation import Estimates
from .risk import summarize, component_es
from .simulate import simulate_log_returns, portfolio_paths


def monte_carlo(est: Estimates, weights, value=1_000_000.0, horizon=10, n_sims=100_000,
                method="student_t", seed=0, cov=None, mu=None, alphas=(0.95, 0.99),
                chunk=50_000):
    """Run a simulation and return (summary dict, pnl array, values, growth_T).

    cov / mu override the estimated parameters (used for stress scenarios).
    Simulation runs in chunks to bound memory for large n_sims * horizon.
    """
    weights = np.asarray(weights, float)
    weights = weights / weights.sum()
    cov = est.cov if cov is None else cov
    mu = est.mu if mu is None else mu

    vals, grows, done, i = [], [], 0, 0
    while done < n_sims:
        n = min(chunk, n_sims - done)
        lr = simulate_log_returns(mu, cov, n, horizon, method, est.df, est.hist, seed=seed + i)
        v, g = portfolio_paths(lr, weights, value)
        vals.append(v); grows.append(g)
        done += n; i += 1
    values, growth_T = np.concatenate(vals), np.concatenate(grows)

    summary = summarize(values, growth_T, weights, value, alphas)
    summary["component_ES_99"] = dict(zip(est.tickers, component_es(growth_T, weights, value, 0.99).tolist()))
    return summary, values[:, -1] - value, values, growth_T

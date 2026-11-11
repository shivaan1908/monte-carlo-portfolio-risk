"""Risk metrics computed from simulated P&L."""
from __future__ import annotations

import numpy as np
from scipy import stats


def var_es(pnl: np.ndarray, alpha: float = 0.99) -> tuple[float, float]:
    """VaR and Expected Shortfall (CVaR) as positive loss amounts."""
    losses = -pnl
    var = float(np.quantile(losses, alpha))
    es = float(losses[losses >= var].mean())
    return var, es


def bootstrap_ci(pnl, alpha=0.99, n_boot=100, level=0.95, seed=0):
    """Percentile CI on VaR: quantifies Monte Carlo (sampling) error."""
    rng = np.random.default_rng(seed)
    n = len(pnl)
    est = [np.quantile(-pnl[rng.integers(0, n, n)], alpha) for _ in range(n_boot)]
    lo, hi = np.quantile(est, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(lo), float(hi)


def max_drawdown(values: np.ndarray, v0: float) -> np.ndarray:
    """Max peak-to-trough drawdown (negative fraction) for each simulated path."""
    full = np.concatenate([np.full((values.shape[0], 1), v0), values], axis=1)
    peak = np.maximum.accumulate(full, axis=1)
    return ((full - peak) / peak).min(axis=1)


def component_es(growth_T, weights, value, alpha=0.99):
    """Euler allocation of ES to assets: E[-asset P&L | portfolio loss >= VaR].
    Components sum exactly to portfolio ES, so you can see *which positions* drive tail risk."""
    asset_pnl = value * weights * (growth_T - 1)
    pnl = asset_pnl.sum(1)
    var, _ = var_es(pnl, alpha)
    tail = -pnl >= var
    return -asset_pnl[tail].mean(0)


def summarize(values, growth_T, weights, value, alphas=(0.95, 0.99)) -> dict:
    pnl = values[:, -1] - value
    out = {
        "mean_pnl": float(pnl.mean()),
        "std_pnl": float(pnl.std()),
        "skew": float(stats.skew(pnl)),
        "excess_kurtosis": float(stats.kurtosis(pnl)),
        "prob_loss": float((pnl < 0).mean()),
    }
    for a in alphas:
        k = int(round(a * 100))
        var, es = var_es(pnl, a)
        lo, hi = bootstrap_ci(pnl, a)
        out[f"VaR_{k}"], out[f"ES_{k}"] = var, es
        out[f"VaR_{k}_ci_lo"], out[f"VaR_{k}_ci_hi"] = lo, hi
    dd = max_drawdown(values, value)
    out["median_max_drawdown"] = float(np.median(dd))
    out["p95_max_drawdown"] = float(np.quantile(dd, 0.05))
    return out

"""Rolling VaR backtest with Kupiec (coverage) and Christoffersen (independence) tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .estimation import build_estimates
from .model import monte_carlo


def kupiec_pof(n_exc: int, n_obs: int, p: float) -> tuple[float, float]:
    """Proportion-of-failures LR test. H0: exception rate == p. Returns (LR, p-value)."""
    x, n = n_exc, n_obs
    ph = x / n
    ll0 = (n - x) * np.log(1 - p) + x * np.log(p)
    ll1 = (n - x) * np.log(1 - ph) if x == 0 else (x * np.log(ph) + (n - x) * np.log(1 - ph) if x < n else x * np.log(ph))
    lr = max(-2 * (ll0 - ll1), 0.0)
    return float(lr), float(stats.chi2.sf(lr, 1))


def christoffersen_ind(hits: np.ndarray) -> tuple[float, float]:
    """Independence LR test: are exceptions clustered? Returns (LR, p-value)."""
    h = hits.astype(int)
    n00 = ((h[:-1] == 0) & (h[1:] == 0)).sum(); n01 = ((h[:-1] == 0) & (h[1:] == 1)).sum()
    n10 = ((h[:-1] == 1) & (h[1:] == 0)).sum(); n11 = ((h[:-1] == 1) & (h[1:] == 1)).sum()
    def ll(a, b):  # a*log(1-pi)+b*log(pi) safely
        tot = a + b
        if tot == 0: return 0.0
        pi = b / tot
        return (a * np.log(1 - pi) if a else 0.0) + (b * np.log(pi) if b else 0.0)
    pi = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)
    ll0 = (n00 + n10) * (np.log(1 - pi) if pi < 1 else 0) + (n01 + n11) * (np.log(pi) if pi > 0 else 0)
    ll1 = ll(n00, n01) + ll(n10, n11)
    lr = max(-2 * (ll0 - ll1), 0.0)
    return float(lr), float(stats.chi2.sf(lr, 1))


def rolling_var_backtest(returns: pd.DataFrame, weights, method="student_t", alpha=0.99,
                         window=500, step=5, n_sims=5000, seed=0) -> pd.DataFrame:
    """Re-estimate every `step` days on the trailing `window`, forecast 1-day VaR (as a
    fraction of portfolio value), and compare to the realised next-day portfolio return."""
    w = np.asarray(weights, float); w = w / w.sum()
    realised = (np.exp(returns.values) - 1) @ w
    rows = []
    for t in range(window, len(returns) - 1, step):
        est = build_estimates(returns.iloc[t - window:t], w)
        s, *_ = monte_carlo(est, w, value=1.0, horizon=1, n_sims=n_sims, method=method,
                            seed=seed + t, alphas=(alpha,))
        var = s[f"VaR_{int(round(alpha * 100))}"]
        for k in range(t, min(t + step, len(returns))):
            rows.append((returns.index[k], var, realised[k]))
    df = pd.DataFrame(rows, columns=["date", "var", "realised"]).set_index("date")
    df["hit"] = df["realised"] < -df["var"]
    return df


def evaluate(bt: pd.DataFrame, alpha: float) -> dict:
    p = 1 - alpha
    n, x = len(bt), int(bt["hit"].sum())
    lr_u, p_u = kupiec_pof(x, n, p)
    lr_i, p_i = christoffersen_ind(bt["hit"].values)
    return {"observations": n, "exceptions": x, "expected": round(n * p, 1),
            "exception_rate": x / n, "kupiec_p": p_u, "christoffersen_p": p_i,
            "pass_5pct": bool(p_u > 0.05 and p_i > 0.05)}

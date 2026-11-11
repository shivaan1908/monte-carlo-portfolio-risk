"""Stress scenarios: re-run the simulation under shocked covariance matrices."""
from __future__ import annotations

import numpy as np

SCENARIOS = {
    "baseline":            dict(vol_mult=1.0, corr_blend=0.0),
    "vol x1.5":            dict(vol_mult=1.5, corr_blend=0.0),
    "vol x2 + corr->1 50%": dict(vol_mult=2.0, corr_blend=0.5),
    "crisis (x3, corr 80%)": dict(vol_mult=3.0, corr_blend=0.8),
}


def stress_cov(cov: np.ndarray, vol_mult: float = 1.0, corr_blend: float = 0.0) -> np.ndarray:
    """Scale all vols and blend the correlation matrix toward all-ones (diversification
    collapses in a crisis). corr_blend=0 keeps correlations, 1 makes every pair perfectly correlated."""
    vols = np.sqrt(np.diag(cov))
    corr = cov / np.outer(vols, vols)
    corr = (1 - corr_blend) * corr + corr_blend * np.ones_like(corr)
    np.fill_diagonal(corr, 1.0)
    # keep PSD (blend of PSD matrices is PSD, but guard numerical noise)
    w, V = np.linalg.eigh(corr)
    corr = (V * np.clip(w, 1e-10, None)) @ V.T
    d = np.sqrt(np.diag(corr)); corr = corr / np.outer(d, d)
    v = vols * vol_mult
    return corr * np.outer(v, v)

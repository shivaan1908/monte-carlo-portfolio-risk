import numpy as np
import pytest
from scipy import stats

from mcrisk.backtest import christoffersen_ind, kupiec_pof
from mcrisk.estimation import Estimates, ledoit_wolf_corr
from mcrisk.model import monte_carlo
from mcrisk.risk import component_es, var_es
from mcrisk.simulate import portfolio_paths, simulate_log_returns
from mcrisk.stress import stress_cov

COV = np.array([[0.04, 0.012, 0.0], [0.012, 0.09, 0.01], [0.0, 0.01, 0.0225]]) / 252
MU = np.zeros(3)


def test_gaussian_var_matches_analytic():
    w = np.array([0.5, 0.3, 0.2])
    est = Estimates(["a", "b", "c"], MU, COV, 6.0, 0.0, np.zeros((10, 3)))
    s, *_ = monte_carlo(est, w, 1e6, horizon=1, n_sims=400_000, method="gaussian", seed=1, alphas=(0.99,))
    sigma = np.sqrt(w @ COV @ w)
    assert s["VaR_99"] == pytest.approx(1e6 * stats.norm.ppf(0.99) * sigma, rel=0.03)


@pytest.mark.parametrize("method", ["gaussian", "student_t"])
def test_simulated_covariance_matches_target(method):
    r = simulate_log_returns(MU, COV, 200_000, 1, method, df=6.0, seed=3)[:, 0, :]
    assert np.allclose(np.cov(r.T), COV, rtol=0.08, atol=1e-6)


def test_student_t_has_fatter_tails():
    g = simulate_log_returns(MU, COV, 300_000, 1, "gaussian", seed=4)[:, 0, 0]
    t = simulate_log_returns(MU, COV, 300_000, 1, "student_t", df=4.0, seed=4)[:, 0, 0]
    assert stats.kurtosis(t) > stats.kurtosis(g) + 1


def test_component_es_sums_to_total():
    w = np.array([0.5, 0.3, 0.2])
    lr = simulate_log_returns(MU, COV * 20, 100_000, 10, "student_t", seed=5)
    values, growth = portfolio_paths(lr, w, 1e6)
    pnl = values[:, -1] - 1e6
    _, es = var_es(pnl, 0.99)
    assert component_es(growth, w, 1e6, 0.99).sum() == pytest.approx(es, rel=1e-9)


def test_es_exceeds_var():
    pnl = np.random.default_rng(0).standard_normal(100_000)
    var, es = var_es(pnl, 0.99)
    assert es > var > 0


def test_stress_increases_risk_and_stays_psd():
    sc = stress_cov(COV, 2.0, 0.5)
    assert np.all(np.linalg.eigvalsh(sc) > 0)
    assert np.sqrt(np.diag(sc)) == pytest.approx(2 * np.sqrt(np.diag(COV)))


def test_ledoit_wolf_shrinkage_in_unit_interval():
    X = np.random.default_rng(0).standard_normal((60, 20))
    corr, k = ledoit_wolf_corr(X)
    assert 0 <= k <= 1 and np.allclose(np.diag(corr), 1)


def test_kupiec_accepts_correct_rate_rejects_wrong():
    assert kupiec_pof(10, 1000, 0.01)[1] > 0.9
    assert kupiec_pof(30, 1000, 0.01)[1] < 0.001


def test_christoffersen_detects_clustering():
    clustered = np.zeros(1000, bool); clustered[100:110] = True
    spread = np.zeros(1000, bool); spread[::100] = True
    assert christoffersen_ind(clustered)[1] < christoffersen_ind(spread)[1]

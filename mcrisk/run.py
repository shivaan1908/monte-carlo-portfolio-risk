"""CLI:  python -m mcrisk.run --help"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import report
from .backtest import evaluate, rolling_var_backtest
from .data import DEFAULT_TICKERS, load_prices_csv, load_prices_yfinance, synthetic_prices
from .estimation import build_estimates, log_returns
from .model import monte_carlo
from .simulate import METHODS
from .stress import SCENARIOS, stress_cov


def main(argv=None):
    ap = argparse.ArgumentParser(description="Monte Carlo portfolio risk model")
    ap.add_argument("--source", choices=["synthetic", "yfinance", "csv"], default="synthetic")
    ap.add_argument("--csv", help="price CSV path (for --source csv)")
    ap.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    ap.add_argument("--weights", nargs="+", type=float, help="default: equal weight")
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--value", type=float, default=1_000_000)
    ap.add_argument("--horizon", type=int, default=10, help="trading days")
    ap.add_argument("--sims", type=int, default=100_000)
    ap.add_argument("--alpha", type=float, default=0.99)
    ap.add_argument("--window", type=int, default=500, help="backtest estimation window")
    ap.add_argument("--no-backtest", action="store_true")
    ap.add_argument("--no-shrink", action="store_true")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="outputs")
    a = ap.parse_args(argv)

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    if a.source == "synthetic":
        prices = synthetic_prices(a.tickers, seed=42)
    elif a.source == "yfinance":
        prices = load_prices_yfinance(a.tickers, a.start)
    else:
        prices = load_prices_csv(a.csv)
    rets = log_returns(prices)
    tickers = list(rets.columns)
    w = np.array(a.weights if a.weights else [1 / len(tickers)] * len(tickers), float)
    if len(w) != len(tickers):
        raise SystemExit("weights must match number of assets")
    w = w / w.sum()

    est = build_estimates(rets, w, shrink=not a.no_shrink)
    print(f"\n{len(rets)} daily obs, {len(tickers)} assets | t d.o.f. = {est.df:.1f} | "
          f"correlation shrinkage = {est.shrinkage:.3f}")
    print("weights:", {t: round(float(x), 3) for t, x in zip(tickers, w)})

    k = int(round(a.alpha * 100))
    results = {}
    for m in METHODS:
        s, pnl, values, growth = monte_carlo(est, w, a.value, a.horizon, a.sims, m, a.seed,
                                             alphas=(0.95, a.alpha))
        results[m] = {"summary": s, "pnl": pnl}
        if m == "student_t":
            main_summary = s
    tbl = pd.DataFrame({m: {f"VaR {k}%": r["summary"][f"VaR_{k}"], f"ES {k}%": r["summary"][f"ES_{k}"],
                            "VaR 95%": r["summary"]["VaR_95"], "p95 max drawdown": r["summary"]["p95_max_drawdown"],
                            "excess kurtosis": r["summary"]["excess_kurtosis"]}
                        for m, r in results.items()})
    print(f"\n=== {a.horizon}-day risk on ${a.value:,.0f} ({a.sims:,} sims) ===")
    print(tbl.to_string(float_format=lambda x: f"{x:,.2f}" if abs(x) < 50 else f"{x:,.0f}"))

    print(f"\nComponent ES {k}% (student_t):")
    for t, v in main_summary["component_ES_99"].items():
        print(f"  {t:>6}: ${v:>10,.0f}")

    # stress
    stress = {}
    for name, p in SCENARIOS.items():
        s, *_ = monte_carlo(est, w, a.value, a.horizon, a.sims // 2, "student_t", a.seed,
                            cov=stress_cov(est.cov, **p), alphas=(0.95, a.alpha))
        stress[name] = s
    print("\n=== Stress scenarios (student_t) ===")
    for n, s in stress.items():
        print(f"  {n:<24} VaR {k}% ${s[f'VaR_{k}']:>10,.0f}   ES ${s[f'ES_{k}']:>10,.0f}")

    # charts (fan chart uses 252-day horizon)
    report.plot_pnl_distribution(results, a.alpha, a.horizon, out / "pnl_distribution.png")
    _, _, fan_vals, _ = monte_carlo(est, w, a.value, 252, 5000, "student_t", a.seed)
    report.plot_fan(fan_vals, a.value, out / "fan_chart.png")
    report.plot_component_es(main_summary, out / "component_es.png")
    report.plot_stress(stress, out / "stress.png", alpha=k)

    bt_out = {}
    if not a.no_backtest and len(rets) > a.window + 100:
        print(f"\n=== Rolling 1-day {k}% VaR backtest (window={a.window}d) ===")
        bts = {}
        for m in METHODS:
            bt = rolling_var_backtest(rets, w, m, a.alpha, a.window)
            bts[m] = bt; bt_out[m] = evaluate(bt, a.alpha)
            e = bt_out[m]
            print(f"  {m:<10} exceptions {e['exceptions']:>3}/{e['observations']} "
                  f"(expected {e['expected']})  Kupiec p={e['kupiec_p']:.3f}  "
                  f"Christoffersen p={e['christoffersen_p']:.3f}  -> {'PASS' if e['pass_5pct'] else 'FAIL'}")
        report.plot_backtests(bts, a.alpha, out / "backtest.png")

    payload = {"config": vars(a), "t_dof": est.df, "shrinkage": est.shrinkage,
               "risk": {m: r["summary"] for m, r in results.items()},
               "stress": stress, "backtest": bt_out}
    (out / "results.json").write_text(json.dumps(payload, indent=2, default=float))
    print(f"\nSaved charts + results.json to {out.resolve()}")


if __name__ == "__main__":
    main()

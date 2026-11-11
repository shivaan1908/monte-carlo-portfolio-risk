"""Matplotlib charts."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COL = {"gaussian": "#4C78A8", "student_t": "#E45756", "bootstrap": "#54A24B"}


def _money(x, _=None):
    sign = "-" if x < 0 else ""
    x = abs(x)
    return f"{sign}${x/1e3:,.0f}k" if x < 1e6 else f"{sign}${x/1e6:,.2f}M"


def plot_pnl_distribution(results, alpha, horizon, path):
    k = int(round(alpha * 100))
    fig, ax = plt.subplots(figsize=(10, 5.5))
    lo = min(np.quantile(r["pnl"], 0.0005) for r in results.values())
    hi = max(np.quantile(r["pnl"], 0.9995) for r in results.values())
    bins = np.linspace(lo, hi, 140)
    for m, r in results.items():
        ax.hist(r["pnl"], bins=bins, density=True, histtype="step", lw=1.8, color=COL[m], label=m)
        ax.axvline(-r["summary"][f"VaR_{k}"], color=COL[m], ls="--", lw=1.2)
    ax.set(title=f"{horizon}-day P&L distribution  (dashed = {k}% VaR)", xlabel="P&L", ylabel="density")
    ax.xaxis.set_major_formatter(_money); ax.legend(); ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def plot_fan(values, value, path, n_sample=40):
    t = np.arange(1, values.shape[1] + 1)
    q = np.quantile(values, [.05, .25, .5, .75, .95], axis=0)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(t, values[:n_sample].T, color="grey", alpha=.18, lw=.7)
    ax.fill_between(t, q[0], q[4], color="#E45756", alpha=.18, label="5-95%")
    ax.fill_between(t, q[1], q[3], color="#E45756", alpha=.30, label="25-75%")
    ax.plot(t, q[2], color="#E45756", lw=2, label="median")
    ax.axhline(value, color="k", lw=.8, ls=":")
    ax.set(title="Simulated portfolio value (Student-t)", xlabel="trading days", ylabel="value")
    ax.yaxis.set_major_formatter(_money); ax.legend(loc="upper left"); ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def plot_component_es(summary, path):
    comp = summary["component_ES_99"]
    names, vals = list(comp), list(comp.values())
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(names, vals, color="#E45756")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, _money(v), ha="center", va="bottom", fontsize=9)
    ax.set(title="Contribution to 99% Expected Shortfall (sums to total ES)", ylabel="loss in tail")
    ax.yaxis.set_major_formatter(_money); ax.grid(alpha=.25, axis="y")
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def plot_stress(stress, path, alpha=99):
    names = list(stress)
    v = [stress[n][f"VaR_{alpha}"] for n in names]; e = [stress[n][f"ES_{alpha}"] for n in names]
    x = np.arange(len(names)); w = .38
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w/2, v, w, label=f"VaR {alpha}%", color="#4C78A8")
    ax.bar(x + w/2, e, w, label=f"ES {alpha}%", color="#E45756")
    ax.set_xticks(x, names, rotation=12); ax.yaxis.set_major_formatter(_money)
    ax.set(title="Stress scenarios"); ax.legend(); ax.grid(alpha=.25, axis="y")
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def plot_backtests(bts, alpha, path):
    fig, axes = plt.subplots(len(bts), 1, figsize=(11, 3.2 * len(bts)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, (m, bt) in zip(axes, bts.items()):
        ax.plot(bt.index, bt["realised"] * 100, color="grey", lw=.6)
        ax.plot(bt.index, -bt["var"] * 100, color=COL[m], lw=1.4, label=f"-VaR {int(alpha*100)}%")
        h = bt[bt["hit"]]
        ax.scatter(h.index, h["realised"] * 100, color="k", s=18, zorder=3, label=f"exceptions ({len(h)})")
        ax.set(title=m, ylabel="daily return %"); ax.legend(loc="lower left"); ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)

"""
Minimal Deep Hedging — Evaluation Suite

Produces 7 figures covering the core results:

  Figure 1  eval_pnl_distributions.png   P&L histograms: NN vs BSM vs Unhedged
  Figure 2  eval_pnl_analysis.png        Empirical CDF + percentile comparison (2-panel)
  Figure 3  eval_delta_paths.png         Representative delta paths: deep ITM / near ATM / deep OTM vs BSM
  Figure 4  eval_delta_surface.png       Learned delta surface vs BSM N(d1) on a (moneyness x tau) grid
  Figure 5  eval_delta_mae.png           Mean absolute delta error over time, split by moneyness bucket
  Figure 6  eval_error_by_moneyness.png  NN RMSE, BSM RMSE, and mean |delta error| per moneyness bin
  Figure 7  eval_summary_table.png       Summary statistics table (rendered as figure for paper inclusion)
  Console                                Full statistics table

Run from project root:
    python "project/tester/evaluate.py"

Requires a trained model at results/models/minimal_model.pt (run project/minimal/pricer.py first).
"""

import sys
import pathlib

_root = pathlib.Path(__file__).resolve()
while not (_root / "pyproject.toml").exists():
    _root = _root.parent
sys.path.insert(0, str(_root))
sys.path.insert(0, str(_root / "project" / "minimal"))

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm

from project.stock.generators import generate_gbm
from project.helpers.path_helpers import project_path
from pricer import (HedgingNet, run_paths, load_model,
                    bsm_call, bsm_delta, bsm_hedge_pnl,
                    get_nn_deltas, bsm_deltas_path,
                    DEVICE, S0, K, sigma, r, N, T, h)

os.makedirs(project_path("results/tester"), exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid")

# Consistent palette used throughout
C_NN     = "#4C72B0"   # blue  — NN hedge
C_BSM    = "#DD8452"   # orange — BSM hedge
C_UNHD   = "#55A868"   # green  — unhedged

N_PATHS_TEST  = 10_000
ALPHA         = 0.95
N_SURFACE_PTS = 60


# ------------------------------------------------------------------------------
#  Helpers
# ------------------------------------------------------------------------------

def nn_deltas_batch(S_test: np.ndarray, net: HedgingNet) -> np.ndarray:
    """Returns (N_PATHS, N) array of NN deltas for all paths and all time steps."""
    net.eval()
    n_paths = S_test.shape[1]
    out     = np.zeros((n_paths, N))
    with torch.no_grad():
        for t in range(N):
            St  = torch.tensor(S_test[t, :] / K, dtype=torch.float32).to(DEVICE)
            tau = torch.full((n_paths,), 1.0 - t / N, device=DEVICE)
            out[:, t] = net(torch.stack([St, tau], dim=1)).cpu().numpy()
    return out


def bsm_deltas_batch(S_test: np.ndarray) -> np.ndarray:
    """Returns (N_PATHS, N) array of BSM deltas for all paths and all time steps."""
    n_paths = S_test.shape[1]
    out     = np.zeros((n_paths, N))
    for t in range(N):
        out[:, t] = bsm_delta(S_test[t, :], K, r, sigma, t * h, T)
    return out


def var_es(pnl: np.ndarray, alpha: float = ALPHA):
    losses = -pnl
    v      = float(np.quantile(losses, alpha))
    e      = float(losses[losses >= v].mean())
    return v, e


def rmse(pnl: np.ndarray) -> float:
    return float(np.sqrt(np.mean(pnl ** 2)))


def variance_reduction(hedged: np.ndarray, unhedged: np.ndarray) -> float:
    return (1 - np.var(hedged) / np.var(unhedged)) * 100


# ------------------------------------------------------------------------------
#  Console output
# ------------------------------------------------------------------------------

def print_results(pnl_nn, pnl_bsm, pnl_unhedged, net):
    bsm_price  = bsm_call(S0, K, r, sigma, T)
    nn_premium = net.premium.item()
    pcts       = [1, 5, 25, 50, 75, 95, 99]
    var_nn,  es_nn  = var_es(pnl_nn)
    var_bsm, es_bsm = var_es(pnl_bsm)
    var_u,   es_u   = var_es(pnl_unhedged)

    w = 72
    print("\n" + "=" * w)
    print("  DEEP HEDGING — EVALUATION RESULTS")
    print("=" * w)
    print(f"\n  Pricing")
    print(f"  {'-'*40}")
    print(f"  {'BSM theoretical price':35s}  {bsm_price:.6f}")
    print(f"  {'Learned premium':35s}  {nn_premium:.6f}  ({(nn_premium/bsm_price - 1)*100:+.3f}%)")

    hdr = f"\n  {'Metric':<22} {'NN Hedge':>12} {'BSM Hedge':>12} {'Unhedged':>12}"
    print(hdr)
    print(f"  {'-'*58}")

    def row(label, a, b, c):
        print(f"  {label:<22} {a:>12.6f} {b:>12.6f} {c:>12.6f}")

    row("Mean P&L",        pnl_nn.mean(),  pnl_bsm.mean(),  pnl_unhedged.mean())
    row("Std P&L",         pnl_nn.std(),   pnl_bsm.std(),   pnl_unhedged.std())
    row("RMSE",            rmse(pnl_nn),   rmse(pnl_bsm),   rmse(pnl_unhedged))
    row("Skewness",        stats.skew(pnl_nn),     stats.skew(pnl_bsm),     stats.skew(pnl_unhedged))
    row("Excess kurtosis", stats.kurtosis(pnl_nn), stats.kurtosis(pnl_bsm), stats.kurtosis(pnl_unhedged))
    row(f"VaR {ALPHA:.0%}",    var_nn,  var_bsm,  var_u)
    row(f"ES  {ALPHA:.0%}",    es_nn,   es_bsm,   es_u)
    print(f"  {'-'*58}")
    for p in pcts:
        row(f"P{p}", np.percentile(pnl_nn, p), np.percentile(pnl_bsm, p), np.percentile(pnl_unhedged, p))

    print(f"\n  Variance reduction vs unhedged")
    print(f"  {'-'*40}")
    print(f"  {'NN Hedge':35s}  {variance_reduction(pnl_nn,  pnl_unhedged):+.2f}%")
    print(f"  {'BSM Hedge':35s}  {variance_reduction(pnl_bsm, pnl_unhedged):+.2f}%")
    print("\n" + "=" * w + "\n")


# ------------------------------------------------------------------------------
#  Figure 1: P&L Distributions
# ------------------------------------------------------------------------------

def fig_pnl_distributions(pnl_nn, pnl_bsm, pnl_unhedged):
    """
    Three-panel histogram comparing terminal P&L distributions.
    NN and BSM panels share an x-axis so spread is directly comparable.
    Unhedged is shown on its own scale to illustrate the magnitude of raw risk.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Terminal P&L Distributions", fontsize=13)

    hedged_lo = min(pnl_nn.min(), pnl_bsm.min())
    hedged_hi = max(pnl_nn.max(), pnl_bsm.max())
    pad       = (hedged_hi - hedged_lo) * 0.05

    for ax, pnl, label, color in zip(
        axes,
        [pnl_nn, pnl_bsm, pnl_unhedged],
        ["NN Hedge", "BSM Hedge", "Unhedged"],
        [C_NN, C_BSM, C_UNHD],
    ):
        ax.hist(pnl, bins=60, color=color, alpha=0.85, edgecolor="none", density=True)
        ax.axvline(0,          color="black", linewidth=0.9, linestyle=":",
                   label="Zero")
        ax.axvline(pnl.mean(), color="red",   linewidth=1.1, linestyle="--",
                   label=f"Mean = {pnl.mean():.4f}")
        ax.set_xlabel("Terminal P&L")
        ax.set_ylabel("Density")
        ax.set_title(label)
        ax.legend(fontsize=8)

    axes[0].set_xlim(hedged_lo - pad, hedged_hi + pad)
    axes[1].set_xlim(hedged_lo - pad, hedged_hi + pad)

    plt.tight_layout()
    plt.savefig(project_path("results/tester/eval_pnl_distributions.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print("Figure 1 saved: eval_pnl_distributions.png")


# ------------------------------------------------------------------------------
#  Figure 2: P&L Analysis — CDF + Percentile comparison
# ------------------------------------------------------------------------------

def fig_pnl_analysis(pnl_nn, pnl_bsm):
    """
    Two-panel figure.
    Left:  Empirical CDFs for NN and BSM, with 5th-percentile reference line.
           The horizontal gap between the two curves at any probability level
           directly shows the additional tail risk of the NN hedge.
    Right: Side-by-side percentile bar chart at key quantiles.
           Makes the left-tail shortfall and right-tail profile concrete.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle("P&L Analysis: Deep Hedging vs BSM", fontsize=13)

    # --- Empirical CDFs ---
    ax = axes[0]
    for pnl, label, color in [(pnl_nn, "NN Hedge", C_NN), (pnl_bsm, "BSM Hedge", C_BSM)]:
        sorted_pnl = np.sort(pnl)
        cdf        = np.arange(1, len(sorted_pnl) + 1) / len(sorted_pnl)
        ax.plot(sorted_pnl, cdf, label=label, color=color, linewidth=1.5)
    ax.axvline(0,    color="black", linewidth=0.8, linestyle=":")
    ax.axhline(0.05, color="grey", linewidth=0.7, linestyle="--", label="5th percentile")
    ax.set_xlabel("Terminal P&L")
    ax.set_ylabel("Cumulative probability")
    ax.set_title("Empirical CDF")
    ax.legend(fontsize=9)

    # --- Percentile bar chart ---
    ax   = axes[1]
    pcts = [1, 5, 10, 25, 75, 90, 95, 99]
    nn_vals  = [np.percentile(pnl_nn,  p) for p in pcts]
    bsm_vals = [np.percentile(pnl_bsm, p) for p in pcts]
    x     = np.arange(len(pcts))
    width = 0.35
    ax.bar(x - width/2, nn_vals,  width, label="NN Hedge",  color=C_NN,  alpha=0.85)
    ax.bar(x + width/2, bsm_vals, width, label="BSM Hedge", color=C_BSM, alpha=0.85)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels([f"P{p}" for p in pcts], fontsize=8)
    ax.set_ylabel("P&L")
    ax.set_title("Percentile Comparison")
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(project_path("results/tester/eval_pnl_analysis.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print("Figure 2 saved: eval_pnl_analysis.png")


# ------------------------------------------------------------------------------
#  Figure 3: Representative Delta Paths
# ------------------------------------------------------------------------------

def fig_delta_paths(S_test, net):
    """
    Three representative paths (deep ITM, near ATM, deep OTM) showing how the
    NN delta tracks the BSM delta over the life of the option.
    The shaded region between them is the instantaneous delta error.
    """
    S_T     = S_test[N, :]
    times   = np.linspace(0, T, N)

    itm_mask = S_T > K * 1.08
    atm_mask = np.abs(S_T / K - 1) < 0.02
    otm_mask = S_T < K * 0.92

    def pick(mask, selector):
        cands = np.where(mask)[0]
        return int(cands[selector(S_T[cands])]) if len(cands) else None

    itm_idx = pick(itm_mask, np.argmax)
    atm_idx = pick(atm_mask, lambda v: np.argmin(np.abs(v / K - 1)))
    otm_idx = pick(otm_mask, np.argmin)

    cases = [
        (itm_idx, f"Deep ITM  (S_T = {S_T[itm_idx]:.3f})" if itm_idx is not None else "Deep ITM"),
        (atm_idx, f"Near ATM  (S_T = {S_T[atm_idx]:.3f})" if atm_idx is not None else "Near ATM"),
        (otm_idx, f"Deep OTM  (S_T = {S_T[otm_idx]:.3f})" if otm_idx is not None else "Deep OTM"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)
    fig.suptitle("Delta Hedge Paths: NN vs BSM", fontsize=13)

    for ax, (idx, title) in zip(axes, cases):
        if idx is None:
            ax.text(0.5, 0.5, "No matching path found", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(title)
            continue
        path  = S_test[:, idx]
        nn_d  = get_nn_deltas(path, net)
        bsm_d = bsm_deltas_path(path)
        ax.plot(times, nn_d,  color=C_NN,  linewidth=1.4, label="NN delta")
        ax.plot(times, bsm_d, color=C_BSM, linewidth=1.4, linestyle="--", label="BSM delta")
        ax.fill_between(times, nn_d, bsm_d, alpha=0.15, color="grey", label="Delta error")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Time")
        ax.set_ylabel("Delta (shares held)")
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(project_path("results/tester/eval_delta_paths.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print("Figure 3 saved: eval_delta_paths.png")


# ------------------------------------------------------------------------------
#  Figure 4: Learned Delta Surface
# ------------------------------------------------------------------------------

def fig_delta_surface(net):
    """
    Heatmap of the NN delta as a function of moneyness S/K and time-to-maturity tau,
    alongside the analytic BSM N(d1) surface and the pointwise error.
    Demonstrates that the network has recovered the BSM delta function from data alone.
    Largest errors appear near ATM at short maturities, where the delta surface is
    steepest (high gamma) and therefore hardest to learn.
    """
    m_vals   = np.linspace(0.70, 1.30, N_SURFACE_PTS)
    tau_vals = np.linspace(0.02, 1.00, N_SURFACE_PTS)
    M, TAU   = np.meshgrid(m_vals, tau_vals)

    net.eval()
    grid_in = torch.tensor(
        np.stack([M.ravel(), TAU.ravel()], axis=1), dtype=torch.float32
    ).to(DEVICE)
    with torch.no_grad():
        nn_surf = net(grid_in).cpu().numpy().reshape(N_SURFACE_PTS, N_SURFACE_PTS)

    def bsm_delta_surface(m, tau):
        tau = np.maximum(tau, 1e-6)
        d1  = (np.log(m) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
        return norm.cdf(d1)

    bsm_surf = bsm_delta_surface(M, TAU)
    err_surf = nn_surf - bsm_surf
    err_lim  = np.abs(err_surf).max()

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Learned Delta Surface vs BSM N(d1)", fontsize=13)

    panels = [
        (axes[0], nn_surf,   "NN Delta  d(S/K, tau)",  "RdYlGn", 0.0,      1.0),
        (axes[1], bsm_surf,  "BSM Delta  N(d1)",        "RdYlGn", 0.0,      1.0),
        (axes[2], err_surf,  "Error  NN - BSM",          "RdBu_r", -err_lim, err_lim),
    ]
    for ax, data, title, cmap, vmin, vmax in panels:
        im = ax.pcolormesh(m_vals, tau_vals, data, cmap=cmap,
                           vmin=vmin, vmax=vmax, shading="auto")
        ax.contour(m_vals, tau_vals, data, levels=10,
                   colors="white", linewidths=0.5, alpha=0.4)
        ax.axvline(1.0, color="white", linewidth=0.8, linestyle="--", alpha=0.7)
        ax.set_xlabel("Moneyness  S/K")
        ax.set_ylabel("Time to maturity  tau")
        ax.set_title(title)
        plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig(project_path("results/tester/eval_delta_surface.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print("Figure 4 saved: eval_delta_surface.png")


# ------------------------------------------------------------------------------
#  Figure 5: Mean Absolute Delta Error Over Time
# ------------------------------------------------------------------------------

def fig_delta_mae(S_test, net):
    """
    Single-panel figure showing how the mean absolute delta error |NN - BSM|
    evolves over the option's life, broken down by terminal moneyness bucket.

    Key feature: all buckets show an error spike near expiry.  This is expected —
    the BSM delta becomes a step function as tau -> 0 (high gamma near ATM), making
    perfect tracking impossible at discrete time steps.  The spike is sharpest for
    ATM-ITM paths, consistent with the gamma surface.
    """
    nn_d      = nn_deltas_batch(S_test, net)
    bsm_d     = bsm_deltas_batch(S_test)
    delta_err = nn_d - bsm_d
    times     = np.linspace(0, T, N)
    S_T       = S_test[N, :]

    buckets = [
        (S_T < 0.95,                     "OTM  (S_T < 0.95)",   "#E74C3C"),
        ((S_T >= 0.95) & (S_T < 1.00),  "OTM-ATM (0.95-1.00)", "#E67E22"),
        ((S_T >= 1.00) & (S_T < 1.05),  "ATM-ITM (1.00-1.05)", "#27AE60"),
        (S_T >= 1.05,                    "ITM  (S_T > 1.05)",   "#2980B9"),
    ]

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.suptitle("Mean Absolute Delta Error Over Time", fontsize=13)

    for mask, label, color in buckets:
        if mask.sum() > 10:
            mae = np.abs(delta_err[mask]).mean(axis=0)
            ax.plot(times, mae, label=f"{label}  (n={mask.sum()})",
                    color=color, linewidth=1.4)

    ax.set_xlabel("Time")
    ax.set_ylabel("|NN delta - BSM delta|")
    ax.set_title("Broken down by terminal moneyness bucket")
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(project_path("results/tester/eval_delta_mae.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print("Figure 5 saved: eval_delta_mae.png")

    return nn_d, bsm_d   # return so Figure 6 can reuse them


# ------------------------------------------------------------------------------
#  Figure 6: Hedging Error Decomposed by Moneyness Bin
# ------------------------------------------------------------------------------

def fig_error_by_moneyness(pnl_nn, pnl_bsm, S_test, nn_d, bsm_d):
    """
    Grouped bar chart with twin y-axes.
    Left axis:  NN RMSE and BSM RMSE of terminal P&L per moneyness bin.
    Right axis: Mean absolute delta error (averaged over all time steps on those paths).

    Shows that both P&L RMSE and delta error are largest near ATM (0.97-1.03),
    where gamma is highest and discrete rebalancing is most costly.
    The NN consistently underperforms BSM on P&L RMSE, but the gap widens at ATM.
    """
    S_T    = S_test[N, :]
    edges  = [0.00, 0.90, 0.97, 1.03, 1.10, np.inf]
    labels = ["<0.90", "0.90-0.97", "0.97-1.03", "1.03-1.10", ">1.10"]

    nn_rmse_vals, bsm_rmse_vals, mae_delta_vals, n_vals = [], [], [], []

    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (S_T >= lo * K) & (S_T < hi * K)
        n_vals.append(mask.sum())
        if mask.sum() < 10:
            nn_rmse_vals.append(np.nan)
            bsm_rmse_vals.append(np.nan)
            mae_delta_vals.append(np.nan)
        else:
            nn_rmse_vals.append(rmse(pnl_nn[mask]))
            bsm_rmse_vals.append(rmse(pnl_bsm[mask]))
            mae_delta_vals.append(np.abs(nn_d[mask] - bsm_d[mask]).mean())

    x     = np.arange(len(labels))
    width = 0.25

    fig, ax1 = plt.subplots(figsize=(11, 5))
    fig.suptitle("Hedging Error Decomposition by Terminal Moneyness", fontsize=13)

    b1 = ax1.bar(x - width,  nn_rmse_vals,  width, label="NN RMSE",  color=C_NN,    alpha=0.85)
    b2 = ax1.bar(x,          bsm_rmse_vals, width, label="BSM RMSE", color=C_BSM,   alpha=0.85)
    ax1.set_ylabel("RMSE of Terminal P&L")
    ax1.set_xticks(x)
    ax1.set_xticklabels(
        [f"{l}\n(n={n})" for l, n in zip(labels, n_vals)], fontsize=9
    )
    ax1.set_xlabel("Terminal moneyness bin  S_T / K")

    ax2 = ax1.twinx()
    b3  = ax2.bar(x + width, mae_delta_vals, width, label="Mean |delta error|",
                  color=C_UNHD, alpha=0.85)
    ax2.set_ylabel("Mean |NN delta - BSM delta|")

    ax1.legend([b1, b2, b3], ["NN RMSE", "BSM RMSE", "Mean |delta error|"],
               fontsize=9, loc="upper right")

    plt.tight_layout()
    plt.savefig(project_path("results/tester/eval_error_by_moneyness.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print("Figure 6 saved: eval_error_by_moneyness.png")


# ------------------------------------------------------------------------------
#  Figure 7: Summary Statistics Table
# ------------------------------------------------------------------------------

def fig_summary_table(pnl_nn, pnl_bsm, pnl_unhedged, net):
    """
    Renders the key summary statistics as a publication-ready table figure.
    Intended for direct inclusion in the paper alongside (or instead of) a
    LaTeX table, since the formatting is consistent with the other figures.
    """
    bsm_price  = bsm_call(S0, K, r, sigma, T)
    nn_premium = net.premium.item()
    var_nn,  es_nn  = var_es(pnl_nn)
    var_bsm, es_bsm = var_es(pnl_bsm)
    var_u,   es_u   = var_es(pnl_unhedged)

    col_labels = ["Metric", "NN Hedge", "BSM Hedge", "Unhedged"]
    rows = [
        ["Mean P&L",
         f"{pnl_nn.mean():.6f}", f"{pnl_bsm.mean():.6f}", f"{pnl_unhedged.mean():.6f}"],
        ["Std P&L",
         f"{pnl_nn.std():.6f}", f"{pnl_bsm.std():.6f}", f"{pnl_unhedged.std():.6f}"],
        ["RMSE",
         f"{rmse(pnl_nn):.6f}", f"{rmse(pnl_bsm):.6f}", f"{rmse(pnl_unhedged):.6f}"],
        ["Skewness",
         f"{stats.skew(pnl_nn):.4f}", f"{stats.skew(pnl_bsm):.4f}",
         f"{stats.skew(pnl_unhedged):.4f}"],
        ["Excess kurtosis",
         f"{stats.kurtosis(pnl_nn):.4f}", f"{stats.kurtosis(pnl_bsm):.4f}",
         f"{stats.kurtosis(pnl_unhedged):.4f}"],
        [f"VaR {ALPHA:.0%}",
         f"{var_nn:.6f}", f"{var_bsm:.6f}", f"{var_u:.6f}"],
        [f"ES {ALPHA:.0%}",
         f"{es_nn:.6f}", f"{es_bsm:.6f}", f"{es_u:.6f}"],
        ["P1",
         f"{np.percentile(pnl_nn,1):.6f}", f"{np.percentile(pnl_bsm,1):.6f}",
         f"{np.percentile(pnl_unhedged,1):.6f}"],
        ["P5",
         f"{np.percentile(pnl_nn,5):.6f}", f"{np.percentile(pnl_bsm,5):.6f}",
         f"{np.percentile(pnl_unhedged,5):.6f}"],
        ["P25",
         f"{np.percentile(pnl_nn,25):.6f}", f"{np.percentile(pnl_bsm,25):.6f}",
         f"{np.percentile(pnl_unhedged,25):.6f}"],
        ["P50",
         f"{np.percentile(pnl_nn,50):.6f}", f"{np.percentile(pnl_bsm,50):.6f}",
         f"{np.percentile(pnl_unhedged,50):.6f}"],
        ["P75",
         f"{np.percentile(pnl_nn,75):.6f}", f"{np.percentile(pnl_bsm,75):.6f}",
         f"{np.percentile(pnl_unhedged,75):.6f}"],
        ["P95",
         f"{np.percentile(pnl_nn,95):.6f}", f"{np.percentile(pnl_bsm,95):.6f}",
         f"{np.percentile(pnl_unhedged,95):.6f}"],
        ["P99",
         f"{np.percentile(pnl_nn,99):.6f}", f"{np.percentile(pnl_bsm,99):.6f}",
         f"{np.percentile(pnl_unhedged,99):.6f}"],
        ["Var. reduction",
         f"{variance_reduction(pnl_nn,  pnl_unhedged):+.2f}%",
         f"{variance_reduction(pnl_bsm, pnl_unhedged):+.2f}%",
         "—"],
    ]

    # Pricing header rows rendered separately above the main table.
    # cell_text = [r[1:] for r in all_rows], so r[1] → NN Hedge column.
    # The relative error is folded into the NN Hedge cell so it reads naturally.
    pricing_rows = [
        ["BSM price C0",    f"{bsm_price:.6f}", "", ""],
        ["Learned premium",
         f"{nn_premium:.6f}  ({(nn_premium/bsm_price-1)*100:+.3f}%)", "", ""],
    ]

    n_rows = len(pricing_rows) + 1 + len(rows)   # +1 for spacer
    fig_h  = 0.38 * n_rows + 1.2
    fig, ax = plt.subplots(figsize=(11, fig_h))
    ax.axis("off")
    fig.suptitle("Summary Statistics", fontsize=13, y=0.98)

    # Build one flat table: pricing block + divider + stats block
    all_rows   = pricing_rows + [[""] * 4] + rows
    cell_text  = [r[1:] for r in all_rows]
    row_labels = [r[0]  for r in all_rows]

    tbl = ax.table(
        cellText=cell_text,
        rowLabels=row_labels,
        colLabels=col_labels[1:],
        cellLoc="center",
        rowLoc="right",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.4)

    # Header row styling
    for j in range(3):
        tbl[0, j].set_facecolor("#2C3E50")
        tbl[0, j].set_text_props(color="white", fontweight="bold")

    # Pricing rows: light blue
    for i in range(1, len(pricing_rows) + 1):
        for j in range(3):
            tbl[i, j].set_facecolor("#D6EAF8")

    # Spacer row
    spacer_i = len(pricing_rows) + 1
    for j in range(3):
        tbl[spacer_i, j].set_facecolor("#F0F0F0")

    # Alternating row shading for stats block
    for i in range(len(pricing_rows) + 2, len(all_rows) + 1):
        shade = "#FDFEFE" if (i % 2 == 0) else "#EBF5FB"
        for j in range(3):
            tbl[i, j].set_facecolor(shade)

    plt.tight_layout()
    plt.savefig(project_path("results/tester/eval_summary_table.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print("Figure 7 saved: eval_summary_table.png")


# ------------------------------------------------------------------------------
#  Main
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    print("-- Loading trained model --")
    net = load_model()
    net.eval()

    print("-- Simulating test paths --")
    S_test = generate_gbm(S0, r, sigma, h, N_PATHS_TEST, N + 1)

    print("-- Computing P&L --")
    S_tensor = torch.tensor(S_test.T, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        pnl_nn = run_paths(S_tensor, net).cpu().numpy()

    pnl_bsm      = bsm_hedge_pnl(S_test)
    payoffs       = np.maximum(S_test[N, :] - K, 0)
    pnl_unhedged  = net.premium.item() - payoffs

    print_results(pnl_nn, pnl_bsm, pnl_unhedged, net)

    print("-- Generating figures --")
    fig_pnl_distributions(pnl_nn, pnl_bsm, pnl_unhedged)   # Figure 1
    fig_pnl_analysis(pnl_nn, pnl_bsm)                       # Figure 2
    fig_delta_paths(S_test, net)                             # Figure 3
    fig_delta_surface(net)                                   # Figure 4
    nn_d, bsm_d = fig_delta_mae(S_test, net)                # Figure 5 — returns deltas
    fig_error_by_moneyness(pnl_nn, pnl_bsm, S_test,
                           nn_d, bsm_d)                      # Figure 6 — reuses deltas
    fig_summary_table(pnl_nn, pnl_bsm, pnl_unhedged, net)  # Figure 7

    print(f"\nAll outputs saved to results/tester/")
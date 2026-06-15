"""
Minimal Deep Hedging — Comprehensive Evaluation Suite

Loads the trained minimal model and produces a full diagnostic output:

  Figure 1  eval_pnl.png            P&L analysis: distributions, CDF, percentiles, moneyness breakdown
  Figure 2  eval_delta_paths.png    Representative delta paths: deep ITM / near ATM / deep OTM vs BSM
  Figure 3  eval_delta_surface.png  Learned delta surface vs BSM N(d1) on a (moneyness x tau) grid
  Figure 4  eval_statistics.png     Statistical diagnostics: QQ plot, delta scatter, error over time
  Console                           Full statistics table (moments, percentiles, VaR, ES, variance reduction)

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
import matplotlib.gridspec as gridspec
from scipy import stats
from scipy.stats import norm

from project.stock.generators import generate_gbm
from project.helpers.path_helpers import project_path
from pricer import (HedgingNet, run_paths, load_model,
                    bsm_call, bsm_delta, bsm_hedge_pnl,
                    get_nn_deltas, bsm_deltas_path,
                    DEVICE, S0, K, sigma, r, N, T, h)

os.makedirs(project_path("results/tester"), exist_ok=True)

N_PATHS_TEST  = 10_000
ALPHA         = 0.95          # for VaR / ES reporting
N_SURFACE_PTS = 60            # grid resolution for delta surface


# -- Vectorised delta helpers ---------------------------------------------------

def nn_deltas_batch(S_test: np.ndarray, net: HedgingNet) -> np.ndarray:
    """Returns (N_PATHS, N) array of NN deltas for all paths and all timesteps."""
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
    """Returns (N_PATHS, N) array of BSM deltas for all paths and all timesteps."""
    n_paths = S_test.shape[1]
    out     = np.zeros((n_paths, N))
    for t in range(N):
        out[:, t] = bsm_delta(S_test[t, :], K, r, sigma, t * h, T)
    return out


# -- Tail risk metrics ---------------------------------------------------------

def var95(pnl: np.ndarray) -> float:
    return float(np.quantile(-pnl, ALPHA))


def es95(pnl: np.ndarray) -> float:
    losses = -pnl
    return float(losses[losses >= np.quantile(losses, ALPHA)].mean())


def variance_reduction(pnl_hedged: np.ndarray, pnl_unhedged: np.ndarray) -> float:
    return (1 - np.var(pnl_hedged) / np.var(pnl_unhedged)) * 100


# -- Console output -------------------------------------------------------------

def print_results(pnl_nn: np.ndarray, pnl_bsm: np.ndarray,
                  pnl_unhedged: np.ndarray, net: HedgingNet) -> None:
    bsm_price  = bsm_call(S0, K, r, sigma, T)
    nn_premium = net.premium.item()
    pcts       = [1, 5, 25, 50, 75, 95, 99]

    w = 72
    print("\n" + "=" * w)
    print("  DEEP HEDGING - EVALUATION RESULTS")
    print("=" * w)

    print(f"\n  Pricing")
    print(f"  {'-'*40}")
    print(f"  {'BSM theoretical price':35s}  {bsm_price:.6f}")
    print(f"  {'Learned premium':35s}  {nn_premium:.6f}  ({(nn_premium/bsm_price - 1)*100:+.2f}%)")

    hdr = f"\n  {'Metric':<22} {'NN Hedge':>12} {'BSM Hedge':>12} {'Unhedged':>12}"
    print(hdr)
    print(f"  {'-'*58}")

    def row(label, nn, bsm, unhed):
        print(f"  {label:<22} {nn:>12.6f} {bsm:>12.6f} {unhed:>12.6f}")

    row("Mean P&L",      pnl_nn.mean(),  pnl_bsm.mean(),  pnl_unhedged.mean())
    row("Std P&L",       pnl_nn.std(),   pnl_bsm.std(),   pnl_unhedged.std())
    row("Skewness",      stats.skew(pnl_nn),  stats.skew(pnl_bsm),  stats.skew(pnl_unhedged))
    row("Excess kurtosis", stats.kurtosis(pnl_nn), stats.kurtosis(pnl_bsm), stats.kurtosis(pnl_unhedged))
    row(f"VaR {ALPHA:.0%}",   var95(pnl_nn),  var95(pnl_bsm),  var95(pnl_unhedged))
    row(f"ES  {ALPHA:.0%}",   es95(pnl_nn),   es95(pnl_bsm),   es95(pnl_unhedged))

    print(f"  {'-'*58}")
    for p in pcts:
        row(f"P{p}",
            np.percentile(pnl_nn,       p),
            np.percentile(pnl_bsm,      p),
            np.percentile(pnl_unhedged, p))

    print(f"\n  Variance reduction vs unhedged")
    print(f"  {'-'*40}")
    print(f"  {'NN Hedge':35s}  {variance_reduction(pnl_nn,  pnl_unhedged):+.2f}%")
    print(f"  {'BSM Hedge':35s}  {variance_reduction(pnl_bsm, pnl_unhedged):+.2f}%")
    print("\n" + "=" * w + "\n")


# -- Figure 1a: P&L Distributions ---------------------------------------------

def fig_pnl_distributions(pnl_nn: np.ndarray, pnl_bsm: np.ndarray,
                           pnl_unhedged: np.ndarray) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Terminal P&L Distributions", fontsize=13)

    strategies = [
        (pnl_nn,       "NN Hedge",  "#4C72B0"),
        (pnl_bsm,      "BSM Hedge", "#DD8452"),
        (pnl_unhedged, "Unhedged",  "#55A868"),
    ]

    hedged_lo = min(pnl_nn.min(), pnl_bsm.min())
    hedged_hi = max(pnl_nn.max(), pnl_bsm.max())
    pad       = (hedged_hi - hedged_lo) * 0.05

    for ax, (pnl, label, color) in zip(axes, strategies):
        ax.hist(pnl, bins=60, color=color, alpha=0.85, edgecolor="none", density=True)
        ax.axvline(0,          color="black", linewidth=0.9, linestyle=":",  label="Zero")
        ax.axvline(pnl.mean(), color="red",   linewidth=1.1, linestyle="--",
                   label=f"Mean = {pnl.mean():.4f}")
        ax.set_xlabel("Terminal P&L")
        ax.set_ylabel("Density")
        ax.set_title(label)
        ax.legend(fontsize=8)

    # NN and BSM share x-axis so spread is directly comparable
    axes[0].set_xlim(hedged_lo - pad, hedged_hi + pad)
    axes[1].set_xlim(hedged_lo - pad, hedged_hi + pad)

    plt.tight_layout()
    plt.savefig(project_path("results/tester/eval_pnl_distributions.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Figure 1a saved: eval_pnl_distributions.png")


# -- Figure 1b: P&L Analysis --------------------------------------------------

def fig_pnl_analysis(pnl_nn: np.ndarray, pnl_bsm: np.ndarray,
                     S_test: np.ndarray) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("P&L Analysis: Deep Hedging vs BSM", fontsize=13)

    # -- (0) Empirical CDFs --
    ax = axes[0]
    for pnl, label, color in [
        (pnl_nn,  "NN Hedge",  "#4C72B0"),
        (pnl_bsm, "BSM Hedge", "#DD8452"),
    ]:
        sorted_pnl = np.sort(pnl)
        cdf        = np.arange(1, len(sorted_pnl) + 1) / len(sorted_pnl)
        ax.plot(sorted_pnl, cdf, label=label, color=color, linewidth=1.3)
    ax.axvline(0, color="black", linewidth=0.8, linestyle=":")
    ax.axhline(0.05, color="grey", linewidth=0.7, linestyle="--", label="5th pct")
    ax.set_xlabel("P&L")
    ax.set_ylabel("Cumulative probability")
    ax.set_title("Empirical CDF")
    ax.legend(fontsize=9)

    # -- (1) Percentile comparison --
    ax   = axes[1]
    pcts = [1, 5, 10, 25, 75, 90, 95, 99]
    nn_vals  = [np.percentile(pnl_nn,  p) for p in pcts]
    bsm_vals = [np.percentile(pnl_bsm, p) for p in pcts]
    x     = np.arange(len(pcts))
    width = 0.35
    ax.bar(x - width/2, nn_vals,  width, label="NN Hedge",  color="#4C72B0")
    ax.bar(x + width/2, bsm_vals, width, label="BSM Hedge", color="#DD8452")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels([f"P{p}" for p in pcts], fontsize=8)
    ax.set_ylabel("P&L")
    ax.set_title("Percentile Comparison")
    ax.legend(fontsize=9)

    # -- (2) P&L by terminal moneyness bucket --
    ax       = axes[2]
    S_T      = S_test[N, :]
    edges    = [0.85, 0.93, 0.97, 1.00, 1.03, 1.07, 1.20]
    labels_m = ["<0.93", "0.93-0.97", "0.97-1.00", "1.00-1.03", "1.03-1.07", ">1.07"]
    data_nn  = []
    data_bsm = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (S_T >= lo) & (S_T < hi)
        if mask.sum() > 10:
            data_nn.append(pnl_nn[mask])
            data_bsm.append(pnl_bsm[mask])
        else:
            data_nn.append(np.array([np.nan]))
            data_bsm.append(np.array([np.nan]))

    pos_nn  = np.arange(len(labels_m)) * 2 - 0.4
    pos_bsm = np.arange(len(labels_m)) * 2 + 0.4
    bp1 = ax.boxplot(data_nn,  positions=pos_nn,  widths=0.6,
                     patch_artist=True, boxprops=dict(facecolor="#4C72B0", alpha=0.6),
                     medianprops=dict(color="white"), showfliers=False)
    bp2 = ax.boxplot(data_bsm, positions=pos_bsm, widths=0.6,
                     patch_artist=True, boxprops=dict(facecolor="#DD8452", alpha=0.6),
                     medianprops=dict(color="white"), showfliers=False)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(np.arange(len(labels_m)) * 2)
    ax.set_xticklabels(labels_m, fontsize=7, rotation=15)
    ax.set_xlabel("Terminal moneyness S_T / K")
    ax.set_ylabel("P&L")
    ax.set_title("P&L by Terminal Moneyness")
    ax.legend([bp1["boxes"][0], bp2["boxes"][0]], ["NN Hedge", "BSM Hedge"], fontsize=9)

    plt.tight_layout()
    plt.savefig(project_path("results/tester/eval_pnl_analysis.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Figure 1b saved: eval_pnl_analysis.png")


# -- Figure 2: Representative Delta Paths --------------------------------------

def fig_delta_paths(S_test: np.ndarray, net: HedgingNet) -> None:
    S_T = S_test[N, :]

    # Pick deep ITM, near ATM, deep OTM
    itm_mask = S_T > K * 1.08
    atm_mask = np.abs(S_T - K) < 0.02
    otm_mask = S_T < K * 0.92

    def best(mask, selector):
        cands = np.where(mask)[0]
        return cands[selector(S_T[cands])] if len(cands) else None

    itm_idx = best(itm_mask, np.argmax)
    atm_idx = best(atm_mask, lambda v: np.argmin(np.abs(v - K)))
    otm_idx = best(otm_mask, np.argmin)

    times  = np.linspace(0, T, N)
    labels = [
        (itm_idx, f"Deep ITM  (S_T = {S_T[itm_idx]:.3f})" if itm_idx is not None else "Deep ITM"),
        (atm_idx, f"Near ATM  (S_T = {S_T[atm_idx]:.3f})" if atm_idx is not None else "Near ATM"),
        (otm_idx, f"Deep OTM  (S_T = {S_T[otm_idx]:.3f})" if otm_idx is not None else "Deep OTM"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)
    fig.suptitle("Delta Hedge Paths: NN vs BSM", fontsize=13)

    for ax, (idx, title) in zip(axes, labels):
        if idx is None:
            ax.text(0.5, 0.5, "No path found", ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_title(title)
            continue
        path      = S_test[:, idx]
        nn_d      = get_nn_deltas(path, net)
        bsm_d     = bsm_deltas_path(path)
        ax.plot(times, nn_d,  color="#4C72B0", linewidth=1.3, label="NN delta")
        ax.plot(times, bsm_d, color="#DD8452", linewidth=1.3, linestyle="--", label="BSM delta")
        ax.fill_between(times, nn_d, bsm_d, alpha=0.12, color="grey", label="Delta error")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Time")
        ax.set_ylabel("Delta (shares held)")
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(project_path("results/tester/eval_delta_paths.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Figure 2 saved: eval_delta_paths.png")


# -- Figure 3: Learned Delta Surface ------------------------------------------

def fig_delta_surface(net: HedgingNet) -> None:
    m_vals   = np.linspace(0.70, 1.30, N_SURFACE_PTS)   # moneyness S/K
    tau_vals = np.linspace(0.02, 1.00, N_SURFACE_PTS)   # time to maturity
    M, TAU   = np.meshgrid(m_vals, tau_vals)             # (N_PTS, N_PTS)

    # NN surface
    net.eval()
    grid_input = torch.tensor(
        np.stack([M.ravel(), TAU.ravel()], axis=1), dtype=torch.float32
    ).to(DEVICE)
    with torch.no_grad():
        nn_surface = net(grid_input).cpu().numpy().reshape(N_SURFACE_PTS, N_SURFACE_PTS)

    # BSM N(d1) surface
    def bsm_d1_surface(m, tau):
        tau = np.maximum(tau, 1e-6)
        d1  = (np.log(m) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
        return norm.cdf(d1)

    bsm_surface  = bsm_d1_surface(M, TAU)
    error_surface = nn_surface - bsm_surface

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Learned Delta Surface vs BSM N(d₁)", fontsize=13)

    vmin, vmax = 0.0, 1.0
    cmap_main  = "RdYlGn"
    cmap_err   = "RdBu"

    for ax, data, title, cm, v0, v1 in [
        (axes[0], nn_surface,    "NN Delta  δ(S/K, τ)",   cmap_main, vmin, vmax),
        (axes[1], bsm_surface,   "BSM Delta  N(d₁)",      cmap_main, vmin, vmax),
        (axes[2], error_surface, "Error  NN − BSM",        cmap_err,
         -np.abs(error_surface).max(), np.abs(error_surface).max()),
    ]:
        im = ax.pcolormesh(m_vals, tau_vals, data, cmap=cm, vmin=v0, vmax=v1,
                           shading="auto")
        ax.contour(m_vals, tau_vals, data, levels=10, colors="white",
                   linewidths=0.5, alpha=0.4)
        ax.axvline(1.0, color="white", linewidth=0.8, linestyle="--", alpha=0.7)
        ax.set_xlabel("Moneyness  S/K")
        ax.set_ylabel("Time to maturity  τ")
        ax.set_title(title)
        plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig(project_path("results/tester/eval_delta_surface.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Figure 3 saved: eval_delta_surface.png")


# -- Figure 4: Statistical Diagnostics ----------------------------------------

def fig_statistics(pnl_nn: np.ndarray, pnl_bsm: np.ndarray,
                   S_test: np.ndarray, net: HedgingNet) -> None:
    nn_d  = nn_deltas_batch(S_test, net)
    bsm_d = bsm_deltas_batch(S_test)
    delta_err = nn_d - bsm_d              # (N_PATHS, N)
    times     = np.linspace(0, T, N)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("Statistical Diagnostics", fontsize=14, y=1.01)

    # -- (0,0) QQ plot of NN P&L vs Normal --
    ax = axes[0, 0]
    (osm, osr), (slope, intercept, _) = stats.probplot(pnl_nn, dist="norm")
    ax.scatter(osm, osr, s=4, alpha=0.4, color="#4C72B0", label="NN P&L quantiles")
    x_line = np.array([osm.min(), osm.max()])
    ax.plot(x_line, slope * x_line + intercept, color="red",
            linewidth=1.2, label="Normal reference")
    ax.set_xlabel("Theoretical normal quantiles")
    ax.set_ylabel("Sample quantiles")
    ax.set_title("QQ Plot: NN P&L vs Normal")
    ax.legend(fontsize=9)

    # -- (0,1) Mean absolute delta error over time by moneyness bucket --
    ax    = axes[0, 1]
    S_T   = S_test[N, :]
    buckets = [
        (S_T < 0.95,          "OTM  (S_T < 0.95)",   "#E74C3C"),
        ((S_T >= 0.95) & (S_T < 1.00), "OTM-ATM",    "#E67E22"),
        ((S_T >= 1.00) & (S_T < 1.05), "ATM-ITM",    "#27AE60"),
        (S_T >= 1.05,          "ITM  (S_T > 1.05)",   "#2980B9"),
    ]
    for mask, label, color in buckets:
        if mask.sum() > 10:
            mean_abs_err = np.abs(delta_err[mask]).mean(axis=0)
            ax.plot(times, mean_abs_err, label=f"{label} (n={mask.sum()})",
                    color=color, linewidth=1.2)
    ax.set_xlabel("Time")
    ax.set_ylabel("|NN delta − BSM delta|")
    ax.set_title("Mean Absolute Delta Error Over Time")
    ax.legend(fontsize=8)

    # -- (1,0) NN delta vs BSM delta scatter at τ = 0.5 --
    ax    = axes[1, 0]
    t_mid = N // 2
    ax.scatter(bsm_d[:, t_mid], nn_d[:, t_mid],
               s=3, alpha=0.25, color="#4C72B0")
    lims = [min(bsm_d[:, t_mid].min(), nn_d[:, t_mid].min()),
            max(bsm_d[:, t_mid].max(), nn_d[:, t_mid].max())]
    ax.plot(lims, lims, color="red", linewidth=1.0, linestyle="--", label="Perfect agreement")
    corr = np.corrcoef(bsm_d[:, t_mid], nn_d[:, t_mid])[0, 1]
    ax.set_xlabel("BSM delta at τ = 0.5")
    ax.set_ylabel("NN delta at τ = 0.5")
    ax.set_title(f"NN vs BSM Delta at Mid-Life  (r = {corr:.4f})")
    ax.legend(fontsize=9)

    # -- (1,1) Distribution of per-path mean absolute delta error --
    ax = axes[1, 1]
    per_path_mae = np.abs(delta_err).mean(axis=1)
    ax.hist(per_path_mae, bins=50, color="#4C72B0", alpha=0.8, edgecolor="none", density=True)
    ax.axvline(per_path_mae.mean(), color="red", linewidth=1.2, linestyle="--",
               label=f"Mean MAE = {per_path_mae.mean():.4f}")
    ax.set_xlabel("Per-path mean |NN delta − BSM delta|")
    ax.set_ylabel("Density")
    ax.set_title("Distribution of Per-Path Delta Tracking Error")
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(project_path("results/tester/eval_statistics.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Figure 4 saved: eval_statistics.png")


# -- Main ----------------------------------------------------------------------

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
    fig_pnl_distributions(pnl_nn, pnl_bsm, pnl_unhedged)
    fig_pnl_analysis(pnl_nn, pnl_bsm, S_test)
    fig_delta_paths(S_test, net)
    fig_delta_surface(net)
    fig_statistics(pnl_nn, pnl_bsm, S_test, net)

    print(f"\nAll outputs saved to results/tester/")

"""
3-Way Loss Function Comparison: MSE vs CVaR vs Entropic Risk Measure

Loads all three sets of models and produces side-by-side comparison plots
showing how the choice of loss function affects hedging strategy and tail risk
across κ values.

Run order:
    python "project/transaction costs/MultiTrain.py"      # MSE models
    python "project/transaction costs/CVaRTrain.py"       # CVaR models
    python "project/transaction costs/EntropicTrain.py"   # Entropic models
    python "project/transaction costs/CompareEvaluate.py" # this file

All plots saved to results/transaction costs/figures/compare_*.png
"""

import sys
import pathlib

_root = pathlib.Path(__file__).resolve()
while not (_root / "pyproject.toml").exists():
    _root = _root.parent
sys.path.insert(0, str(_root))
sys.path.insert(0, str(pathlib.Path(__file__).parent))

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import norm

from project.stock.generators import generate_gbm
from project.helpers.path_helpers import project_path

from pricer          import (HedgingNet as NetMSE,
                              run_paths  as run_mse,
                              load_model as load_mse,
                              DEVICE, S0, K, sigma, r, N, T, h, N_PATHS_TEST)
from pricer_cvar     import (HedgingNet as NetCVaR,
                              run_paths  as run_cvar,
                              load_model as load_cvar,
                              ALPHA)
from pricer_entropic import (HedgingNet as NetEntropic,
                              run_paths  as run_entropic,
                              load_model as load_entropic,
                              GAMMA)

os.makedirs(project_path("results/transaction costs/figures"), exist_ok=True)

KAPPAS = [0.0, 0.001, 0.005, 0.01, 0.02]

# Colour scheme — consistent across all plots
C_MSE      = "#4C72B0"   # blue
C_CVAR     = "#DD8452"   # orange
C_ENTROPIC = "#55A868"   # green
C_BSM      = "black"


# ── BSM helpers ───────────────────────────────────────────────────────────────

def bsm_delta(S, K, r, sigma, t, T):
    tau = T - t
    if tau <= 0:
        return np.where(S > K, 1.0, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    return norm.cdf(d1)


def bsm_call(S0, K, r, sigma, T):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bsm_hedge_pnl(S: np.ndarray, kappa: float = 0.0) -> np.ndarray:
    n_paths    = S.shape[1]
    currency   = np.zeros(n_paths)
    underlying = np.zeros(n_paths)
    prev_delta = np.zeros(n_paths)
    premium    = bsm_call(S0, K, r, sigma, T)
    for t in range(N):
        St         = S[t, :]
        delta      = bsm_delta(St, K, r, sigma, t * h, T)
        trade      = delta - prev_delta
        currency  -= trade * St + kappa * np.abs(trade) * St
        underlying += trade
        currency  *= np.exp(r * h)
        prev_delta = delta
    S_T    = S[N, :]
    payoff = np.maximum(S_T - K, 0)
    return premium + underlying * S_T + currency - payoff


def get_nn_deltas(S_test: np.ndarray, net) -> np.ndarray:
    net.eval()
    n_paths    = S_test.shape[1]
    all_deltas = np.zeros((n_paths, N))
    with torch.no_grad():
        for t in range(N):
            St  = torch.tensor(S_test[t, :] / K, dtype=torch.float32).to(DEVICE)
            tau = torch.full((n_paths,), 1.0 - t / N, device=DEVICE)
            all_deltas[:, t] = net(torch.stack([St, tau], dim=1)).cpu().numpy()
    return all_deltas


# ── Tail risk metrics ─────────────────────────────────────────────────────────

def var_metric(pnl: np.ndarray, alpha: float = ALPHA) -> float:
    return float(np.quantile(-pnl, alpha))


def cvar_metric(pnl: np.ndarray, alpha: float = ALPHA) -> float:
    losses = -pnl
    var    = np.quantile(losses, alpha)
    return float(losses[losses >= var].mean())


# ── Per-kappa evaluation ──────────────────────────────────────────────────────

def evaluate_kappa(kappa: float, S_test: np.ndarray) -> dict:
    S_tensor = torch.tensor(S_test.T, dtype=torch.float32).to(DEVICE)

    net_mse      = load_mse(kappa=kappa)
    net_cvar     = load_cvar(kappa=kappa)
    net_entropic = load_entropic(kappa=kappa)

    with torch.no_grad():
        pnl_mse      = run_mse(S_tensor,      net_mse,      kappa=kappa).cpu().numpy()
        pnl_cvar     = run_cvar(S_tensor,     net_cvar,     kappa=kappa).cpu().numpy()
        pnl_entropic = run_entropic(S_tensor, net_entropic, kappa=kappa).cpu().numpy()

    return {
        "kappa":         kappa,
        "pnl_mse":       pnl_mse,
        "pnl_cvar":      pnl_cvar,
        "pnl_entropic":  pnl_entropic,
        "pnl_bsm":       bsm_hedge_pnl(S_test, kappa=kappa),
        "deltas_mse":      get_nn_deltas(S_test, net_mse),
        "deltas_cvar":     get_nn_deltas(S_test, net_cvar),
        "deltas_entropic": get_nn_deltas(S_test, net_entropic),
        "premium_mse":      net_mse.premium.item(),
        "premium_cvar":     net_cvar.premium.item(),
        "premium_entropic": net_entropic.premium.item(),
    }


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_tail_comparison(results: list) -> None:
    """VaR and ES for all three loss functions across kappas."""
    kappas = [r["kappa"] for r in results]
    x      = np.arange(len(kappas))
    width  = 0.25

    metrics = {
        "VaR":  ([var_metric(r["pnl_mse"])      for r in results],
                 [var_metric(r["pnl_cvar"])      for r in results],
                 [var_metric(r["pnl_entropic"])  for r in results]),
        "CVaR / ES": ([cvar_metric(r["pnl_mse"])      for r in results],
                      [cvar_metric(r["pnl_cvar"])      for r in results],
                      [cvar_metric(r["pnl_entropic"])  for r in results]),
    }

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    fig.suptitle(f"Tail Risk: MSE vs CVaR (α={ALPHA}) vs Entropic (γ={GAMMA})", fontsize=13)

    for ax, (title, (m, c, e)) in zip(axes, metrics.items()):
        ax.bar(x - width, m, width, label="MSE",      color=C_MSE)
        ax.bar(x,          c, width, label=f"CVaR α={ALPHA}", color=C_CVAR)
        ax.bar(x + width,  e, width, label=f"Entropic γ={GAMMA}", color=C_ENTROPIC)
        ax.set_xticks(x)
        ax.set_xticklabels([str(k) for k in kappas])
        ax.set_xlabel("κ")
        ax.set_ylabel(f"{title} of Losses (lower = better)")
        ax.set_title(title)
        ax.legend(fontsize=9)

    plt.tight_layout()
    path = project_path("results/transaction costs/figures/compare_tail_risk.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print("Tail risk comparison saved.")


def plot_pnl_distributions(results: list) -> None:
    """P&L distributions for all three loss functions — 3 rows × 5 columns."""
    n_kappas = len(results)
    rows     = [("MSE", "pnl_mse", C_MSE),
                (f"CVaR α={ALPHA}", "pnl_cvar", C_CVAR),
                (f"Entropic γ={GAMMA}", "pnl_entropic", C_ENTROPIC)]

    fig, axes = plt.subplots(3, n_kappas, figsize=(4 * n_kappas, 9), sharey="row")
    fig.suptitle("P&L Distributions: MSE vs CVaR vs Entropic", fontsize=13)

    for row_idx, (label, key, color) in enumerate(rows):
        for col_idx, res in enumerate(results):
            pnl = res[key]
            ax  = axes[row_idx, col_idx]
            ax.hist(pnl, bins=50, color=color, alpha=0.8, edgecolor="none")
            ax.axvline(pnl.mean(), color="red",   linestyle="--", linewidth=1.2,
                       label=f"Mean {pnl.mean():.4f}")
            ax.axvline(np.quantile(pnl, 0.05), color="black", linestyle=":",
                       linewidth=1.1, label=f"P5 {np.quantile(pnl,0.05):.4f}")
            ax.axvline(0, color="grey", linewidth=0.6)
            ax.set_title(f"κ={res['kappa']} — {label}")
            ax.legend(fontsize=7)
            if col_idx == 0:
                ax.set_ylabel("Count")
            if row_idx == 2:
                ax.set_xlabel("P&L")

    plt.tight_layout()
    path = project_path("results/transaction costs/figures/compare_pnl_distributions.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print("P&L distributions saved.")


def plot_delta_paths(results: list, S_test: np.ndarray) -> None:
    """Delta paths on the same ITM path: MSE / CVaR / Entropic / BSM."""
    S_T  = S_test[N, :]
    itm  = np.where(S_T > K * 1.05)[0]
    if len(itm) == 0:
        print("No ITM path — skipping delta comparison.")
        return
    idx   = itm[np.argmax(S_T[itm])]
    times = np.linspace(0, T, N)
    bsm_d = np.array([bsm_delta(S_test[t, idx], K, r, sigma, t * h, T) for t in range(N)])

    n   = len(results)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3.5), sharey=True)
    fig.suptitle(f"Delta Paths (ITM, S_T={S_T[idx]:.3f}): MSE vs CVaR vs Entropic", fontsize=13)

    for ax, res in zip(axes, results):
        ax.plot(times, res["deltas_mse"][idx],      color=C_MSE,      linewidth=1.2,
                label="MSE")
        ax.plot(times, res["deltas_cvar"][idx],     color=C_CVAR,     linewidth=1.2,
                label=f"CVaR α={ALPHA}")
        ax.plot(times, res["deltas_entropic"][idx], color=C_ENTROPIC, linewidth=1.2,
                label=f"Entropic γ={GAMMA}")
        ax.plot(times, bsm_d,                       color=C_BSM,      linewidth=1.0,
                linestyle="--", label="BSM")
        ax.set_title(f"κ = {res['kappa']}")
        ax.set_xlabel("Time")
        if ax == axes[0]:
            ax.set_ylabel("Delta")
        ax.legend(fontsize=7)

    plt.tight_layout()
    path = project_path("results/transaction costs/figures/compare_delta_paths.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print("Delta paths comparison saved.")


def plot_std_comparison(results: list) -> None:
    """P&L standard deviation for all three loss functions across kappas."""
    kappas = [r["kappa"] for r in results]
    x      = np.arange(len(kappas))
    width  = 0.25

    std_mse      = [r["pnl_mse"].std()      for r in results]
    std_cvar     = [r["pnl_cvar"].std()     for r in results]
    std_entropic = [r["pnl_entropic"].std() for r in results]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - width, std_mse,      width, label="MSE",                color=C_MSE)
    ax.bar(x,          std_cvar,     width, label=f"CVaR α={ALPHA}",    color=C_CVAR)
    ax.bar(x + width,  std_entropic, width, label=f"Entropic γ={GAMMA}", color=C_ENTROPIC)
    ax.set_xticks(x)
    ax.set_xticklabels([str(k) for k in kappas])
    ax.set_xlabel("κ")
    ax.set_ylabel("Std P&L")
    ax.set_title("P&L Volatility: MSE vs CVaR vs Entropic")
    ax.legend(fontsize=9)
    plt.tight_layout()
    path = project_path("results/transaction costs/figures/compare_std.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print("Std comparison saved.")


def print_summary_table(results: list) -> None:
    alpha = ALPHA
    w = 110
    print("\n" + "=" * w)
    print("  3-WAY LOSS FUNCTION COMPARISON SUMMARY")
    print("=" * w)
    hdr = (f"  {'κ':>6}  "
           f"{'MSE Mean':>9} {'MSE Std':>8} {'MSE ES':>8}  |  "
           f"{'CVaR Mean':>9} {'CVaR Std':>9} {'CVaR ES':>8}  |  "
           f"{'ER Mean':>8} {'ER Std':>8} {'ER ES':>7}")
    print(hdr)
    print("  " + "─" * (w - 2))
    for res in results:
        pm = res["pnl_mse"]
        pc = res["pnl_cvar"]
        pe = res["pnl_entropic"]
        print(f"  {res['kappa']:>6}  "
              f"{pm.mean():>9.4f} {pm.std():>8.4f} {cvar_metric(pm):>8.4f}  |  "
              f"{pc.mean():>9.4f} {pc.std():>9.4f} {cvar_metric(pc):>8.4f}  |  "
              f"{pe.mean():>8.4f} {pe.std():>8.4f} {cvar_metric(pe):>7.4f}")
    print("=" * w)
    print(f"\n  ES = Expected Shortfall at α={alpha}  |  "
          f"CVaR uses α={ALPHA}  |  Entropic uses γ={GAMMA}")
    print(f"  Premiums — MSE: {results[0]['premium_mse']:.4f}  "
          f"CVaR: {results[0]['premium_cvar']:.4f}  "
          f"Entropic: {results[0]['premium_entropic']:.4f}  (at κ=0)\n")


if __name__ == "__main__":
    print("── 3-way loss function comparison ──")

    print("\n── Simulating test paths ──")
    S_test = generate_gbm(S0, r, sigma, h, N_PATHS_TEST, N + 1)

    print("\n── Evaluating each κ ──")
    results = []
    for kappa in KAPPAS:
        print(f"  κ = {kappa} ...", end=" ", flush=True)
        results.append(evaluate_kappa(kappa, S_test))
        print("done")

    print_summary_table(results)

    print("\n── Generating plots ──")
    plot_tail_comparison(results)
    plot_pnl_distributions(results)
    plot_delta_paths(results, S_test)
    plot_std_comparison(results)

    print("\nAll done. Figures saved to results/transaction costs/figures/compare_*.png")

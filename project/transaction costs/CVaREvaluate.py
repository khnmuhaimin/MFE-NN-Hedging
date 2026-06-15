"""
CVaR vs MSE Evaluation
Loads both CVaR and MSE models and produces side-by-side comparison plots
focused on tail risk, P&L distributions, and hedging behaviour.

Run order:
    python "project/transaction costs/MultiTrain.py"   # train MSE models
    python "project/transaction costs/CVaRTrain.py"    # train CVaR models
    python "project/transaction costs/CVaREvaluate.py" # compare

All plots saved to results/transaction costs/figures/cvar_*.png
"""

import sys
import pathlib

_root = pathlib.Path(__file__).resolve()
while not (_root / "pyproject.toml").exists():
    _root = _root.parent
sys.path.insert(0, str(_root))
sys.path.insert(0, str(pathlib.Path(__file__).parent))

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import norm

from project.stock.generators import generate_gbm
from project.helpers.path_helpers import project_path

# Import HedgingNet and run_paths from each pricer.
# Both use the same architecture and path rollout — only the loss differs.
from pricer      import (HedgingNet as HedgingNetMSE,
                         run_paths  as run_paths_mse,
                         load_model as load_model_mse,
                         DEVICE, S0, K, sigma, r, N, T, h, N_PATHS_TEST)
from pricer_cvar import (HedgingNet as HedgingNetCVaR,
                         run_paths  as run_paths_cvar,
                         load_model as load_model_cvar,
                         ALPHA)

KAPPAS = [0.0, 0.001, 0.005, 0.01, 0.02]

os_makedirs = __import__("os").makedirs
os_makedirs(project_path("results/transaction costs/figures"), exist_ok=True)


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


def get_nn_deltas(S_test: np.ndarray, hedging_net) -> np.ndarray:
    hedging_net.eval()
    n_paths    = S_test.shape[1]
    all_deltas = np.zeros((n_paths, N))
    with torch.no_grad():
        for t in range(N):
            St  = torch.tensor(S_test[t, :] / K, dtype=torch.float32).to(DEVICE)
            tau = torch.full((n_paths,), 1.0 - t / N, device=DEVICE)
            all_deltas[:, t] = hedging_net(torch.stack([St, tau], dim=1)).cpu().numpy()
    return all_deltas


def cvar_metric(pnl: np.ndarray, alpha: float = ALPHA) -> float:
    losses = -pnl
    var    = np.quantile(losses, alpha)
    return float(losses[losses >= var].mean())


def var_metric(pnl: np.ndarray, alpha: float = ALPHA) -> float:
    return float(np.quantile(-pnl, alpha))


# ── Per-kappa evaluation ──────────────────────────────────────────────────────

def evaluate_pair(kappa: float, S_test: np.ndarray) -> dict:
    S_tensor = torch.tensor(S_test.T, dtype=torch.float32).to(DEVICE)

    net_mse  = load_model_mse(kappa=kappa)
    net_cvar = load_model_cvar(kappa=kappa)

    with torch.no_grad():
        pnl_mse  = run_paths_mse(S_tensor,  net_mse,  kappa=kappa).cpu().numpy()
        pnl_cvar = run_paths_cvar(S_tensor, net_cvar, kappa=kappa).cpu().numpy()

    return {
        "kappa":         kappa,
        "pnl_mse":       pnl_mse,
        "pnl_cvar":      pnl_cvar,
        "pnl_bsm":       bsm_hedge_pnl(S_test, kappa=kappa),
        "premium_mse":   net_mse.premium.item(),
        "premium_cvar":  net_cvar.premium.item(),
        "deltas_mse":    get_nn_deltas(S_test, net_mse),
        "deltas_cvar":   get_nn_deltas(S_test, net_cvar),
    }


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_pnl_comparison(results: list) -> None:
    """Side-by-side P&L distributions: MSE vs CVaR for each kappa."""
    n   = len(results)
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 7), sharey="row")
    fig.suptitle("P&L Distributions: MSE vs CVaR Loss", fontsize=13)

    for col, res in enumerate(results):
        kappa = res["kappa"]
        for row, (pnl, label, color) in enumerate([
            (res["pnl_mse"],  "MSE",  "#4C72B0"),
            (res["pnl_cvar"], "CVaR", "#DD8452"),
        ]):
            ax = axes[row, col]
            ax.hist(pnl, bins=50, color=color, alpha=0.8, edgecolor="none")
            ax.axvline(pnl.mean(), color="red",   linestyle="--", linewidth=1.2,
                       label=f"Mean {pnl.mean():.4f}")
            ax.axvline(np.quantile(pnl, 0.05), color="black", linestyle=":",
                       linewidth=1.1, label=f"P5 {np.quantile(pnl, 0.05):.4f}")
            ax.axvline(0, color="grey", linewidth=0.7)
            ax.set_title(f"κ={kappa} — {label}")
            ax.set_xlabel("P&L")
            ax.legend(fontsize=7)
            if col == 0:
                ax.set_ylabel("Count")

    plt.tight_layout()
    path = project_path("results/transaction costs/figures/cvar_pnl_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print("P&L comparison plot saved.")


def plot_tail_metrics(results: list) -> None:
    """VaR and CVaR at alpha=0.95 for MSE vs CVaR across kappas."""
    kappas = [r["kappa"] for r in results]

    var_mse  = [var_metric(r["pnl_mse"])  for r in results]
    var_cvar = [var_metric(r["pnl_cvar"]) for r in results]
    es_mse   = [cvar_metric(r["pnl_mse"])  for r in results]
    es_cvar  = [cvar_metric(r["pnl_cvar"]) for r in results]

    x      = np.arange(len(kappas))
    width  = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"Tail Risk Comparison: MSE vs CVaR (α={ALPHA})", fontsize=13)

    axes[0].bar(x - width/2, var_mse,  width, label="MSE",  color="#4C72B0")
    axes[0].bar(x + width/2, var_cvar, width, label="CVaR", color="#DD8452")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([str(k) for k in kappas])
    axes[0].set_xlabel("κ")
    axes[0].set_ylabel(f"VaR_{ALPHA} of Losses")
    axes[0].set_title(f"VaR (worst {100*(1-ALPHA):.0f}% threshold)")
    axes[0].legend(fontsize=9)

    axes[1].bar(x - width/2, es_mse,  width, label="MSE",  color="#4C72B0")
    axes[1].bar(x + width/2, es_cvar, width, label="CVaR", color="#DD8452")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([str(k) for k in kappas])
    axes[1].set_xlabel("κ")
    axes[1].set_ylabel(f"CVaR_{ALPHA} of Losses")
    axes[1].set_title(f"CVaR / Expected Shortfall (worst {100*(1-ALPHA):.0f}%)")
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    path = project_path("results/transaction costs/figures/cvar_tail_metrics.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print("Tail metrics plot saved.")


def plot_learning_curves_cvar(results: list) -> None:
    """MSE vs CVaR learning curves for each kappa on separate subplots."""
    n   = len(results)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3.5), sharey=False)
    fig.suptitle("Learning Curves: MSE vs CVaR", fontsize=13)

    for ax, res in zip(axes, results):
        kappa = res["kappa"]

        mse_path  = project_path(f"results/transaction costs/models/tc_epoch_losses_kappa{kappa}.npy")
        cvar_path = project_path(f"results/transaction costs/models/tc_epoch_losses_kappa{kappa}_cvar.npy")

        mse_losses  = np.load(mse_path)
        cvar_losses = np.load(cvar_path)

        epochs = np.arange(1, len(mse_losses) + 1)
        ax2    = ax.twinx()

        l1, = ax.plot(epochs, mse_losses,  color="#4C72B0", linewidth=1.3, label="MSE loss")
        l2, = ax2.plot(epochs, cvar_losses, color="#DD8452", linewidth=1.3, label="CVaR loss")

        ax.set_title(f"κ = {kappa}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss",  color="#4C72B0")
        ax2.set_ylabel("CVaR Loss", color="#DD8452")
        ax.tick_params(axis="y", colors="#4C72B0")
        ax2.tick_params(axis="y", colors="#DD8452")
        ax.legend(handles=[l1, l2], fontsize=7, loc="upper right")

    plt.tight_layout()
    path = project_path("results/transaction costs/figures/cvar_learning_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print("CVaR learning curves plot saved.")


def plot_delta_comparison(results: list, S_test: np.ndarray) -> None:
    """MSE vs CVaR delta on the same ITM path for each kappa."""
    S_T  = S_test[N, :]
    itm  = np.where(S_T > K * 1.05)[0]
    if len(itm) == 0:
        print("No ITM path found — skipping delta comparison.")
        return
    idx   = itm[np.argmax(S_T[itm])]
    times = np.linspace(0, T, N)

    bsm_d = np.array([bsm_delta(S_test[t, idx], K, r, sigma, t * h, T) for t in range(N)])

    n   = len(results)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3.5), sharey=True)
    fig.suptitle(f"Delta Path Comparison: MSE vs CVaR (ITM path, S_T={S_T[idx]:.3f})", fontsize=13)

    for ax, res in zip(axes, results):
        ax.plot(times, res["deltas_mse"][idx],  label="MSE",       color="#4C72B0", linewidth=1.2)
        ax.plot(times, res["deltas_cvar"][idx],  label=f"CVaR α={ALPHA}", color="#DD8452", linewidth=1.2)
        ax.plot(times, bsm_d,                    label="BSM",       color="black",   linewidth=1.0, linestyle="--")
        ax.set_title(f"κ = {res['kappa']}")
        ax.set_xlabel("Time")
        if ax == axes[0]:
            ax.set_ylabel("Delta")
        ax.legend(fontsize=7)

    plt.tight_layout()
    path = project_path("results/transaction costs/figures/cvar_delta_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print("Delta comparison plot saved.")


def print_summary_table(results: list) -> None:
    alpha = ALPHA
    w = 100
    print("\n" + "=" * w)
    print("  CVaR vs MSE COMPARISON SUMMARY")
    print("=" * w)
    hdr = (f"  {'κ':>6}  "
           f"{'MSE Mean':>10}  {'MSE Std':>8}  "
           f"{'CVaR Mean':>10}  {'CVaR Std':>9}  "
           f"|  {'MSE VaR':>9}  {'MSE ES':>8}  "
           f"{'CVaR VaR':>9}  {'CVaR ES':>8}")
    print(hdr)
    print("  " + "─" * (w - 2))
    for res in results:
        pm = res["pnl_mse"]
        pc = res["pnl_cvar"]
        print(f"  {res['kappa']:>6}  "
              f"{pm.mean():>10.4f}  {pm.std():>8.4f}  "
              f"{pc.mean():>10.4f}  {pc.std():>9.4f}  "
              f"|  {var_metric(pm):>9.4f}  {cvar_metric(pm):>8.4f}  "
              f"{var_metric(pc):>9.4f}  {cvar_metric(pc):>8.4f}")
    print("=" * w + "\n")
    print(f"  VaR / CVaR computed at α = {alpha} (left-tail losses, i.e. -P&L)")


if __name__ == "__main__":
    print("── CVaR vs MSE evaluation ──")

    print("\n── Simulating test paths ──")
    S_test = generate_gbm(S0, r, sigma, h, N_PATHS_TEST, N + 1)

    print("\n── Evaluating each κ ──")
    results = []
    for kappa in KAPPAS:
        print(f"  κ = {kappa} ...", end=" ", flush=True)
        results.append(evaluate_pair(kappa, S_test))
        print("done")

    print_summary_table(results)

    print("\n── Generating plots ──")
    plot_learning_curves_cvar(results)
    plot_pnl_comparison(results)
    plot_tail_metrics(results)
    plot_delta_comparison(results, S_test)

    print("\nAll done. Figures saved to results/transaction costs/figures/cvar_*.png")

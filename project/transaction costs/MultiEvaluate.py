"""
Multi-Kappa Evaluation
Loads all trained models and produces comparison plots across kappa values.
Run MultiTrain.py first.
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
from pricer import HedgingNet, run_paths, load_model, DEVICE, S0, K, sigma, r, N, T, h, N_PATHS_TEST

KAPPAS = [0.0, 0.001, 0.005, 0.01, 0.02]


# ── BSM helpers ───────────────────────────────────────────────────────────────

def bsm_call(S0, K, r, sigma, T):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bsm_delta(S, K, r, sigma, t, T):
    tau = T - t
    if tau <= 0:
        return np.where(S > K, 1.0, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    return norm.cdf(d1)


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


def get_nn_deltas_batch(S_test: np.ndarray, hedging_net: HedgingNet) -> np.ndarray:
    """Returns (N_PATHS, N) array of NN deltas."""
    hedging_net.eval()
    n_paths    = S_test.shape[1]
    all_deltas = np.zeros((n_paths, N))
    with torch.no_grad():
        for t in range(N):
            St  = torch.tensor(S_test[t, :] / K, dtype=torch.float32).to(DEVICE)
            tau = torch.full((n_paths,), 1.0 - t / N, device=DEVICE)
            all_deltas[:, t] = hedging_net(torch.stack([St, tau], dim=1)).cpu().numpy()
    return all_deltas


def get_bsm_deltas_batch(S_test: np.ndarray) -> np.ndarray:
    """Returns (N_PATHS, N) array of BSM deltas."""
    n_paths    = S_test.shape[1]
    all_deltas = np.zeros((n_paths, N))
    for t in range(N):
        all_deltas[:, t] = bsm_delta(S_test[t, :], K, r, sigma, t * h, T)
    return all_deltas


def compute_total_tc(S_test: np.ndarray, deltas: np.ndarray, kappa: float) -> np.ndarray:
    """Total TC paid per path given a (N_PATHS, N) delta array."""
    trades  = np.abs(np.diff(deltas, prepend=0, axis=1))
    S_steps = S_test[:N, :].T
    return (kappa * trades * S_steps).sum(axis=1)


# ── Evaluation per kappa ──────────────────────────────────────────────────────

def evaluate_kappa(kappa: float, S_test: np.ndarray):
    """
    Load the model for this kappa and compute all stats.
    Returns a dict of results.
    """
    hedging_net = load_model(kappa=kappa)
    hedging_net.eval()

    S_tensor = torch.tensor(S_test.T, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        nn_pnl = run_paths(S_tensor, hedging_net, kappa=kappa).cpu().numpy()

    bsm_pnl_tc  = bsm_hedge_pnl(S_test, kappa=kappa)
    bsm_pnl_no_tc = bsm_hedge_pnl(S_test, kappa=0.0)

    nn_deltas  = get_nn_deltas_batch(S_test, hedging_net)
    bsm_deltas = get_bsm_deltas_batch(S_test)

    nn_tc  = compute_total_tc(S_test, nn_deltas,  kappa)
    bsm_tc = compute_total_tc(S_test, bsm_deltas, kappa)

    tc_reduction = (1 - nn_tc.mean() / bsm_tc.mean()) * 100 if bsm_tc.mean() > 0 else 0.0

    return {
        "kappa":          kappa,
        "nn_pnl":         nn_pnl,
        "bsm_pnl_tc":     bsm_pnl_tc,
        "bsm_pnl_no_tc":  bsm_pnl_no_tc,
        "nn_tc":          nn_tc,
        "bsm_tc":         bsm_tc,
        "tc_reduction":   tc_reduction,
        "premium":        hedging_net.premium.item(),
        "nn_deltas":      nn_deltas,
        "bsm_deltas":     bsm_deltas,
    }


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_pnl_distributions(results: list) -> None:
    """P&L distribution for NN hedge at each kappa on one figure."""
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), sharey=True)
    fig.suptitle("NN Terminal P&L Distribution by κ", fontsize=13)

    for ax, res in zip(axes, results):
        ax.hist(res["nn_pnl"], bins=50, edgecolor="none", alpha=0.8)
        ax.axvline(res["nn_pnl"].mean(), color="red",   linestyle="--", linewidth=1.2,
                   label=f"Mean {res['nn_pnl'].mean():.4f}")
        ax.axvline(0, color="black", linestyle=":", linewidth=1.0)
        ax.set_title(f"κ = {res['kappa']}")
        ax.set_xlabel("P&L")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(project_path("results/transaction costs/figures/multi_pnl_distributions.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print("P&L distributions plot saved.")


def plot_summary_stats(results: list) -> None:
    """Mean P&L and Std P&L for NN vs BSM across kappas."""
    kappas       = [r["kappa"]              for r in results]
    nn_means     = [r["nn_pnl"].mean()      for r in results]
    bsm_means    = [r["bsm_pnl_tc"].mean()  for r in results]
    nn_stds      = [r["nn_pnl"].std()       for r in results]
    bsm_stds     = [r["bsm_pnl_tc"].std()   for r in results]

    x     = np.arange(len(kappas))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("NN vs BSM Performance Across κ Values", fontsize=13)

    axes[0].bar(x - width/2, nn_means,  width, label="NN Hedge")
    axes[0].bar(x + width/2, bsm_means, width, label="BSM Hedge (with TC)")
    axes[0].axhline(0, color="black", linewidth=0.8, linestyle="--")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([str(k) for k in kappas])
    axes[0].set_xlabel("κ")
    axes[0].set_ylabel("Mean P&L")
    axes[0].set_title("Mean P&L")
    axes[0].legend(fontsize=9)

    axes[1].bar(x - width/2, nn_stds,  width, label="NN Hedge")
    axes[1].bar(x + width/2, bsm_stds, width, label="BSM Hedge (with TC)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([str(k) for k in kappas])
    axes[1].set_xlabel("κ")
    axes[1].set_ylabel("Std P&L")
    axes[1].set_title("P&L Volatility")
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(project_path("results/transaction costs/figures/multi_summary_stats.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print("Summary stats plot saved.")


def plot_tc_reduction(results: list) -> None:
    """TC reduction % and average total TC paid (NN vs BSM) across kappas."""
    kappas       = [r["kappa"]            for r in results]
    reductions   = [r["tc_reduction"]     for r in results]
    nn_tc_means  = [r["nn_tc"].mean()     for r in results]
    bsm_tc_means = [r["bsm_tc"].mean()    for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Transaction Cost Analysis Across κ Values", fontsize=13)

    axes[0].plot(kappas, reductions, marker="o", linewidth=1.5)
    axes[0].set_xlabel("κ")
    axes[0].set_ylabel("TC Reduction (%)")
    axes[0].set_title("TC Reduction: NN vs BSM")

    axes[1].plot(kappas, nn_tc_means,  marker="o", label="NN Hedge",  linewidth=1.5)
    axes[1].plot(kappas, bsm_tc_means, marker="s", label="BSM Hedge", linewidth=1.5, linestyle="--")
    axes[1].set_xlabel("κ")
    axes[1].set_ylabel("Average Total TC Paid")
    axes[1].set_title("Avg Total TC per Path")
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(project_path("results/transaction costs/figures/multi_tc_reduction.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print("TC reduction plot saved.")


def plot_delta_paths_comparison(results: list, S_test: np.ndarray) -> None:
    """NN delta on one ITM path for each kappa, compared to BSM."""
    S_T            = S_test[N, :]
    itm_candidates = np.where(S_T > K * 1.05)[0]
    if len(itm_candidates) == 0:
        print("No ITM paths found — skipping delta comparison plot.")
        return

    itm_idx = itm_candidates[np.argmax(S_T[itm_candidates])]
    times   = np.linspace(0, T, N)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_title(f"NN Delta on ITM Path (S_T = {S_T[itm_idx]:.3f}) — All κ Values", fontsize=13)

    for res in results:
        ax.plot(times, res["nn_deltas"][itm_idx], label=f"NN κ={res['kappa']}", linewidth=1.2)

    bsm_d = results[0]["bsm_deltas"][itm_idx]
    ax.plot(times, bsm_d, label="BSM (no TC)", linewidth=1.5, linestyle="--", color="black")

    ax.set_xlabel("Time")
    ax.set_ylabel("Delta (shares held)")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(project_path("results/transaction costs/figures/multi_delta_paths.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print("Delta path comparison plot saved.")


def plot_learning_curves(results: list) -> None:
    """Learning curves for all kappa models on one plot."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_title("Training Loss by κ", fontsize=13)

    for res in results:
        kappa = res["kappa"]
        path  = project_path(f"results/transaction costs/models/tc_epoch_losses_kappa{kappa}.npy")
        losses = np.load(path).tolist()
        ax.plot(range(1, len(losses) + 1), losses, label=f"κ={kappa}", linewidth=1.2)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(project_path("results/transaction costs/figures/multi_learning_curves.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print("Learning curves plot saved.")


def print_summary_table(results: list) -> None:
    print("\n" + "=" * 85)
    print("  MULTI-KAPPA SUMMARY")
    print("=" * 85)
    print(f"  {'κ':>6}  {'NN Mean':>10}  {'NN Std':>10}  {'BSM Mean':>10}  "
          f"{'BSM Std':>10}  {'TC Reduc%':>10}  {'NN Premium':>10}")
    print(f"  {'─'*77}")
    for res in results:
        print(f"  {res['kappa']:>6}  "
              f"{res['nn_pnl'].mean():>10.4f}  "
              f"{res['nn_pnl'].std():>10.4f}  "
              f"{res['bsm_pnl_tc'].mean():>10.4f}  "
              f"{res['bsm_pnl_tc'].std():>10.4f}  "
              f"{res['tc_reduction']:>9.1f}%  "
              f"{res['premium']:>10.4f}")
    print("=" * 85 + "\n")


if __name__ == "__main__":
    print("── Multi-kappa evaluation ──")

    print("\n── Simulating test paths ──")
    S_test = generate_gbm(S0, r, sigma, h, N_PATHS_TEST, N + 1)

    print("\n── Evaluating each κ ──")
    results = []
    for kappa in KAPPAS:
        print(f"  κ = {kappa} ...", end=" ")
        results.append(evaluate_kappa(kappa, S_test))
        print("done")

    print_summary_table(results)

    print("── Generating plots ──")
    plot_learning_curves(results)
    plot_pnl_distributions(results)
    plot_summary_stats(results)
    plot_tc_reduction(results)
    plot_delta_paths_comparison(results, S_test)

    print("\nAll done.")

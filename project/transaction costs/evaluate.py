"""
Deep Hedging with Transaction Costs — Evaluation & Plots
Load the trained model from pricer.py and run full analysis.

Run pricer.py first to train and save the model, then run this file.
"""

import sys
import pathlib

_root = pathlib.Path(__file__).resolve()
while not (_root / "pyproject.toml").exists():
    _root = _root.parent
sys.path.insert(0, str(_root))
sys.path.insert(0, str(pathlib.Path(__file__).parent))  # for importing pricer

import numpy as np
import torch
from scipy.stats import norm
import matplotlib.pyplot as plt

from project.stock.generators import generate_gbm
from project.helpers.path_helpers import project_path
from pricer import (
    HedgingNet, run_paths, load_model,
    DEVICE, S0, K, sigma, r, N, T, h, KAPPA, N_PATHS_TEST
)


# ── BSM benchmarks ─────────────────────────────────────────────────────────────

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
    """
    BSM delta hedge P&L, optionally with transaction costs.
    BSM still rebalances to theoretical delta at every step — it does not
    adjust for costs. This shows the cost of ignoring frictions.

    Parameters
    ----------
    S     : (N+1, N_PATHS)
    kappa : proportional TC rate (0 = frictionless benchmark)
    """
    n_paths    = S.shape[1]
    currency   = np.zeros(n_paths)
    underlying = np.zeros(n_paths)
    prev_delta = np.zeros(n_paths)
    premium    = bsm_call(S0, K, r, sigma, T)

    for t in range(N):
        St         = S[t, :]
        delta      = bsm_delta(St, K, r, sigma, t * h, T)
        trade      = delta - prev_delta
        tc         = kappa * np.abs(trade) * St
        currency  -= trade * St + tc
        underlying += trade
        currency  *= np.exp(r * h)
        prev_delta = delta

    S_T    = S[N, :]
    payoff = np.maximum(S_T - K, 0)
    return premium + underlying * S_T + currency - payoff


# ── Delta computation helpers ─────────────────────────────────────────────────

def get_nn_deltas_single(S_path: np.ndarray, hedging_net: HedgingNet) -> np.ndarray:
    """NN delta at each of the N timesteps for one path. Returns (N,)."""
    hedging_net.eval()
    deltas = np.zeros(N)
    with torch.no_grad():
        for t in range(N):
            state = torch.tensor(
                [[S_path[t] / K, 1.0 - t / N]], dtype=torch.float32
            ).to(DEVICE)
            deltas[t] = hedging_net(state).item()
    return deltas


def get_nn_deltas_batch(S_test: np.ndarray, hedging_net: HedgingNet) -> np.ndarray:
    """
    NN deltas for all paths at all timesteps. Returns (N_PATHS, N).
    Vectorised across paths at each timestep for efficiency.
    """
    hedging_net.eval()
    n_paths    = S_test.shape[1]
    all_deltas = np.zeros((n_paths, N))

    with torch.no_grad():
        for t in range(N):
            St  = torch.tensor(S_test[t, :] / K, dtype=torch.float32).to(DEVICE)
            tau = torch.full((n_paths,), 1.0 - t / N, device=DEVICE)
            state = torch.stack([St, tau], dim=1)
            all_deltas[:, t] = hedging_net(state).cpu().numpy()

    return all_deltas


def get_bsm_deltas_batch(S_test: np.ndarray) -> np.ndarray:
    """BSM deltas for all paths at all timesteps. Returns (N_PATHS, N)."""
    n_paths    = S_test.shape[1]
    all_deltas = np.zeros((n_paths, N))
    for t in range(N):
        all_deltas[:, t] = bsm_delta(S_test[t, :], K, r, sigma, t * h, T)
    return all_deltas


def bsm_deltas_single(S_path: np.ndarray) -> np.ndarray:
    """BSM delta at each of the N timesteps for one path. Returns (N,)."""
    return np.array([bsm_delta(S_path[t], K, r, sigma, t * h, T) for t in range(N)])


def _pick_itm_otm(S_test: np.ndarray):
    """Return indices of the most extreme ITM and OTM paths in S_test."""
    S_T            = S_test[N, :]
    itm_candidates = np.where(S_T > K * 1.05)[0]
    otm_candidates = np.where(S_T < K * 0.95)[0]
    if len(itm_candidates) == 0 or len(otm_candidates) == 0:
        return None, None
    return (
        itm_candidates[np.argmax(S_T[itm_candidates])],
        otm_candidates[np.argmin(S_T[otm_candidates])],
    )


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_learning_curve() -> None:
    """Load saved epoch losses from pricer.py and plot the training curve."""
    epoch_losses = np.load(project_path(f"results/transaction costs/models/tc_epoch_losses_kappa{KAPPA}.npy")).tolist()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, len(epoch_losses) + 1), epoch_losses, linewidth=1.2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"Training Loss — Transaction Costs (κ={KAPPA})")
    plt.tight_layout()
    plt.savefig(project_path("results/transaction costs/figures/tc_learning_curve.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Learning curve saved.")


def plot_delta_paths(S_test: np.ndarray, hedging_net: HedgingNet) -> None:
    """Delta level over time: NN vs BSM on one ITM and one OTM path."""
    itm_idx, otm_idx = _pick_itm_otm(S_test)
    if itm_idx is None:
        print("Could not find suitable ITM/OTM paths — skipping delta path plot.")
        return

    times = np.linspace(0, T, N)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"Delta Level: NN (κ={KAPPA}) vs BSM (no TC)", fontsize=13)

    for ax, idx, label in zip(axes, [itm_idx, otm_idx], ["In-the-Money", "Out-of-the-Money"]):
        path       = S_test[:, idx]
        nn_deltas  = get_nn_deltas_single(path, hedging_net)
        bsm_deltas = bsm_deltas_single(path)

        ax.plot(times, nn_deltas,  label=f"NN (κ={KAPPA})", linewidth=1.2)
        ax.plot(times, bsm_deltas, label="BSM (no TC)",     linewidth=1.2, linestyle="--")
        ax.set_title(f"{label}  (S_T = {path[N]:.3f})")
        ax.set_xlabel("Time")
        ax.set_ylabel("Delta (shares held)")
        ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(project_path("results/transaction costs/figures/tc_delta_paths.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Delta path plot saved.")


def plot_trade_sizes(S_test: np.ndarray, hedging_net: HedgingNet) -> None:
    """
    Trade size |δ_t - δ_{t-1}| at each rebalancing step: NN vs BSM.
    This is direct evidence of whether the NN trades less aggressively.
    """
    itm_idx, otm_idx = _pick_itm_otm(S_test)
    if itm_idx is None:
        print("Could not find suitable ITM/OTM paths — skipping trade size plot.")
        return

    times = np.linspace(0, T, N)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"Trade Size |Δδ| per Step: NN (κ={KAPPA}) vs BSM (no TC)", fontsize=13)

    for ax, idx, label in zip(axes, [itm_idx, otm_idx], ["In-the-Money", "Out-of-the-Money"]):
        path       = S_test[:, idx]
        nn_trades  = np.abs(np.diff(get_nn_deltas_single(path, hedging_net),  prepend=0))
        bsm_trades = np.abs(np.diff(bsm_deltas_single(path),                  prepend=0))

        ax.plot(times, nn_trades,  label=f"NN (κ={KAPPA})", linewidth=1.0, alpha=0.8)
        ax.plot(times, bsm_trades, label="BSM (no TC)",     linewidth=1.0, linestyle="--", alpha=0.8)
        ax.set_title(f"{label}  (S_T = {path[N]:.3f})")
        ax.set_xlabel("Time")
        ax.set_ylabel("|Δδ| (trade size)")
        ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(project_path("results/transaction costs/figures/tc_trade_sizes.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Trade size plot saved.")


def plot_total_tc_comparison(S_test: np.ndarray, hedging_net: HedgingNet) -> None:
    """
    Total transaction cost paid per path across all test paths.
    NN (trained with TC) vs BSM (naive delta hedge subject to same TC).
    NN should pay less in aggregate by rebalancing less aggressively.
    """
    nn_deltas_all  = get_nn_deltas_batch(S_test, hedging_net)  # (N_PATHS, N)
    bsm_deltas_all = get_bsm_deltas_batch(S_test)              # (N_PATHS, N)

    # Trade sizes at each step; prepend 0 for initial position
    nn_trades  = np.abs(np.diff(nn_deltas_all,  prepend=0, axis=1))  # (N_PATHS, N)
    bsm_trades = np.abs(np.diff(bsm_deltas_all, prepend=0, axis=1))  # (N_PATHS, N)

    # Stock price at each rebalancing step: S_test[:N, :].T is (N_PATHS, N)
    S_steps = S_test[:N, :].T

    nn_total_tc  = (KAPPA * nn_trades  * S_steps).sum(axis=1)  # (N_PATHS,)
    bsm_total_tc = (KAPPA * bsm_trades * S_steps).sum(axis=1)  # (N_PATHS,)

    reduction = (1 - nn_total_tc.mean() / bsm_total_tc.mean()) * 100
    print(f"\n  Average total TC paid — NN:  {nn_total_tc.mean():.6f}")
    print(f"  Average total TC paid — BSM: {bsm_total_tc.mean():.6f}")
    print(f"  TC reduction:                {reduction:.1f}%\n")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(nn_total_tc,  bins=50, alpha=0.7, label=f"NN (κ={KAPPA})  mean={nn_total_tc.mean():.5f}")
    ax.hist(bsm_total_tc, bins=50, alpha=0.7, label=f"BSM             mean={bsm_total_tc.mean():.5f}")
    ax.axvline(nn_total_tc.mean(),  color="C0",    linestyle="--", linewidth=1.2)
    ax.axvline(bsm_total_tc.mean(), color="C1",    linestyle="--", linewidth=1.2)
    ax.set_xlabel("Total transaction cost paid per path")
    ax.set_ylabel("Count")
    ax.set_title(f"Total TC per Path (κ={KAPPA})")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(project_path("results/transaction costs/figures/tc_total_costs.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Total TC comparison plot saved.")


def test(S_test: np.ndarray, hedging_net: HedgingNet) -> None:
    """P&L statistics and distribution: NN vs BSM (with TC, no TC) vs Unhedged."""
    hedging_net.eval()
    S_tensor = torch.tensor(S_test.T, dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        pnl_np = run_paths(S_tensor, hedging_net).cpu().numpy()

    payoffs          = np.maximum(S_test[N, :] - K, 0)
    bsm_pnl_with_tc  = bsm_hedge_pnl(S_test, kappa=KAPPA)
    bsm_pnl_no_tc    = bsm_hedge_pnl(S_test, kappa=0.0)
    unhedged_pnl     = hedging_net.premium.item() - payoffs

    percentiles = [1, 5, 25, 75, 95, 99]
    nn_pcts     = np.percentile(pnl_np,           percentiles)
    bsm_tc_pcts = np.percentile(bsm_pnl_with_tc,  percentiles)
    bsm_pcts    = np.percentile(bsm_pnl_no_tc,    percentiles)

    print("=" * 75)
    print("  TEST RESULTS")
    print("=" * 75)
    print()
    print(f"  {'':30s}  {'NN (TC)':>10}  {'BSM (TC)':>10}  {'BSM (no TC)':>12}")
    print(f"  {'─'*66}")
    print(f"  {'Mean P&L':30s}  {pnl_np.mean():>10.4f}  {bsm_pnl_with_tc.mean():>10.4f}  {bsm_pnl_no_tc.mean():>12.4f}")
    print(f"  {'Std P&L':30s}  {pnl_np.std():>10.4f}  {bsm_pnl_with_tc.std():>10.4f}  {bsm_pnl_no_tc.std():>12.4f}")
    for p, nn_v, bsm_tc_v, bsm_v in zip(percentiles, nn_pcts, bsm_tc_pcts, bsm_pcts):
        print(f"  {f'P{p}':30s}  {nn_v:>10.4f}  {bsm_tc_v:>10.4f}  {bsm_v:>12.4f}")
    print()
    print(f"  {'BSM price':30s}  {bsm_call(S0, K, r, sigma, T):>10.4f}")
    print(f"  {'Learned premium':30s}  {hedging_net.premium.item():>10.4f}")
    print(f"  {'Transaction cost rate (κ)':30s}  {KAPPA:>10.4f}")
    print()
    print("=" * 75)

    fig, axes = plt.subplots(1, 4, figsize=(20, 4), sharey=True)
    fig.suptitle(f"Terminal P&L Distribution (κ={KAPPA})", fontsize=13)

    for ax, data, title in zip(
        axes,
        [pnl_np, bsm_pnl_with_tc, bsm_pnl_no_tc, unhedged_pnl],
        [f"NN Hedge (κ={KAPPA})", f"BSM Hedge (κ={KAPPA})", "BSM Hedge (no TC)", "Unhedged"],
    ):
        ax.hist(data, bins=50, edgecolor="none", alpha=0.8)
        ax.axvline(data.mean(), color="red",   linestyle="--", linewidth=1.2, label=f"Mean {data.mean():.4f}")
        ax.axvline(0,           color="black", linestyle=":",  linewidth=1.0, label="Zero")
        ax.set_title(title)
        ax.set_xlabel("P&L")
        ax.set_ylabel("Count")
        ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(project_path("results/transaction costs/figures/tc_pnl.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("P&L distribution plot saved.")


if __name__ == "__main__":
    print(f"── Evaluating transaction costs model (κ={KAPPA}) ──")

    print("\n── Loading model ──")
    hedging_net = load_model(kappa=KAPPA)

    print("\n── Simulating test paths ──")
    S_test = generate_gbm(S0, r, sigma, h, N_PATHS_TEST, N + 1)

    print("\n── Learning curve ──")
    plot_learning_curve()

    print("\n── P&L statistics and distribution ──")
    test(S_test, hedging_net)

    print("\n── Delta path comparison ──")
    plot_delta_paths(S_test, hedging_net)

    print("\n── Trade size comparison ──")
    plot_trade_sizes(S_test, hedging_net)

    print("\n── Total transaction cost comparison ──")
    plot_total_tc_comparison(S_test, hedging_net)

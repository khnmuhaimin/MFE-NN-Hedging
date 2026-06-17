"""
Deep Hedging - Option Pricing & Replication via Neural Networks
Based on Buehler et al. (2018) "Deep Hedging"
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import norm
import matplotlib.pyplot as plt
import os

from project.stock.generators import generate_gbm
from project.helpers.path_helpers import project_path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

os.makedirs(project_path("results/models"), exist_ok=True)


# parameters
S0 = 1
K = 1
sigma = 0.1
r = 0
N = 100
T = 1
h = T/N


# hyperparameters
N_PATHS_TRAIN  = 10_000
N_PATHS_TEST   = 10_000
BATCH_SIZE     = 512
N_EPOCHS       = 20
LEARNING_RATE  = 3e-3
CLIP_NORM = 1.0


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


class HedgingNet(nn.Module):
    def __init__(self, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid()
        )
        self.premium = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def loss_function(pnl):
    return (pnl**2).mean()


def run_paths(S_batch: torch.Tensor, hedging_net: HedgingNet) -> torch.Tensor:
    """
    Roll out the hedging strategy over all N timesteps.

    Parameters
    ----------
    S_batch : (batch, N+1)  stock prices for this batch of paths

    Returns
    -------
    pnl : (batch,)  terminal P&L including collected premium
    """
    batch = S_batch.shape[0]

    currency    = torch.zeros(batch, device=DEVICE)
    underlying  = torch.zeros(batch, device=DEVICE)
    prev_delta  = torch.zeros(batch, device=DEVICE)

    for t in range(N):
        St  = S_batch[:, t]
        tau = 1.0 - t / N

        state = torch.stack([
            St / K,
            torch.full((batch,), tau, device=DEVICE),
        ], dim=1)                           # (batch, 2)

        delta = hedging_net(state)          # (batch,)

        trade         = delta - prev_delta
        currency     -= trade * St
        underlying   += trade

        prev_delta = delta

    S_T     = S_batch[:, N]
    payoff  = torch.clamp(S_T - K, min=0)
    pnl     = hedging_net.premium + underlying * S_T + currency - payoff

    return pnl


def train(S_train: np.ndarray):
    """
    S_train : (N+1, N_PATHS)  — each column is one simulated path

    Returns
    -------
    hedging_net : trained HedgingNet
    epoch_losses : list of mean loss per epoch (for learning curve)
    """
    hedging_net = HedgingNet().to(DEVICE)

    optimizer = optim.Adam(hedging_net.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    S_tensor = torch.tensor(S_train.T, dtype=torch.float32)
    dataset  = TensorDataset(S_tensor)
    loader   = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    print(f"\n{'─'*63}")
    print(f"{'Epoch':>6}  {'Loss':>12}  {'Mean P&L':>12}  {'Std P&L':>10}  {'Premium':>10}")
    print(f"{'─'*63}")

    epoch_losses = []

    for epoch in range(1, N_EPOCHS + 1):
        hedging_net.train()
        batch_losses = []

        for (S_batch,) in loader:
            S_batch = S_batch.to(DEVICE)
            optimizer.zero_grad()
            pnl  = run_paths(S_batch, hedging_net)
            loss = loss_function(pnl)
            loss.backward()
            nn.utils.clip_grad_norm_(hedging_net.parameters(), CLIP_NORM)
            optimizer.step()
            batch_losses.append(loss.item())

        scheduler.step()
        epoch_losses.append(np.mean(batch_losses))

        if epoch % 10 == 0 or epoch == 1:
            hedging_net.eval()
            with torch.no_grad():
                pnl_all = run_paths(S_tensor[:2000].to(DEVICE), hedging_net)

            print(f"{epoch:>6}  {epoch_losses[-1]:>12.4f}  "
                  f"{pnl_all.mean().item():>12.4f}  {pnl_all.std().item():>10.4f}  "
                  f"{hedging_net.premium.item():>10.4f}")

    print(f"{'─'*63}\n")
    return hedging_net, epoch_losses


def plot_learning_curve(epoch_losses: list) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, len(epoch_losses) + 1), epoch_losses, linewidth=1.2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss (Learning Curve)")
    plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, "learning_curve.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Learning curve saved.")


def bsm_call(S0, K, r, sigma, T):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bsm_delta(S, K, r, sigma, t, T):
    """BSM delta (∂C/∂S) at time t given stock price S."""
    tau = T - t
    if tau <= 0:
        return np.where(S > K, 1.0, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    return norm.cdf(d1)


def bsm_hedge_pnl(S: np.ndarray) -> np.ndarray:
    """
    Roll out BSM delta hedging on paths S.

    Parameters
    ----------
    S : (N+1, N_PATHS)

    Returns
    -------
    pnl : (N_PATHS,)
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
        currency  -= trade * St
        underlying += trade
        currency  *= np.exp(r * h)
        prev_delta = delta

    S_T    = S[N, :]
    payoff = np.maximum(S_T - K, 0)
    pnl    = premium + underlying * S_T + currency - payoff
    return pnl


def get_nn_deltas(S_path: np.ndarray, hedging_net: HedgingNet) -> np.ndarray:
    """
    NN delta at each of the N timesteps for a single path.

    Parameters
    ----------
    S_path : (N+1,)  stock prices for one path

    Returns
    -------
    deltas : (N,)
    """
    hedging_net.eval()
    deltas = np.zeros(N)
    with torch.no_grad():
        for t in range(N):
            state = torch.tensor(
                [[S_path[t] / K, 1.0 - t / N]], dtype=torch.float32
            ).to(DEVICE)
            deltas[t] = hedging_net(state).item()
    return deltas


def bsm_deltas_path(S_path: np.ndarray) -> np.ndarray:
    """BSM delta at each of the N timesteps for a single path."""
    return np.array([bsm_delta(S_path[t], K, r, sigma, t * h, T) for t in range(N)])


def plot_delta_paths(S_test: np.ndarray, hedging_net: HedgingNet) -> None:
    """
    Plot NN vs BSM delta over time for one ITM and one OTM path.
    """
    S_T = S_test[N, :]

    itm_candidates = np.where(S_T > K * 1.05)[0]
    otm_candidates = np.where(S_T < K * 0.95)[0]

    if len(itm_candidates) == 0 or len(otm_candidates) == 0:
        print("Could not find suitable ITM/OTM paths — skipping delta path plot.")
        return

    # Pick the most extreme ITM and OTM paths
    itm_idx = itm_candidates[np.argmax(S_T[itm_candidates])]
    otm_idx = otm_candidates[np.argmin(S_T[otm_candidates])]

    times = np.linspace(0, T, N)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Delta Hedge Path: NN vs BSM", fontsize=13)

    for ax, idx, label in zip(axes, [itm_idx, otm_idx], ["In-the-Money", "Out-of-the-Money"]):
        path       = S_test[:, idx]
        nn_deltas  = get_nn_deltas(path, hedging_net)
        bsm_deltas = bsm_deltas_path(path)

        ax.plot(times, nn_deltas,  label="NN hedge",  linewidth=1.2)
        ax.plot(times, bsm_deltas, label="BSM hedge", linewidth=1.2, linestyle="--")
        ax.set_title(f"{label}  (S_T = {path[N]:.3f})")
        ax.set_xlabel("Time")
        ax.set_ylabel("Delta (shares held)")
        ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, "delta_paths.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Delta path plot saved.")


def test(S_test: np.ndarray, hedging_net: HedgingNet) -> None:
    """
    S_test : (N+1, N_PATHS)  — each column is one simulated path
    """
    hedging_net.eval()

    S_tensor = torch.tensor(S_test.T, dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        pnl    = run_paths(S_tensor, hedging_net)
        pnl_np = pnl.cpu().numpy()

    payoffs      = np.maximum(S_test[N, :] - K, 0)
    bsm_pnl      = bsm_hedge_pnl(S_test)
    unhedged_pnl = hedging_net.premium.item() - payoffs

    percentiles = [1, 5, 25, 75, 95, 99]
    nn_pcts     = np.percentile(pnl_np, percentiles)
    bsm_pcts    = np.percentile(bsm_pnl, percentiles)

    print("=" * 63)
    print("  TEST RESULTS")
    print("=" * 63)
    print()
    print(f"  {'':30s}  {'NN Hedge':>10}  {'BSM Hedge':>10}")
    print(f"  {'─'*52}")
    print(f"  {'Mean P&L':30s}  {pnl_np.mean():>10.4f}  {bsm_pnl.mean():>10.4f}")
    print(f"  {'Std P&L':30s}  {pnl_np.std():>10.4f}  {bsm_pnl.std():>10.4f}")
    for p, nn_v, bsm_v in zip(percentiles, nn_pcts, bsm_pcts):
        print(f"  {f'P{p}':30s}  {nn_v:>10.4f}  {bsm_v:>10.4f}")
    print()
    print(f"  {'BSM price':30s}  {bsm_call(S0, K, r, sigma, T):>10.4f}")
    print(f"  {'Learned premium':30s}  {hedging_net.premium.item():>10.4f}")
    print()
    print("=" * 63)

    # ── P&L distribution: NN vs BSM vs Unhedged ───────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 4), sharey=True)
    fig.suptitle("Terminal P&L Distribution", fontsize=13)

    for ax, data, title in zip(
        axes,
        [pnl_np, bsm_pnl, unhedged_pnl],
        ["Deep Hedging (learned)", "BSM Delta Hedge", "Unhedged"],
    ):
        ax.hist(data, bins=50, edgecolor="none", alpha=0.8)
        ax.axvline(data.mean(), color="red",   linestyle="--", linewidth=1.2, label=f"Mean {data.mean():.4f}")
        ax.axvline(0,           color="black", linestyle=":",  linewidth=1.0, label="Zero")
        ax.set_title(title)
        ax.set_xlabel("P&L")
        ax.set_ylabel("Count")
        ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, "minimal_pnl.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("P&L distribution plot saved.")


def save_model(hedging_net: HedgingNet) -> None:
    path = project_path("results/models/minimal_model.pt")
    torch.save({
        "hedging_net_state": hedging_net.state_dict(),
        "params": {
            "S0": S0, "K": K, "sigma": sigma,
            "r": r, "T": T, "N": N, "h": h
        }
    }, path)
    print(f"\nModel saved to '{path}'")


def load_model() -> HedgingNet:
    path = project_path("results/models/minimal_model.pt")
    checkpoint  = torch.load(path, map_location=DEVICE)
    hedging_net = HedgingNet().to(DEVICE)
    hedging_net.load_state_dict(checkpoint["hedging_net_state"])
    return hedging_net


if __name__ == "__main__":
    print("── Simulating training paths ──")
    S_train = generate_gbm(S0, r, sigma, h, N_PATHS_TRAIN, N+1)

    print("\n── Training ──")
    hedging_net, epoch_losses = train(S_train)

    print("\n── Plotting learning curve ──")
    plot_learning_curve(epoch_losses)

    print("\n── Simulating test paths ──")
    S_test = generate_gbm(S0, r, sigma, h, N_PATHS_TEST, N+1)

    print("── Testing ──")
    test(S_test, hedging_net)

    print("\n── Plotting delta paths ──")
    plot_delta_paths(S_test, hedging_net)

    save_model(hedging_net)
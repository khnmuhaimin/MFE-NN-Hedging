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

os.makedirs(project_path("results/figures"), exist_ok=True)
os.makedirs(project_path("results/models"), exist_ok=True)

S0 = 1
K = 1
sigma = 0.1
r = 0
N = 100
T = 1
h = T/N


N_PATHS_TRAIN  = 10_000
N_PATHS_TEST   = 10_000
BATCH_SIZE     = 2048
N_EPOCHS       = 100
LEARNING_RATE  = 1e-3


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


class HedgingNet(nn.Module):
    def __init__(self, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
            nn.Tanh()
        )
        self.premium = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        return self.net(x).squeeze(-1)



def loss_function(pnl):
    # penalise mean P&L being nonzero in either direction
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


def train(S_train: np.ndarray) -> HedgingNet:
    """
    S_train : (N+1, N_PATHS)  — each column is one simulated path
    """
    hedging_net = HedgingNet().to(DEVICE)

    optimizer = optim.Adam(hedging_net.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    # Transpose to (N_PATHS, N+1) so each row is one path
    S_tensor = torch.tensor(S_train.T, dtype=torch.float32)
    dataset  = TensorDataset(S_tensor)
    loader   = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    print(f"\n{'─'*63}")
    print(f"{'Epoch':>6}  {'Loss':>12}  {'Mean P&L':>12}  {'Std P&L':>10}  {'Premium':>10}")
    print(f"{'─'*63}")

    for epoch in range(1, N_EPOCHS + 1):
        hedging_net.train()
        epoch_losses = []

        for (S_batch,) in loader:
            S_batch = S_batch.to(DEVICE)
            optimizer.zero_grad()
            pnl  = run_paths(S_batch, hedging_net)
            loss = loss_function(pnl)
            loss.backward()
            nn.utils.clip_grad_norm_(hedging_net.parameters(), 1.0)
            optimizer.step()
            epoch_losses.append(loss.item())

        scheduler.step()

        if epoch % 10 == 0 or epoch == 1:
            hedging_net.eval()
            with torch.no_grad():
                pnl_all = run_paths(S_tensor[:2000].to(DEVICE), hedging_net)

            print(f"{epoch:>6}  {np.mean(epoch_losses):>12.4f}  "
                  f"{pnl_all.mean().item():>12.4f}  {pnl_all.std().item():>10.4f}  "
                  f"{hedging_net.premium.item():>10.4f}")

    print(f"{'─'*63}\n")
    return hedging_net


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


def test(S_test: np.ndarray, hedging_net: HedgingNet) -> None:
    """
    S_test : (N+1, N_PATHS)  — each column is one simulated path
    """
    hedging_net.eval()

    S_tensor = torch.tensor(S_test.T, dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        pnl    = run_paths(S_tensor, hedging_net)
        pnl_np = pnl.cpu().numpy()

    payoffs = np.maximum(S_test[N, :] - K, 0)
    bsm_pnl = bsm_hedge_pnl(S_test)

    print("=" * 55)
    print("  TEST RESULTS")
    print("=" * 55)
    print()
    print(f"  {'Mean terminal P&L':30s}  {pnl_np.mean():>8.4f}")
    print(f"  {'Std of P&L':30s}  {pnl_np.std():>8.4f}")
    print(f"  {'Mean payoff':30s}  {payoffs.mean():>8.4f}")
    print(f"  {'BSM price':30s}  {bsm_call(S0, K, r, sigma, T):>8.4f}")
    print(f"  {'Learned premium':30s}  {hedging_net.premium.item():>8.4f}")
    print()
    print("=" * 55)

    # ── Plot ──────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    fig.suptitle("Terminal P&L Distribution", fontsize=13)

    for ax, data, title in zip(
        axes,
        [pnl_np, bsm_pnl],
        ["Deep Hedging (learned)", "BSM Delta Hedge"],
    ):
        ax.hist(data, bins=50, edgecolor="none", alpha=0.8)
        ax.axvline(data.mean(), color="red",    linestyle="--", linewidth=1.2, label=f"Mean {data.mean():.4f}")
        ax.axvline(0,           color="black",  linestyle=":",  linewidth=1.0, label="Zero")
        ax.set_title(title)
        ax.set_xlabel("P&L")
        ax.set_ylabel("Count")
        ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(project_path("results/figures/minimal_pnl.png"), dpi=150, bbox_inches="tight")


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
    hedging_net = train(S_train)

    print("\n── Simulating test paths ──")
    S_test = generate_gbm(S0, r, sigma, h, N_PATHS_TEST, N+1)

    print("── Testing ──")
    test(S_test, hedging_net)

    save_model(hedging_net)


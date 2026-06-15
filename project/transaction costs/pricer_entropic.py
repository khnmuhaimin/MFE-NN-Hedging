"""
Deep Hedging with Transaction Costs — Entropic Risk Measure Training

The entropic risk measure (ERM) is the certainty equivalent under CARA
(Constant Absolute Risk Aversion) exponential utility:

    ER_γ(pnl) = (1/γ) * log( E[ exp(-γ * pnl) ] )

Unlike CVaR which applies a binary weight (1 to worst α%, 0 elsewhere), ERM
weights every path exponentially by how bad it is — a smoother, continuous
version of tail-risk optimisation. γ is the risk aversion parameter:
  γ → 0  : reduces to -E[pnl]  (risk-neutral)
  γ → ∞  : approaches the worst-case outcome (minimax)

Premium anchoring: d(ER)/dπ = -1 always (same structural issue as CVaR).
Fixed with the same gradient routing: CVaR gradient is blocked from the
premium, and a zero-mean penalty anchors the premium independently.

Run this file to train a single ERM model.
Run EntropicTrain.py for multi-kappa training.
Run CompareEvaluate.py to compare MSE / CVaR / Entropic side by side.
"""

import sys
import pathlib

_root = pathlib.Path(__file__).resolve()
while not (_root / "pyproject.toml").exists():
    _root = _root.parent
sys.path.insert(0, str(_root))

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from project.stock.generators import generate_gbm
from project.helpers.path_helpers import project_path

os.makedirs(project_path("results/transaction costs/figures"), exist_ok=True)
os.makedirs(project_path("results/transaction costs/models"), exist_ok=True)

# ── Model parameters ──────────────────────────────────────────────────────────
S0    = 1
K     = 1
sigma = 0.1
r     = 0
N     = 100
T     = 1
h     = T / N

KAPPA       = 0.001   # proportional transaction cost rate
GAMMA       = 100.0   # risk aversion; γ·σ(pnl) ≈ 0.6 → meaningful tail weighting
LAMBDA_MEAN = 20.0    # weight on zero-mean penalty

# ── Training hyperparameters ───────────────────────────────────────────────────
N_PATHS_TRAIN = 10_000
N_PATHS_TEST  = 10_000
BATCH_SIZE    = 2048
N_EPOCHS      = 100
LEARNING_RATE = 1e-3

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
            nn.Sigmoid()
        )
        self.premium = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def entropic_loss(pnl: torch.Tensor, premium: nn.Parameter,
                  gamma: float = GAMMA,
                  lambda_mean: float = LAMBDA_MEAN) -> torch.Tensor:
    """
    Entropic risk measure with gradient-routed premium anchoring.

    The entropic risk measure is linear in the premium (d(ER)/dπ = -1),
    so without intervention the gradient pushes π → +∞, same as CVaR.
    Fix: detach premium from the ER computation; anchor it via E[pnl]² penalty.

    Numerical stability: log-mean-exp trick subtracts the batch maximum
    before exponentiating, preventing overflow for large γ.

    Parameters
    ----------
    pnl         : (batch,)  terminal P&L
    premium     : learnable premium parameter (needed to detach from ER)
    gamma       : risk aversion (γ·σ(pnl) ≈ 0.6 for γ=100, σ≈0.006)
    lambda_mean : weight on zero-mean penalty
    """
    # ER gradient blocked from premium — only shapes the delta strategy
    pnl_for_er  = pnl - premium + premium.detach()
    neg_scaled  = -gamma * pnl_for_er
    max_val     = neg_scaled.detach().max()            # subtract for stability
    er          = (1.0 / gamma) * (max_val + torch.log(
                      torch.mean(torch.exp(neg_scaled - max_val))))

    # Zero-mean penalty anchors the premium to fair value
    mean_penalty = lambda_mean * pnl.mean() ** 2

    return er + mean_penalty


def run_paths(S_batch: torch.Tensor, hedging_net: HedgingNet,
              kappa: float = KAPPA) -> torch.Tensor:
    batch      = S_batch.shape[0]
    currency   = torch.zeros(batch, device=DEVICE)
    underlying = torch.zeros(batch, device=DEVICE)
    prev_delta = torch.zeros(batch, device=DEVICE)

    for t in range(N):
        St    = S_batch[:, t]
        tau   = 1.0 - t / N
        state = torch.stack([
            St / K,
            torch.full((batch,), tau, device=DEVICE),
        ], dim=1)
        delta      = hedging_net(state)
        trade      = delta - prev_delta
        tc         = kappa * trade.abs() * St
        currency  -= trade * St + tc
        underlying += trade
        prev_delta = delta

    S_T    = S_batch[:, N]
    payoff = torch.clamp(S_T - K, min=0)
    return hedging_net.premium + underlying * S_T + currency - payoff


def train(S_train: np.ndarray, kappa: float = KAPPA,
          gamma: float = GAMMA, lambda_mean: float = LAMBDA_MEAN):
    hedging_net = HedgingNet().to(DEVICE)
    optimizer   = optim.Adam(hedging_net.parameters(), lr=LEARNING_RATE)
    scheduler   = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    S_tensor = torch.tensor(S_train.T, dtype=torch.float32)
    dataset  = TensorDataset(S_tensor)
    loader   = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    print(f"\n{'─'*63}")
    print(f"{'Epoch':>6}  {'ER Loss':>12}  {'Mean P&L':>12}  {'Std P&L':>10}  {'Premium':>10}")
    print(f"{'─'*63}")

    epoch_losses = []

    for epoch in range(1, N_EPOCHS + 1):
        hedging_net.train()
        batch_losses = []

        for (S_batch,) in loader:
            S_batch = S_batch.to(DEVICE)
            optimizer.zero_grad()
            pnl  = run_paths(S_batch, hedging_net, kappa=kappa)
            loss = entropic_loss(pnl, hedging_net.premium,
                                 gamma=gamma, lambda_mean=lambda_mean)
            loss.backward()
            nn.utils.clip_grad_norm_(hedging_net.parameters(), 1.0)
            optimizer.step()
            batch_losses.append(loss.item())

        scheduler.step()
        epoch_losses.append(np.mean(batch_losses))

        if epoch % 10 == 0 or epoch == 1:
            hedging_net.eval()
            with torch.no_grad():
                pnl_all = run_paths(S_tensor[:2000].to(DEVICE), hedging_net, kappa=kappa)
            print(f"{epoch:>6}  {epoch_losses[-1]:>12.6f}  "
                  f"{pnl_all.mean().item():>12.4f}  {pnl_all.std().item():>10.4f}  "
                  f"{hedging_net.premium.item():>10.4f}")

    print(f"{'─'*63}\n")
    return hedging_net, epoch_losses


def save_model(hedging_net: HedgingNet, epoch_losses: list,
               kappa: float = KAPPA, gamma: float = GAMMA) -> None:
    model_path  = project_path(f"results/transaction costs/models/tc_model_kappa{kappa}_entropic.pt")
    losses_path = project_path(f"results/transaction costs/models/tc_epoch_losses_kappa{kappa}_entropic.npy")
    torch.save({
        "hedging_net_state": hedging_net.state_dict(),
        "params": {"S0": S0, "K": K, "sigma": sigma,
                   "r": r, "T": T, "N": N, "h": h, "kappa": kappa, "gamma": gamma}
    }, model_path)
    np.save(losses_path, np.array(epoch_losses))
    print(f"Model saved to '{model_path}'")


def load_model(kappa: float = KAPPA) -> HedgingNet:
    model_path  = project_path(f"results/transaction costs/models/tc_model_kappa{kappa}_entropic.pt")
    checkpoint  = torch.load(model_path, map_location=DEVICE)
    hedging_net = HedgingNet().to(DEVICE)
    hedging_net.load_state_dict(checkpoint["hedging_net_state"])
    return hedging_net


if __name__ == "__main__":
    print(f"── Entropic risk measure (κ={KAPPA}, γ={GAMMA}) ──")

    print("\n── Simulating training paths ──")
    S_train = generate_gbm(S0, r, sigma, h, N_PATHS_TRAIN, N + 1)

    print("\n── Training ──")
    hedging_net, epoch_losses = train(S_train, kappa=KAPPA, gamma=GAMMA)

    save_model(hedging_net, epoch_losses, kappa=KAPPA, gamma=GAMMA)

"""
Deep Hedging with Transaction Costs — CVaR Loss Training
Identical to pricer.py except the loss function is Conditional Value-at-Risk
(CVaR) instead of mean-squared P&L. CVaR optimises the expected loss in the
worst (1-alpha) fraction of outcomes, producing a hedger that is more
protective of the left tail of the P&L distribution.

CVaR is computed via the Rockafellar-Uryasev (2000) formula:
    CVaR_alpha(L) = z + 1/(1-alpha) * E[max(L - z, 0)]
where L = -P&L (losses) and z is the alpha-quantile of L estimated per batch.

Run this file to train and save a CVaR model.
Run CVaREvaluate.py to compare CVaR vs MSE models.
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
ALPHA       = 0.95    # CVaR confidence level (optimises worst 5% of outcomes)
LAMBDA_MEAN = 20.0    # weight on zero-mean penalty (see cvar_loss docstring)

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
            nn.Sigmoid()    # constrains delta to (0, 1), correct for a European call
        )
        self.premium = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def cvar_loss(pnl: torch.Tensor, premium: torch.nn.Parameter,
              alpha: float = ALPHA, lambda_mean: float = LAMBDA_MEAN) -> torch.Tensor:
    """
    CVaR loss with gradient-routed premium anchoring.

    Root cause of the naive CVaR problem
    ─────────────────────────────────────
    d(CVaR) / d(π) = -1 always (derivable from the Rockafellar-Uryasev formula
    with detached VaR estimate). Gradient descent therefore pushes π → +∞
    regardless of where the P&L distribution sits, making the comparison with
    MSE meaningless (CVaR just charges a higher premium, not a better hedge).

    Fix: gradient routing
    ─────────────────────
    Detach the premium from the CVaR computation so the CVaR gradient cannot
    reach the premium. CVaR only shapes the *delta* strategy. A separate
    zero-mean penalty (λ * E[pnl]²) anchors the premium; its gradient IS
    allowed through the premium but NOT through the delta parameters (since at
    equilibrium E[pnl] ≈ 0 → penalty gradient ≈ 0 for delta anyway).

    Result: delta is optimised purely by CVaR; premium converges to the
    zero-mean fair price for any λ > 0 (λ only affects convergence speed).

    Parameters
    ----------
    pnl         : (batch,)  terminal P&L (includes learnable premium)
    premium     : the nn.Parameter for the premium (needed to detach it)
    alpha       : CVaR confidence level (0.95 → worst 5%)
    lambda_mean : weight on zero-mean penalty (controls premium convergence speed)
    """
    # CVaR computed with premium detached — gradient cannot push premium up
    pnl_for_cvar = pnl - premium + premium.detach()
    losses = -pnl_for_cvar
    z      = torch.quantile(losses.detach(), alpha)   # VaR, no gradient
    cvar   = z + (1.0 / (1.0 - alpha)) * torch.mean(torch.relu(losses - z))

    # Zero-mean penalty acts on the full pnl (premium has gradient here)
    mean_penalty = lambda_mean * pnl.mean() ** 2

    return cvar + mean_penalty


def run_paths(S_batch: torch.Tensor, hedging_net: HedgingNet, kappa: float = KAPPA) -> torch.Tensor:
    """
    Roll out the hedging strategy with proportional transaction costs.

    Parameters
    ----------
    S_batch : (batch, N+1)
    kappa   : proportional transaction cost rate

    Returns
    -------
    pnl : (batch,)  terminal P&L net of all costs
    """
    batch = S_batch.shape[0]

    currency   = torch.zeros(batch, device=DEVICE)
    underlying = torch.zeros(batch, device=DEVICE)
    prev_delta = torch.zeros(batch, device=DEVICE)

    for t in range(N):
        St  = S_batch[:, t]
        tau = 1.0 - t / N

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
    pnl    = hedging_net.premium + underlying * S_T + currency - payoff

    return pnl


def train(S_train: np.ndarray, kappa: float = KAPPA, alpha: float = ALPHA,
          lambda_mean: float = LAMBDA_MEAN):
    """
    Parameters
    ----------
    S_train     : (N+1, N_PATHS)
    kappa       : proportional transaction cost rate
    alpha       : CVaR confidence level
    lambda_mean : weight on zero-mean P&L penalty (see cvar_loss)

    Returns
    -------
    hedging_net  : trained HedgingNet
    epoch_losses : list[float]
    """
    hedging_net = HedgingNet().to(DEVICE)
    optimizer   = optim.Adam(hedging_net.parameters(), lr=LEARNING_RATE)
    scheduler   = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    S_tensor = torch.tensor(S_train.T, dtype=torch.float32)
    dataset  = TensorDataset(S_tensor)
    loader   = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    print(f"\n{'─'*63}")
    print(f"{'Epoch':>6}  {'CVaR Loss':>12}  {'Mean P&L':>12}  {'Std P&L':>10}  {'Premium':>10}")
    print(f"{'─'*63}")

    epoch_losses = []

    for epoch in range(1, N_EPOCHS + 1):
        hedging_net.train()
        batch_losses = []

        for (S_batch,) in loader:
            S_batch = S_batch.to(DEVICE)
            optimizer.zero_grad()
            pnl  = run_paths(S_batch, hedging_net, kappa=kappa)
            loss = cvar_loss(pnl, hedging_net.premium, alpha=alpha, lambda_mean=lambda_mean)
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


def save_model(hedging_net: HedgingNet, epoch_losses: list, kappa: float = KAPPA, alpha: float = ALPHA) -> None:
    model_path  = project_path(f"results/transaction costs/models/tc_model_kappa{kappa}_cvar.pt")
    losses_path = project_path(f"results/transaction costs/models/tc_epoch_losses_kappa{kappa}_cvar.npy")
    torch.save({
        "hedging_net_state": hedging_net.state_dict(),
        "params": {
            "S0": S0, "K": K, "sigma": sigma,
            "r": r, "T": T, "N": N, "h": h, "kappa": kappa, "alpha": alpha
        }
    }, model_path)
    np.save(losses_path, np.array(epoch_losses))
    print(f"Model saved to '{model_path}'")


def load_model(kappa: float = KAPPA) -> HedgingNet:
    model_path  = project_path(f"results/transaction costs/models/tc_model_kappa{kappa}_cvar.pt")
    checkpoint  = torch.load(model_path, map_location=DEVICE)
    hedging_net = HedgingNet().to(DEVICE)
    hedging_net.load_state_dict(checkpoint["hedging_net_state"])
    return hedging_net


if __name__ == "__main__":
    print(f"── CVaR transaction costs (κ={KAPPA}, α={ALPHA}) ──")

    print("\n── Simulating training paths ──")
    S_train = generate_gbm(S0, r, sigma, h, N_PATHS_TRAIN, N + 1)

    print("\n── Training ──")
    hedging_net, epoch_losses = train(S_train, kappa=KAPPA, alpha=ALPHA)

    save_model(hedging_net, epoch_losses, kappa=KAPPA, alpha=ALPHA)

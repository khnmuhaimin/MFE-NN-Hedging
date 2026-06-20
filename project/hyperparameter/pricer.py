"""Quick check that the tuned base network (moneyness, tau) hedges as well as the BS delta.
Trains and tests on independent GBM draws. Flip REBALANCE to switch monthly <-> daily."""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import norm
import matplotlib.pyplot as plt

from project.stock.generators import generate_gbm
from project.helpers.path_helpers import project_path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- edit here ----
REBALANCE = "monthly"          # "monthly" or "daily"

# tuned winners per frequency (base feature set)
HP = {
    "monthly": dict(N=12,  hidden=64,  depth=4, lr=0.03, batch_size=512, clip_norm=1.0),
    "daily":   dict(N=252, hidden=128, depth=4, lr=0.03, batch_size=512, clip_norm=0.5),
}[REBALANCE]

EPOCHS = 50
STEP_SIZE, GAMMA = 15, 0.5
N_TRAIN, N_TEST = 10_000, 10_000
# -------------------

S0, K, sigma, r, T = 1, 1, 0.1, 0, 1
N = HP["N"]
h = T / N
TAG = f"base_{REBALANCE}"
print(f"Using device: {DEVICE}   version: {TAG}  (N={N})")


class HedgingNet(nn.Module):
    """Feed-forward delta net; sigmoid output keeps a call delta in [0, 1]."""
    def __init__(self, input_dim=2, hidden=128, depth=4):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden), nn.ReLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.ReLU()]
        layers += [nn.Linear(hidden, 1), nn.Sigmoid()]
        self.net = nn.Sequential(*layers)
        self.premium = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def build_state(St, t, batch):
    """Two inputs: moneyness and time-to-maturity."""
    tau = 1.0 - t / N
    return torch.stack([St / K, torch.full((batch,), tau, device=DEVICE)], dim=1)


def run_paths(S_batch, net):
    """Roll the hedge forward over all N steps and return terminal P&L per path."""
    batch = S_batch.shape[0]
    currency = torch.zeros(batch, device=DEVICE)
    underlying = torch.zeros(batch, device=DEVICE)
    prev_delta = torch.zeros(batch, device=DEVICE)

    for t in range(N):
        St = S_batch[:, t]
        delta = net(build_state(St, t, batch))
        trade = delta - prev_delta
        currency -= trade * St
        underlying += trade
        prev_delta = delta

    S_T = S_batch[:, N]
    payoff = torch.clamp(S_T - K, min=0)
    return net.premium + underlying * S_T + currency - payoff


def bsm_call(S0, K, r, sigma, T):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bsm_delta(S, t):
    tau = T - t
    if tau <= 0:
        return np.where(S > K, 1.0, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * tau) / (sigma * np.sqrt(tau))
    return norm.cdf(d1)


def bsm_hedge_pnl(S):
    """Discrete BS-delta hedge P&L: the benchmark floor."""
    n_paths = S.shape[1]
    currency = np.zeros(n_paths)
    underlying = np.zeros(n_paths)
    prev_delta = np.zeros(n_paths)
    premium = bsm_call(S0, K, r, sigma, T)

    for t in range(N):
        St = S[t, :]
        delta = bsm_delta(St, t * h)
        trade = delta - prev_delta
        currency -= trade * St
        underlying += trade
        currency *= np.exp(r * h)
        prev_delta = delta

    S_T = S[N, :]
    return premium + underlying * S_T + currency - np.maximum(S_T - K, 0)


def train(S_train):
    """Train one network on the given paths and return it plus the loss curve."""
    net = HedgingNet(2, HP["hidden"], HP["depth"]).to(DEVICE)
    opt = optim.Adam(net.parameters(), lr=HP["lr"])
    sched = optim.lr_scheduler.StepLR(opt, step_size=STEP_SIZE, gamma=GAMMA)

    S_tensor = torch.tensor(S_train.T, dtype=torch.float32)
    loader = DataLoader(TensorDataset(S_tensor), batch_size=HP["batch_size"], shuffle=True)

    print(f"\n{'epoch':>6}  {'loss':>12}  {'mean pnl':>10}  {'std pnl':>10}  {'premium':>9}")
    losses = []
    for epoch in range(1, EPOCHS + 1):
        net.train()
        batch_losses = []
        for (S_batch,) in loader:
            S_batch = S_batch.to(DEVICE)
            opt.zero_grad()
            loss = (run_paths(S_batch, net) ** 2).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), HP["clip_norm"])
            opt.step()
            batch_losses.append(loss.item())
        sched.step()
        losses.append(np.mean(batch_losses))

        if epoch == 1 or epoch % 10 == 0:
            net.eval()
            with torch.no_grad():
                pnl = run_paths(S_tensor[:2000].to(DEVICE), net)
            print(f"{epoch:>6}  {losses[-1]:>12.4f}  {pnl.mean().item():>10.4f}  "
                  f"{pnl.std().item():>10.4f}  {net.premium.item():>9.4f}")
    return net, losses


def evaluate(S_test, net):
    """Compare NN and BS hedges on held-out paths and save the usual plots."""
    net.eval()
    with torch.no_grad():
        nn_pnl = run_paths(torch.tensor(S_test.T, dtype=torch.float32).to(DEVICE), net).cpu().numpy()
    bsm_pnl = bsm_hedge_pnl(S_test)
    unhedged = net.premium.item() - np.maximum(S_test[N, :] - K, 0)

    pcts = [1, 5, 25, 75, 95, 99]
    print("\n" + "=" * 52)
    print(f"  {'':18s}{'NN':>12}{'BSM':>12}")
    print("  " + "-" * 42)
    print(f"  {'mean pnl':18s}{nn_pnl.mean():>12.4f}{bsm_pnl.mean():>12.4f}")
    print(f"  {'std pnl':18s}{nn_pnl.std():>12.4f}{bsm_pnl.std():>12.4f}")
    for p, a, b in zip(pcts, np.percentile(nn_pnl, pcts), np.percentile(bsm_pnl, pcts)):
        print(f"  {'P' + str(p):18s}{a:>12.4f}{b:>12.4f}")
    print(f"  {'bsm price':18s}{bsm_call(S0, K, r, sigma, T):>12.4f}")
    print(f"  {'learned premium':18s}{net.premium.item():>12.4f}")
    gap = nn_pnl.std() - bsm_pnl.std()
    print("  " + "-" * 42)
    print(f"  std gap vs BSM: {gap:+.4f}   "
          f"({'matches the floor' if abs(gap) < 5e-4 else 'above the floor'})")
    print("=" * 52)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4), sharey=True)
    fig.suptitle(f"Terminal P&L  ({TAG})", fontsize=13)
    for ax, data, title in zip(axes, [nn_pnl, bsm_pnl, unhedged],
                               ["Deep hedge", "BS delta hedge", "Unhedged"]):
        ax.hist(data, bins=50, alpha=0.8)
        ax.axvline(data.mean(), color="red", linestyle="--", linewidth=1.1, label=f"mean {data.mean():.4f}")
        ax.axvline(0, color="black", linestyle=":", linewidth=1.0)
        ax.set_title(title)
        ax.set_xlabel("P&L")
        ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, f"pnl_{TAG}.png"), dpi=150, bbox_inches="tight")
    plt.close()

    plot_delta_paths(S_test, net)
    print(f"Plots saved with tag '{TAG}'.")


def nn_deltas_path(S_path, net):
    net.eval()
    deltas = np.zeros(N)
    with torch.no_grad():
        for t in range(N):
            St = torch.tensor([S_path[t]], dtype=torch.float32, device=DEVICE)
            deltas[t] = net(build_state(St, t, 1)).item()
    return deltas


def plot_delta_paths(S_test, net):
    """NN vs BS delta over time for one ITM and one OTM path."""
    S_T = S_test[N, :]
    itm = np.where(S_T > K * 1.05)[0]
    otm = np.where(S_T < K * 0.95)[0]
    if len(itm) == 0 or len(otm) == 0:
        return
    idx = [itm[np.argmax(S_T[itm])], otm[np.argmin(S_T[otm])]]
    times = np.linspace(0, T, N)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"Delta path: NN vs BS  ({TAG})", fontsize=13)
    for ax, i, label in zip(axes, idx, ["in-the-money", "out-of-the-money"]):
        path = S_test[:, i]
        ax.plot(times, nn_deltas_path(path, net), label="NN", linewidth=1.2)
        ax.plot(times, [bsm_delta(path[t], t * h) for t in range(N)],
                label="BS", linewidth=1.2, linestyle="--")
        ax.set_title(f"{label}  (S_T={path[N]:.3f})")
        ax.set_xlabel("time")
        ax.set_ylabel("delta")
        ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, f"delta_{TAG}.png"), dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    S_train = generate_gbm(S0, r, sigma, h, N_TRAIN, N + 1)
    net, _ = train(S_train)
    S_test = generate_gbm(S0, r, sigma, h, N_TEST, N + 1)
    evaluate(S_test, net)
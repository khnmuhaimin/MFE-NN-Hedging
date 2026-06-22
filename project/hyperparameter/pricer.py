"""Base pricer (moneyness, tau): trains and tests with the methodology's parameter ranges.

Initial moneyness is sampled from [0.85, 1.15] and volatility from [0.1, 0.3], with the
strike fixed at K=1 and the risk-free rate at r=0. GBM is simulated at daily resolution
in every case; for the monthly variant, the hedge rebalances at every 21st daily step.
Flip REBALANCE to switch monthly <-> daily.

The BSM benchmark uses the true per-path sigma and is therefore an oracle floor: the
base network cannot match it in general because it never sees sigma. The relvol variant
of this script is the one expected to close that gap.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import norm
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, "models")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- edit here ----
REBALANCE = "daily"          # "monthly" or "daily"

HP = {
    "monthly": dict(N=12,  hidden=64,  depth=4, lr=0.03, batch_size=512, clip_norm=1.0),
    "daily":   dict(N=252, hidden=128, depth=4, lr=0.03, batch_size=512, clip_norm=0.5),
}[REBALANCE]

EPOCHS = 50
STEP_SIZE, GAMMA = 30, 0.5
N_TRAIN, N_TEST = 10_000, 10_000
# -------------------

# market parameters (fixed by the methodology)
K, r, T = 1.0, 0.0, 1.0
MONEYNESS_RANGE = (0.85, 1.15)
SIGMA_RANGE = (0.1, 0.3)

# always simulate at daily resolution
DAILY_STEPS = 252
H_DAILY = T / DAILY_STEPS

# rebalancing dates expressed as daily indices
N = HP["N"]
REBAL_STRIDE = DAILY_STEPS // N
REBAL_INDICES = list(range(0, DAILY_STEPS, REBAL_STRIDE))
REBAL_TAUS = [1.0 - i / DAILY_STEPS for i in REBAL_INDICES]
assert len(REBAL_INDICES) == N

TAG = f"base_{REBALANCE}"
print(f"Using device: {DEVICE}   version: {TAG}  (N={N}, daily_steps={DAILY_STEPS})")


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


def build_state(S_hist, daily_idx, tau):
    """Two inputs: moneyness (S_t/K) and time-to-maturity. daily_idx is unused here
    but kept in the signature so the relvol variant stays drop-in compatible."""
    batch = S_hist.shape[0]
    St = S_hist[:, -1]
    return torch.stack([St / K, torch.full((batch,), tau, device=DEVICE)], dim=1)


def run_paths(S_batch, net):
    """Roll the hedge forward, rebalancing only at REBAL_INDICES.

    S_batch has shape (batch, DAILY_STEPS+1). The N trades happen at REBAL_INDICES;
    P&L accrues from each holding through to the next rebalancing date or maturity.
    """
    batch = S_batch.shape[0]
    currency = torch.zeros(batch, device=DEVICE)
    underlying = torch.zeros(batch, device=DEVICE)
    prev_delta = torch.zeros(batch, device=DEVICE)

    for i, daily_idx in enumerate(REBAL_INDICES):
        S_hist = S_batch[:, :daily_idx + 1]    # full daily history up to this rebal date
        St = S_hist[:, -1]
        delta = net(build_state(S_hist, daily_idx, REBAL_TAUS[i]))
        trade = delta - prev_delta
        currency -= trade * St
        underlying += trade
        prev_delta = delta

    S_T = S_batch[:, DAILY_STEPS]
    payoff = torch.clamp(S_T - K, min=0)
    return net.premium + underlying * S_T + currency - payoff


def generate_paths(n_paths, seed=None):
    """Generate n_paths daily GBM trajectories with random initial moneyness and sigma.

    Returns:
        S: (DAILY_STEPS+1, n_paths) array of daily prices
        sigmas: (n_paths,) true volatilities used per path
        moneynesses: (n_paths,) initial moneynesses S_0/K (= S_0 since K=1)
    """
    rng = np.random.default_rng(seed)
    moneynesses = rng.uniform(*MONEYNESS_RANGE, n_paths)
    sigmas = rng.uniform(*SIGMA_RANGE, n_paths)

    Z = rng.standard_normal((DAILY_STEPS, n_paths))
    W = np.cumsum(np.sqrt(H_DAILY) * Z, axis=0)                  # (DAILY_STEPS, n_paths)
    t_grid = np.arange(1, DAILY_STEPS + 1)[:, None] * H_DAILY     # (DAILY_STEPS, 1)

    # log S_t = log S_0 - sigma^2 t / 2 + sigma W_t   (r = 0)
    log_S = (np.log(moneynesses)[None, :]
             - 0.5 * (sigmas[None, :] ** 2) * t_grid
             + sigmas[None, :] * W)
    S = np.exp(log_S)
    S_full = np.concatenate([moneynesses[None, :], S], axis=0)    # prepend S_0 row
    return S_full, sigmas, moneynesses


def bsm_call_vec(S0_vec, K, r, sigma_vec, T):
    """BSM call price, vectorised over S0 and sigma."""
    d1 = (np.log(S0_vec / K) + (r + 0.5 * sigma_vec ** 2) * T) / (sigma_vec * np.sqrt(T))
    d2 = d1 - sigma_vec * np.sqrt(T)
    return S0_vec * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bsm_delta_vec(S, sigma_vec, t):
    """BSM call delta, vectorised over S and sigma."""
    tau = T - t
    if tau <= 0:
        return np.where(S > K, 1.0, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma_vec ** 2) * tau) / (sigma_vec * np.sqrt(tau))
    return norm.cdf(d1)


def bsm_hedge_pnl(S, sigmas, moneynesses):
    """Oracle BSM hedge: per-path BSM premium and deltas computed with each path's TRUE sigma."""
    n_paths = S.shape[1]
    currency = np.zeros(n_paths)
    underlying = np.zeros(n_paths)
    prev_delta = np.zeros(n_paths)
    premiums = bsm_call_vec(moneynesses, K, r, sigmas, T)

    for daily_idx in REBAL_INDICES:
        St = S[daily_idx, :]
        delta = bsm_delta_vec(St, sigmas, daily_idx * H_DAILY)
        trade = delta - prev_delta
        currency -= trade * St
        underlying += trade
        # currency accrual is a no-op while r = 0
        prev_delta = delta

    S_T = S[DAILY_STEPS, :]
    return premiums + underlying * S_T + currency - np.maximum(S_T - K, 0)


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


def save_model(net, losses):
    """Save the trained network + enough metadata to rebuild it later."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    path = os.path.join(MODELS_DIR, f"{TAG}.pt")
    torch.save({
        "tag": TAG,
        "feature_set": "base",
        "rebalance": REBALANCE,
        "input_dim": 2,
        "N": N,
        "daily_steps": DAILY_STEPS,
        "h_daily": H_DAILY,
        "rebal_indices": REBAL_INDICES,
        "K": K, "r": r, "T": T,
        "moneyness_range": MONEYNESS_RANGE,
        "sigma_range": SIGMA_RANGE,
        "hyperparameters": HP,
        "epochs": EPOCHS,
        "step_size": STEP_SIZE,
        "gamma": GAMMA,
        "n_train": N_TRAIN,
        "premium": net.premium.item(),
        "final_train_loss": float(losses[-1]) if losses else None,
        "state_dict": net.state_dict(),
    }, path)
    print(f"Saved model to {path}")


def evaluate(S_test, sigmas, moneynesses, net):
    """Compare NN and oracle-BSM hedges on held-out paths and save the usual plots."""
    net.eval()
    with torch.no_grad():
        nn_pnl = run_paths(torch.tensor(S_test.T, dtype=torch.float32).to(DEVICE), net).cpu().numpy()
    bsm_pnl = bsm_hedge_pnl(S_test, sigmas, moneynesses)
    mean_bsm_price = bsm_call_vec(moneynesses, K, r, sigmas, T).mean()
    unhedged = net.premium.item() - np.maximum(S_test[DAILY_STEPS, :] - K, 0)

    pcts = [1, 5, 25, 75, 95, 99]
    print("\n" + "=" * 56)
    print(f"  {'':18s}{'NN':>12}{'BSM (oracle)':>14}")
    print("  " + "-" * 44)
    print(f"  {'mean pnl':18s}{nn_pnl.mean():>12.4f}{bsm_pnl.mean():>14.4f}")
    print(f"  {'std pnl':18s}{nn_pnl.std():>12.4f}{bsm_pnl.std():>14.4f}")
    for p, a, b in zip(pcts, np.percentile(nn_pnl, pcts), np.percentile(bsm_pnl, pcts)):
        print(f"  {'P' + str(p):18s}{a:>12.4f}{b:>14.4f}")
    print(f"  {'mean bsm price':18s}{mean_bsm_price:>12.4f}")
    print(f"  {'learned premium':18s}{net.premium.item():>12.4f}")
    gap = nn_pnl.std() - bsm_pnl.std()
    print("  " + "-" * 44)
    print(f"  std gap vs oracle BSM: {gap:+.4f}")
    print(f"  (base model lacks the sigma signal, so a gap > 0 is expected)")
    print("=" * 56)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4), sharey=True)
    fig.suptitle(f"Terminal P&L  ({TAG})", fontsize=13)
    for ax, data, title in zip(axes, [nn_pnl, bsm_pnl, unhedged],
                               ["Deep hedge", "BS delta hedge (oracle)", "Unhedged"]):
        ax.hist(data, bins=50, alpha=0.8)
        ax.axvline(data.mean(), color="red", linestyle="--", linewidth=1.1, label=f"mean {data.mean():.4f}")
        ax.axvline(0, color="black", linestyle=":", linewidth=1.0)
        ax.set_title(title)
        ax.set_xlabel("P&L")
        ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, f"pnl_{TAG}.png"), dpi=150, bbox_inches="tight")
    plt.close()

    plot_delta_paths(S_test, sigmas, net)
    print(f"Plots saved with tag '{TAG}'.")


def nn_deltas_path(S_path, net):
    """Network's hedge ratios at each REBAL date for a single daily price path."""
    net.eval()
    deltas = np.zeros(len(REBAL_INDICES))
    S = torch.tensor(S_path, dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        for i, daily_idx in enumerate(REBAL_INDICES):
            S_hist = S[:daily_idx + 1].unsqueeze(0)
            deltas[i] = net(build_state(S_hist, daily_idx, REBAL_TAUS[i])).item()
    return deltas


def plot_delta_paths(S_test, sigmas, net):
    """NN vs BS delta over time for one ITM and one OTM path (using each path's true sigma)."""
    S_T = S_test[DAILY_STEPS, :]
    itm = np.where(S_T > K * 1.05)[0]
    otm = np.where(S_T < K * 0.95)[0]
    if len(itm) == 0 or len(otm) == 0:
        return
    idx = [itm[np.argmax(S_T[itm])], otm[np.argmin(S_T[otm])]]
    rebal_times = np.array(REBAL_INDICES) * H_DAILY

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"Delta path: NN vs BS  ({TAG})", fontsize=13)
    for ax, i, label in zip(axes, idx, ["in-the-money", "out-of-the-money"]):
        path = S_test[:, i]
        sig = sigmas[i]
        nn_deltas = nn_deltas_path(path, net)
        bs_deltas = []
        for daily_idx in REBAL_INDICES:
            tau = T - daily_idx * H_DAILY
            if tau <= 0:
                bs_deltas.append(float(path[daily_idx] > K))
            else:
                d1 = (np.log(path[daily_idx] / K) + 0.5 * sig ** 2 * tau) / (sig * np.sqrt(tau))
                bs_deltas.append(norm.cdf(d1))
        ax.plot(rebal_times, nn_deltas, label="NN", linewidth=1.2)
        ax.plot(rebal_times, bs_deltas, label=f"BS ($\\sigma={sig:.2f}$)",
                linewidth=1.2, linestyle="--")
        ax.set_title(f"{label}  ($S_0$={path[0]:.3f}, $S_T$={path[DAILY_STEPS]:.3f})")
        ax.set_xlabel("time")
        ax.set_ylabel("delta")
        ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, f"delta_{TAG}.png"), dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    S_train, _, _ = generate_paths(N_TRAIN, seed=0)
    net, losses = train(S_train)
    save_model(net, losses)
    S_test, sigmas_test, moneynesses_test = generate_paths(N_TEST, seed=1)
    evaluate(S_test, sigmas_test, moneynesses_test, net)
"""Relvol pricer (moneyness, tau, realized_vol, bs_delta): trains and tests with the
methodology's parameter ranges.

Initial moneyness is sampled from [0.85, 1.15] and volatility from [0.1, 0.3], with
the strike fixed at K=1 and the risk-free rate at r=0. GBM is always simulated at
daily resolution (252 steps); for the monthly variant the hedge rebalances at every
21st daily step while realized vol is computed from the full daily history available
at that point. Flip REBALANCE to switch monthly <-> daily.

The BSM benchmark uses the true per-path sigma (oracle). This network has access to
realized vol as a feature, so it is expected to close the gap to the oracle floor
that the base model leaves open.
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
    "monthly": dict(N=12,  hidden=128, depth=4, lr=0.03, batch_size=1024, clip_norm=1.0),
    "daily":   dict(N=252, hidden=128, depth=4, lr=0.03, batch_size=512,  clip_norm=0.5),
}[REBALANCE]

EPOCHS = 100
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

TAG = f"relvol_{REBALANCE}"
print(f"Using device: {DEVICE}   version: {TAG}  (N={N}, daily_steps={DAILY_STEPS})")


class HedgingNet(nn.Module):
    """Feed-forward delta net; sigmoid output keeps a call delta in (0, 1)."""
    def __init__(self, input_dim=4, hidden=128, depth=4):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden), nn.ReLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.ReLU()]
        layers += [nn.Linear(hidden, 1), nn.Sigmoid()]
        self.net = nn.Sequential(*layers)
        self.premium = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def realized_vol(S_hist):
    """Annualised realized vol from the full daily history up to this rebalancing date.
    Returns zero when fewer than 2 prices are available.
    The network learns that this feature is uninformative early in the path.
    """
    batch, n_obs = S_hist.shape
    if n_obs < 2:
        return torch.zeros(batch, device=DEVICE)
    log_returns = torch.log(S_hist[:, 1:] / S_hist[:, :-1])
    return log_returns.std(dim=1, unbiased=False) / (H_DAILY ** 0.5)


def bs_delta_feat(St, sigma_hat, tau):
    """BS delta built from the realized vol estimate, never the true sigma."""
    sig = torch.clamp(sigma_hat, min=0.01)
    if tau <= 0:
        return (St > K).float()
    d1 = (torch.log(St / K) + 0.5 * sig ** 2 * tau) / (sig * tau ** 0.5)
    return 0.5 * (1.0 + torch.erf(d1 / (2 ** 0.5)))


def build_state(S_hist, tau):
    """Four inputs: moneyness, tau, realized vol, BS delta from realized vol."""
    batch = S_hist.shape[0]
    St    = S_hist[:, -1]
    sig   = realized_vol(S_hist)
    return torch.stack([
        St / K,
        torch.full((batch,), tau, device=DEVICE),
        sig,
        bs_delta_feat(St, sig, tau),
    ], dim=1)


def run_paths(S_batch, net):
    """Roll the hedge forward, rebalancing only at REBAL_INDICES.

    S_batch has shape (batch, DAILY_STEPS+1). Realized vol at each rebalancing
    date is computed from the full daily history up to that point.
    """
    batch      = S_batch.shape[0]
    currency   = torch.zeros(batch, device=DEVICE)
    underlying = torch.zeros(batch, device=DEVICE)
    prev_delta = torch.zeros(batch, device=DEVICE)

    for i, daily_idx in enumerate(REBAL_INDICES):
        S_hist = S_batch[:, :daily_idx + 1]
        delta  = net(build_state(S_hist, REBAL_TAUS[i]))
        trade  = delta - prev_delta
        currency   -= trade * S_hist[:, -1]
        underlying += trade
        prev_delta  = delta

    S_T    = S_batch[:, DAILY_STEPS]
    payoff = torch.clamp(S_T - K, min=0)
    return net.premium + underlying * S_T + currency - payoff


def generate_paths(n_paths, seed=None):
    """Generate n_paths daily GBM trajectories with random initial moneyness and sigma.

    Returns:
        S: (DAILY_STEPS+1, n_paths) array of daily prices
        sigmas: (n_paths,) true volatilities used per path
        moneynesses: (n_paths,) initial moneynesses (= S_0 since K=1)
    """
    rng         = np.random.default_rng(seed)
    moneynesses = rng.uniform(*MONEYNESS_RANGE, n_paths)
    sigmas      = rng.uniform(*SIGMA_RANGE, n_paths)

    Z      = rng.standard_normal((DAILY_STEPS, n_paths))
    W      = np.cumsum(np.sqrt(H_DAILY) * Z, axis=0)
    t_grid = np.arange(1, DAILY_STEPS + 1)[:, None] * H_DAILY
    log_S  = (np.log(moneynesses)[None, :]
              - 0.5 * sigmas[None, :] ** 2 * t_grid
              + sigmas[None, :] * W)
    S_full = np.concatenate([moneynesses[None, :], np.exp(log_S)], axis=0)
    return S_full, sigmas, moneynesses


def bsm_call_vec(S0_vec, sigma_vec):
    """BSM call price, vectorised over S0 and sigma."""
    d1 = (np.log(S0_vec / K) + 0.5 * sigma_vec ** 2 * T) / (sigma_vec * np.sqrt(T))
    d2 = d1 - sigma_vec * np.sqrt(T)
    return S0_vec * norm.cdf(d1) - K * norm.cdf(d2)


def bsm_delta_vec(S, sigma_vec, t):
    """BSM call delta, vectorised over S and sigma."""
    tau = T - t
    if tau <= 0:
        return np.where(S > K, 1.0, 0.0)
    d1 = (np.log(S / K) + 0.5 * sigma_vec ** 2 * tau) / (sigma_vec * np.sqrt(tau))
    return norm.cdf(d1)


def bsm_hedge_pnl(S, sigmas, moneynesses):
    """Oracle BSM hedge using each path's TRUE sigma."""
    n_paths    = S.shape[1]
    currency   = np.zeros(n_paths)
    underlying = np.zeros(n_paths)
    prev_delta = np.zeros(n_paths)
    premiums   = bsm_call_vec(moneynesses, sigmas)

    for daily_idx in REBAL_INDICES:
        St    = S[daily_idx, :]
        delta = bsm_delta_vec(St, sigmas, daily_idx * H_DAILY)
        trade = delta - prev_delta
        currency   -= trade * St
        underlying += trade
        prev_delta  = delta

    S_T = S[DAILY_STEPS, :]
    return premiums + underlying * S_T + currency - np.maximum(S_T - K, 0)


def train(S_train):
    net   = HedgingNet(4, HP["hidden"], HP["depth"]).to(DEVICE)
    opt   = optim.Adam(net.parameters(), lr=HP["lr"])
    sched = optim.lr_scheduler.StepLR(opt, step_size=STEP_SIZE, gamma=GAMMA)

    S_tensor = torch.tensor(S_train.T, dtype=torch.float32)
    loader   = DataLoader(TensorDataset(S_tensor), batch_size=HP["batch_size"], shuffle=True)

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
    os.makedirs(MODELS_DIR, exist_ok=True)
    path = os.path.join(MODELS_DIR, f"{TAG}.pt")
    torch.save({
        "tag":             TAG,
        "feature_set":     "base_relvol_bsdelta",
        "rebalance":       REBALANCE,
        "input_dim":       4,
        "N":               N,
        "daily_steps":     DAILY_STEPS,
        "h_daily":         H_DAILY,
        "rebal_indices":   REBAL_INDICES,
        "K": K, "r": r, "T": T,
        "moneyness_range": MONEYNESS_RANGE,
        "sigma_range":     SIGMA_RANGE,
        "hyperparameters": HP,
        "epochs":          EPOCHS,
        "step_size":       STEP_SIZE,
        "gamma":           GAMMA,
        "n_train":         N_TRAIN,
        "premium":         net.premium.item(),
        "final_train_loss": float(losses[-1]) if losses else None,
        "state_dict":      net.state_dict(),
    }, path)
    print(f"Saved model to {path}")


def diagnose_features(S_test, sigmas, net):
    """Check realised vol and BS delta feature quality at key daily indices."""
    print("\n=== Feature quality diagnostic ===")
    check_points = [1, 5, 21, 63, 126, 210]
    for daily_idx in check_points:
        if daily_idx >= S_test.shape[0]:
            continue
        S_hist = torch.tensor(
            S_test[:daily_idx + 1, :].T, dtype=torch.float32, device=DEVICE
        )
        with torch.no_grad():
            rv  = realized_vol(S_hist).cpu().numpy()
            St  = S_hist[:, -1].cpu().numpy()
            tau = 1.0 - daily_idx / DAILY_STEPS
            sig = torch.tensor(rv, device=DEVICE)
            bsd = bs_delta_feat(
                torch.tensor(St, device=DEVICE), sig, tau
            ).cpu().numpy()
        corr = np.corrcoef(rv, sigmas)[0, 1] if rv.std() > 0 else float("nan")
        print(f"  day={daily_idx:>3d}  "
              f"rv_mean={rv.mean():.3f}  rv_std={rv.std():.3f}  "
              f"true_sigma_mean={sigmas.mean():.3f}  "
              f"corr(rv,sigma)={corr:.3f}  "
              f"bsd_mean={bsd.mean():.3f}  bsd_std={bsd.std():.3f}")


def diagnose_delta_by_sigma(S_test, sigmas, net):
    """Check whether the network outputs different deltas for different sigma paths
    at ATM moneyness. If corr(NN delta, true sigma) ~ 0, the network is ignoring
    the vol signal entirely and outputting an average delta like the base model.
    """
    print("\n=== Delta sensitivity to sigma (ATM paths only) ===")
    atm_mask = np.abs(S_test[0, :] - 1.0) < 0.05
    if atm_mask.sum() < 100:
        print("  Not enough ATM paths to diagnose.")
        return

    S_atm   = S_test[:, atm_mask]
    sig_atm = sigmas[atm_mask]

    check_points = [21, 63, 126, 210]
    net.eval()
    for daily_idx in check_points:
        if daily_idx >= S_atm.shape[0]:
            continue
        tau    = 1.0 - daily_idx / DAILY_STEPS
        S_hist = torch.tensor(
            S_atm[:daily_idx + 1, :].T, dtype=torch.float32, device=DEVICE
        )
        with torch.no_grad():
            deltas = net(build_state(S_hist, tau)).cpu().numpy()

        corr = np.corrcoef(deltas, sig_atm)[0, 1] if deltas.std() > 0 else float("nan")

        # oracle BSM delta at ATM for each sigma
        St_np = S_atm[daily_idx, :]
        oracle_deltas = bsm_delta_vec(St_np, sig_atm, daily_idx * H_DAILY)
        oracle_corr   = np.corrcoef(oracle_deltas, sig_atm)[0, 1]

        print(f"  day={daily_idx:>3d}  "
              f"corr(NN_delta, sigma)={corr:+.3f}  "
              f"corr(oracle_delta, sigma)={oracle_corr:+.3f}  "
              f"NN_delta_std={deltas.std():.3f}  "
              f"oracle_delta_std={oracle_deltas.std():.3f}")

    print(f"\n  Interpretation:")
    print(f"  If corr(NN_delta, sigma) ~ corr(oracle_delta, sigma), the network")
    print(f"  is correctly using the vol signal.")
    print(f"  If corr(NN_delta, sigma) ~ 0 while oracle corr is high, the network")
    print(f"  is ignoring the vol signal and outputting an average delta.")


def evaluate(S_test, sigmas, moneynesses, net):
    net.eval()
    with torch.no_grad():
        nn_pnl = run_paths(
            torch.tensor(S_test.T, dtype=torch.float32).to(DEVICE), net
        ).cpu().numpy()
    bsm_pnl        = bsm_hedge_pnl(S_test, sigmas, moneynesses)
    mean_bsm_price = bsm_call_vec(moneynesses, sigmas).mean()
    unhedged       = net.premium.item() - np.maximum(S_test[DAILY_STEPS, :] - K, 0)

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
    print(f"  ({'matches oracle floor' if abs(gap) < 5e-4 else 'above oracle floor'})")
    print("=" * 56)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4), sharey=True)
    fig.suptitle(f"Terminal P&L  ({TAG})", fontsize=13)
    for ax, data, title in zip(
        axes, [nn_pnl, bsm_pnl, unhedged],
        ["Deep hedge (relvol)", "BS delta hedge (oracle)", "Unhedged"]
    ):
        ax.hist(data, bins=50, alpha=0.8)
        ax.axvline(data.mean(), color="red", linestyle="--", linewidth=1.1,
                   label=f"mean {data.mean():.4f}")
        ax.axvline(0, color="black", linestyle=":", linewidth=1.0)
        ax.set_title(title)
        ax.set_xlabel("P&L")
        ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(SCRIPT_DIR, f"pnl_{TAG}.png"), dpi=150, bbox_inches="tight")
    plt.close()

    plot_delta_paths(S_test, sigmas, net)
    diagnose_features(S_test, sigmas, net)
    diagnose_delta_by_sigma(S_test, sigmas, net)
    print(f"\nPlots saved with tag '{TAG}'.")


def nn_deltas_path(S_path, net):
    net.eval()
    S      = torch.tensor(S_path, dtype=torch.float32, device=DEVICE)
    deltas = np.zeros(len(REBAL_INDICES))
    with torch.no_grad():
        for i, daily_idx in enumerate(REBAL_INDICES):
            S_hist    = S[:daily_idx + 1].unsqueeze(0)
            deltas[i] = net(build_state(S_hist, REBAL_TAUS[i])).item()
    return deltas


def plot_delta_paths(S_test, sigmas, net):
    S_T = S_test[DAILY_STEPS, :]
    itm = np.where(S_T > K * 1.05)[0]
    otm = np.where(S_T < K * 0.95)[0]
    if len(itm) == 0 or len(otm) == 0:
        return
    idx         = [itm[np.argmax(S_T[itm])], otm[np.argmin(S_T[otm])]]
    rebal_times = np.array(REBAL_INDICES) * H_DAILY

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"Delta path: NN vs BS  ({TAG})", fontsize=13)
    for ax, i, label in zip(axes, idx, ["in-the-money", "out-of-the-money"]):
        path = S_test[:, i]
        sig  = sigmas[i]
        nn_deltas = nn_deltas_path(path, net)
        bs_deltas = []
        for daily_idx in REBAL_INDICES:
            tau = T - daily_idx * H_DAILY
            if tau <= 0:
                bs_deltas.append(float(path[daily_idx] > K))
            else:
                d1 = (np.log(path[daily_idx] / K) + 0.5 * sig ** 2 * tau) / (sig * np.sqrt(tau))
                bs_deltas.append(norm.cdf(d1))
        ax.plot(rebal_times, nn_deltas, label="NN (relvol)", linewidth=1.2)
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
    S_train, _, _                         = generate_paths(N_TRAIN, seed=0)
    net, losses                           = train(S_train)
    save_model(net, losses)
    S_test, sigmas_test, moneynesses_test = generate_paths(N_TEST, seed=1)
    evaluate(S_test, sigmas_test, moneynesses_test, net)
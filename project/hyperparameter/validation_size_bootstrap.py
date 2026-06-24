"""Relvol pricer + bootstrap CI for validation/test set sizing.

Trains the hedging network (delta net + premium net), saves the checkpoint,
then immediately runs the bootstrap CI procedure on the saved model to
determine sufficient validation and test set sizes. Everything runs in one
script so the checkpoint is always consistent with the bootstrap.

Architecture
------------
- Delta network  : feedforward, ReLU hidden layers, sigmoid output in (0,1)
- Premium network: S0/K -> premium  (2-layer MLP, 16 hidden units)
Both are trained jointly under MSE loss on terminal P&L.

Edit the block below to switch between monthly and daily rebalancing.
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import norm
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── edit here ─────────────────────────────────────────────────────────────────
REBALANCE = "monthly"       # "monthly" or "daily"

HP = {
    "monthly": dict(N=12,  hidden=64,  depth=4, lr=0.01, batch_size=1024, clip_norm=1.0),
    "daily":   dict(N=252, hidden=128, depth=3, lr=0.01, batch_size=1024, clip_norm=0.5),
}[REBALANCE]

EPOCHS           = 100
STEP_SIZE, GAMMA = 20, 0.7
N_TRAIN          = 10_000
PREMIUM_LR_SCALE = 0.5

# bootstrap settings
N_BOOTSTRAP      = 2_000
CONFIDENCE       = 0.95
REL_TOL          = 0.01        # 1% relative CI half-width
BOOTSTRAP_SEED   = 99999
CANDIDATE_SIZES  = [1_000, 2_000, 5_000, 10_000, 20_000, 50_000, 100_000]
# ─────────────────────────────────────────────────────────────────────────────

K, r, T         = 1.0, 0.0, 1.0
MONEYNESS_RANGE = (0.85, 1.15)
SIGMA_RANGE     = (0.1, 0.3)
DAILY_STEPS     = 252
H_DAILY         = T / DAILY_STEPS

N            = HP["N"]
REBAL_STRIDE = DAILY_STEPS // N
REBAL_INDICES = list(range(0, DAILY_STEPS, REBAL_STRIDE))
REBAL_TAUS    = [1.0 - i / DAILY_STEPS for i in REBAL_INDICES]
assert len(REBAL_INDICES) == N

TAG = f"relvol_{REBALANCE}"
print(f"Device: {DEVICE}   tag: {TAG}   N={N}")
print(f"Hyperparameters: {HP}\n")


# ── network ───────────────────────────────────────────────────────────────────

class HedgingNet(nn.Module):
    def __init__(self, input_dim=4, hidden=128, depth=4):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden), nn.ReLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.ReLU()]
        layers += [nn.Linear(hidden, 1), nn.Sigmoid()]
        self.net = nn.Sequential(*layers)

        self.premium_net = nn.Sequential(
            nn.Linear(1, 16), nn.ReLU(),
            nn.Linear(16, 1),
        )
        nn.init.zeros_(self.premium_net[-1].weight)
        nn.init.zeros_(self.premium_net[-1].bias)

    def forward(self, x):
        return self.net(x).squeeze(-1)

    def premium_for(self, S0):
        return self.premium_net((S0 / K).unsqueeze(-1)).squeeze(-1)


# ── features ──────────────────────────────────────────────────────────────────

def realized_vol(S_hist):
    batch, n_obs = S_hist.shape
    if n_obs < 2:
        return torch.zeros(batch, device=DEVICE)
    log_returns = torch.log(S_hist[:, 1:] / S_hist[:, :-1])
    return log_returns.std(dim=1, unbiased=False) / (H_DAILY ** 0.5)


def realized_vol_np(S_hist_np):
    if S_hist_np.shape[0] < 2:
        return np.zeros(S_hist_np.shape[1])
    log_returns = np.log(S_hist_np[1:] / S_hist_np[:-1])
    return log_returns.std(axis=0, ddof=0) / (H_DAILY ** 0.5)


def bs_delta_feat(St, sigma_hat, tau):
    sig = torch.clamp(sigma_hat, min=0.01)
    if tau <= 0:
        return (St > K).float()
    d1 = (torch.log(St / K) + 0.5 * sig ** 2 * tau) / (sig * tau ** 0.5)
    return 0.5 * (1.0 + torch.erf(d1 / (2 ** 0.5)))


def build_state(S_hist, tau):
    batch = S_hist.shape[0]
    St    = S_hist[:, -1]
    sig   = realized_vol(S_hist)
    return torch.stack([
        St / K,
        torch.full((batch,), tau, device=DEVICE),
        sig,
        bs_delta_feat(St, sig, tau),
    ], dim=1)


# ── path generation ───────────────────────────────────────────────────────────

def generate_paths(n_paths, seed=None):
    rng         = np.random.default_rng(seed)
    moneynesses = rng.uniform(*MONEYNESS_RANGE, n_paths)
    sigmas      = rng.uniform(*SIGMA_RANGE, n_paths)
    Z           = rng.standard_normal((DAILY_STEPS, n_paths))
    W           = np.cumsum(np.sqrt(H_DAILY) * Z, axis=0)
    t_grid      = np.arange(1, DAILY_STEPS + 1)[:, None] * H_DAILY
    log_S       = (np.log(moneynesses)[None, :]
                   - 0.5 * sigmas[None, :] ** 2 * t_grid
                   + sigmas[None, :] * W)
    S_full      = np.concatenate([moneynesses[None, :], np.exp(log_S)], axis=0)
    return S_full, sigmas, moneynesses


# ── forward pass ──────────────────────────────────────────────────────────────

def run_paths(S_batch, net):
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

    premium = net.premium_for(S_batch[:, 0])
    S_T     = S_batch[:, DAILY_STEPS]
    payoff  = torch.clamp(S_T - K, min=0)
    return premium + underlying * S_T + currency - payoff


# ── BSM helpers ───────────────────────────────────────────────────────────────

def bsm_call_vec(S0_vec, sigma_vec):
    d1 = (np.log(S0_vec / K) + 0.5 * sigma_vec ** 2 * T) / (sigma_vec * np.sqrt(T))
    d2 = d1 - sigma_vec * np.sqrt(T)
    return S0_vec * norm.cdf(d1) - K * norm.cdf(d2)


def bsm_delta_vec(S, sigma_vec, t):
    tau = T - t
    if tau <= 0:
        return np.where(S > K, 1.0, 0.0)
    d1 = (np.log(S / K) + 0.5 * sigma_vec ** 2 * tau) / (sigma_vec * np.sqrt(tau))
    return norm.cdf(d1)


def oracle_bsm_hedge_pnl(S, sigmas, moneynesses):
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


def practitioner_bsm_hedge_pnl(S, moneynesses):
    n_paths    = S.shape[1]
    currency   = np.zeros(n_paths)
    underlying = np.zeros(n_paths)
    prev_delta = np.zeros(n_paths)
    premium    = bsm_call_vec(moneynesses, np.full(n_paths, 0.2))  # per-path

    for daily_idx in REBAL_INDICES:
        S_hist_np = S[:daily_idx + 1, :]
        rv        = np.clip(realized_vol_np(S_hist_np), 0.01, None)
        St        = S[daily_idx, :]
        delta     = bsm_delta_vec(St, rv, daily_idx * H_DAILY)
        trade     = delta - prev_delta
        currency   -= trade * St
        underlying += trade
        prev_delta  = delta

    S_T = S[DAILY_STEPS, :]
    return premium + underlying * S_T + currency - np.maximum(S_T - K, 0)


# ── training ──────────────────────────────────────────────────────────────────

def train(S_train):
    net = HedgingNet(4, HP["hidden"], HP["depth"]).to(DEVICE)
    opt = optim.Adam([
        {"params": net.net.parameters()},
        {"params": net.premium_net.parameters(), "lr": HP["lr"] * PREMIUM_LR_SCALE},
    ], lr=HP["lr"])
    sched    = optim.lr_scheduler.StepLR(opt, step_size=STEP_SIZE, gamma=GAMMA)
    S_tensor = torch.tensor(S_train.T, dtype=torch.float32)
    loader   = DataLoader(TensorDataset(S_tensor), batch_size=HP["batch_size"], shuffle=True)

    print(f"\n{'epoch':>6}  {'loss':>12}  {'mean pnl':>10}  {'std pnl':>10}  {'prem@1':>9}")
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
                pnl      = run_paths(S_tensor[:2000].to(DEVICE), net)
                prem_at1 = net.premium_for(torch.tensor([1.0], device=DEVICE)).item()
            print(f"{epoch:>6}  {losses[-1]:>12.4f}  {pnl.mean().item():>10.4f}  "
                  f"{pnl.std().item():>10.4f}  {prem_at1:>9.4f}")
    return net, losses


# ── save / load ───────────────────────────────────────────────────────────────

def save_model(net, losses):
    path = os.path.join(MODELS_DIR, f"{TAG}.pt")
    with torch.no_grad():
        prem_at1 = net.premium_for(torch.tensor([1.0], device=DEVICE)).item()
    torch.save({
        "tag":              TAG,
        "feature_set":      "base_relvol_bsdelta",
        "rebalance":        REBALANCE,
        "input_dim":        4,
        "N":                N,
        "daily_steps":      DAILY_STEPS,
        "h_daily":          H_DAILY,
        "rebal_indices":    REBAL_INDICES,
        "K": K, "r": r, "T": T,
        "moneyness_range":  MONEYNESS_RANGE,
        "sigma_range":      SIGMA_RANGE,
        "hyperparameters":  HP,
        "epochs":           EPOCHS,
        "premium_at_atm":   prem_at1,
        "final_train_loss": float(losses[-1]),
        "state_dict":       net.state_dict(),
    }, path)
    print(f"\nModel saved to {path}")
    return path


def load_model(path):
    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    net  = HedgingNet(ckpt["input_dim"],
                      ckpt["hyperparameters"]["hidden"],
                      ckpt["hyperparameters"]["depth"]).to(DEVICE)
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    print(f"Loaded {ckpt['tag']}  premium@ATM={ckpt['premium_at_atm']:.4f}")
    return net


# ── bootstrap CI ──────────────────────────────────────────────────────────────

def compute_pnl_np(S_np, net):
    """Run the hedging strategy on numpy paths and return P&L as numpy array."""
    S_tensor   = torch.tensor(S_np.T, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        pnl = run_paths(S_tensor, net)
    return pnl.cpu().numpy()


def bootstrap_ci(pnl):
    rng      = np.random.default_rng(0)
    n        = len(pnl)
    boot_std = np.array([
        pnl[rng.integers(0, n, n)].std()
        for _ in range(N_BOOTSTRAP)
    ])
    alpha  = 1 - CONFIDENCE
    lo, hi = np.percentile(boot_std, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi), float(boot_std.mean())


def run_bootstrap(net, set_type, seed_offset):
    print(f"\n{'=' * 62}")
    print(f"Bootstrap CI -- {set_type} set sizing  "
          f"(target rel. half-width <= {REL_TOL:.0%})")
    print(f"{'=' * 62}")

    max_size     = max(CANDIDATE_SIZES)
    S_pool, _, _ = generate_paths(max_size, seed=BOOTSTRAP_SEED + seed_offset)

    results = []
    chosen  = None

    for size in CANDIDATE_SIZES:
        S    = S_pool[:, :size]
        pnl  = compute_pnl_np(S, net)
        lo, hi, boot_mean = bootstrap_ci(pnl)
        ci_half  = (hi - lo) / 2
        rel_half = ci_half / boot_mean if boot_mean > 0 else float("inf")
        sufficient = rel_half <= REL_TOL

        print(f"  size={size:>7,}  pnl_std={pnl.std():.5f}  "
              f"95% CI=[{lo:.5f}, {hi:.5f}]  "
              f"rel={rel_half:.3%}  "
              f"{'✓' if sufficient else ''}")

        results.append({
            "size":     size,
            "pnl_std":  float(pnl.std()),
            "ci_lo":    lo,
            "ci_hi":    hi,
            "ci_half":  ci_half,
            "rel_half": rel_half,
        })

        if sufficient and chosen is None:
            chosen = size

    if chosen is None:
        chosen = max(CANDIDATE_SIZES)
        print(f"\n  WARNING: target not met -- using {chosen:,}")
    else:
        print(f"\n  Sufficient {set_type} set size: {chosen:,} paths")

    return chosen, results


# ── evaluation (same as upgraded_pricer) ─────────────────────────────────────

def evaluate(S_test, sigmas, moneynesses, net):
    net.eval()
    with torch.no_grad():
        nn_pnl = run_paths(
            torch.tensor(S_test.T, dtype=torch.float32).to(DEVICE), net
        ).cpu().numpy()
        prem_at1  = net.premium_for(torch.tensor([1.0], device=DEVICE)).item()
        prem_test = net.premium_for(
            torch.tensor(moneynesses, dtype=torch.float32, device=DEVICE)
        ).cpu().numpy()

    oracle_pnl       = oracle_bsm_hedge_pnl(S_test, sigmas, moneynesses)
    practitioner_pnl = practitioner_bsm_hedge_pnl(S_test, moneynesses)
    mean_bsm_price   = bsm_call_vec(moneynesses, sigmas).mean()

    pcts = [1, 5, 25, 75, 95, 99]
    print("\n" + "=" * 70)
    print(f"  {'':18s}{'NN':>12}{'BSM (prac.)':>14}{'BSM (oracle)':>14}")
    print("  " + "-" * 58)
    for label, fn in [("mean pnl", np.mean), ("std pnl", np.std)]:
        print(f"  {label:18s}{fn(nn_pnl):>12.4f}{fn(practitioner_pnl):>14.4f}"
              f"{fn(oracle_pnl):>14.4f}")
    for p in pcts:
        print(f"  {'P' + str(p):18s}{np.percentile(nn_pnl, p):>12.4f}"
              f"{np.percentile(practitioner_pnl, p):>14.4f}"
              f"{np.percentile(oracle_pnl, p):>14.4f}")
    print(f"  {'mean bsm price':18s}{mean_bsm_price:>12.4f}")
    print(f"  {'learned prem@1':18s}{prem_at1:>12.4f}")
    print(f"  {'mean learned prem':18s}{prem_test.mean():>12.4f}")
    print("  " + "-" * 58)
    gap_prac   = nn_pnl.std() - practitioner_pnl.std()
    gap_oracle = nn_pnl.std() - oracle_pnl.std()
    print(f"  std gap vs practitioner BSM: {gap_prac:+.4f}  "
          f"({'NN wins' if gap_prac < 0 else 'BSM wins'})")
    print(f"  std gap vs oracle BSM:       {gap_oracle:+.4f}")
    print("=" * 70)


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # 1. train
    print("── Generating training paths ──")
    S_train, _, _ = generate_paths(N_TRAIN, seed=0)
    print("── Training ──")
    net, losses = train(S_train)

    # 2. save checkpoint
    model_path = save_model(net, losses)

    # 3. quick evaluation on a held-out set
    print("\n── Quick evaluation ──")
    S_eval, sigmas_eval, mon_eval = generate_paths(5_000, seed=42)
    evaluate(S_eval, sigmas_eval, mon_eval, net)

    # 4. bootstrap CI for validation set size
    #    load from disk to confirm the saved checkpoint is self-consistent
    print("\n── Loading saved model for bootstrap ──")
    net = load_model(model_path)

    chosen_val, results_val = run_bootstrap(net, set_type="validation", seed_offset=0)
    chosen_test, results_test = run_bootstrap(net, set_type="test", seed_offset=1)

    # 5. summary table
    print(f"\n{'=' * 62}")
    print(f"  DATASET SIZE SUMMARY  ({TAG})")
    print(f"{'=' * 62}")
    print(f"  Training set :  {N_TRAIN:,} paths  (fixed)")

    for set_type, chosen, results in [
        ("Validation", chosen_val,  results_val),
        ("Test",       chosen_test, results_test),
    ]:
        row = next(r for r in results if r["size"] == chosen)
        print(f"  {set_type:12s}:  {chosen:,} paths  "
              f"pnl_std={row['pnl_std']:.5f}  "
              f"95% CI=[{row['ci_lo']:.5f}, {row['ci_hi']:.5f}]  "
              f"rel={row['rel_half']:.3%}")

    # 6. save bootstrap results
    out = {
        "tag":         TAG,
        "n_train":     N_TRAIN,
        "validation":  {"chosen": chosen_val,  "results": results_val},
        "test":        {"chosen": chosen_test, "results": results_test},
    }
    out_path = os.path.join(SCRIPT_DIR, f"bootstrap_{TAG}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nBootstrap results saved to {out_path}")
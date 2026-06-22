"""Hyperparameter tuning for the deep hedging network: 2 feature sets x 2 rebalancing frequencies.

Paths are generated under the methodology's parameter ranges: initial moneyness sampled
uniformly from [0.85, 1.15], effective volatility from [0.1, 0.3], K=1, r=0, T=1.
GBM is always simulated at daily resolution (252 steps); the monthly variant rebalances
at every 21st step while computing features from the full daily history.

Training set: 10k paths. Validation set: 20k paths (both per methodology chapter).

Benchmarks used for scoring:
  - Base versions:   practitioner BSM with fixed sigma=0.2 (same info as base network)
  - Relvol versions: practitioner BSM with realised vol   (same info as relvol network)

Step-decay schedule (s=30, gamma=0.5) and zero weight decay are held fixed throughout.
"""

import dataclasses
import itertools
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import norm


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── methodology constants ──────────────────────────────────────────────────────
K, r, T          = 1.0, 0.0, 1.0
MONEYNESS_RANGE  = (0.85, 1.15)
SIGMA_RANGE      = (0.1, 0.3)
SIGMA_FIXED      = 0.2          # practitioner BSM sigma for base versions
DAILY_STEPS      = 252
H_DAILY          = T / DAILY_STEPS

# fixed schedule — not tuned
STEP_SIZE        = 30
GAMMA            = 0.5
WEIGHT_DECAY     = 0.0

# dataset sizes per methodology chapter
N_TRAIN          = 10_000
N_VAL            = 20_000
VAL_SEED         = 12345

# feature sets
FEATURE_CONFIGS = {
    "base":               ["moneyness", "tau"],
    "base_relvol_bsdelta": ["moneyness", "tau", "realized_vol", "bs_delta_feature"],
}

# rebalancing schedules — always daily resolution, downsampled for monthly
REBALANCE_CONFIGS = {
    "monthly": 12,
    "daily":   252,
}


# ── module-level mutable state (set by configure_version) ─────────────────────
N            = 12
REBAL_INDICES = list(range(0, DAILY_STEPS, DAILY_STEPS // 12))
REBAL_TAUS    = [1.0 - i / DAILY_STEPS for i in REBAL_INDICES]
FEATURE_NAMES = FEATURE_CONFIGS["base"]
INPUT_DIM     = 2
VERSION_NAME  = "base__monthly"


def configure_version(feature_set, rebalance_freq):
    """Switch the module to one (feature set, rebalancing frequency) combination."""
    global N, REBAL_INDICES, REBAL_TAUS, FEATURE_NAMES, INPUT_DIM, VERSION_NAME

    if feature_set not in FEATURE_CONFIGS:
        raise ValueError(f"Unknown feature_set '{feature_set}'. Choose from {list(FEATURE_CONFIGS)}.")
    if rebalance_freq not in REBALANCE_CONFIGS:
        raise ValueError(f"Unknown rebalance_freq '{rebalance_freq}'. Choose from {list(REBALANCE_CONFIGS)}.")

    N             = REBALANCE_CONFIGS[rebalance_freq]
    stride        = DAILY_STEPS // N
    REBAL_INDICES = list(range(0, DAILY_STEPS, stride))
    REBAL_TAUS    = [1.0 - i / DAILY_STEPS for i in REBAL_INDICES]
    FEATURE_NAMES = FEATURE_CONFIGS[feature_set]
    INPUT_DIM     = len(FEATURE_NAMES)
    VERSION_NAME  = f"{feature_set}__{rebalance_freq}"
    print(f"[configure_version] {VERSION_NAME}:  features={FEATURE_NAMES}  "
          f"N={N}  rebal_stride={stride}")
    return VERSION_NAME


# ── path generation ────────────────────────────────────────────────────────────

def generate_paths(n_paths, seed=None):
    """GBM paths with random moneyness in [0.85,1.15] and sigma in [0.1,0.3].

    Always simulates DAILY_STEPS=252 daily steps. Returns:
        S:           (DAILY_STEPS+1, n_paths)
        sigmas:      (n_paths,)
        moneynesses: (n_paths,)
    """
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


# ── feature builders ───────────────────────────────────────────────────────────

def realized_vol_feat(S_hist):
    """Annualised realised vol from the full daily history at this rebalancing date."""
    batch, n_obs = S_hist.shape
    if n_obs < 2:
        return torch.zeros(batch, device=DEVICE)
    log_ret = torch.log(S_hist[:, 1:] / S_hist[:, :-1])
    return log_ret.std(dim=1, unbiased=False) / (H_DAILY ** 0.5)


def bs_delta_feat(St, sigma_hat, tau):
    """BS delta from the realised vol estimate — never the true sigma."""
    sig = torch.clamp(sigma_hat, min=0.01)
    if tau <= 0:
        return (St > K).float()
    d1 = (torch.log(St / K) + 0.5 * sig ** 2 * tau) / (sig * tau ** 0.5)
    return 0.5 * (1.0 + torch.erf(d1 / (2 ** 0.5)))


def build_state(S_hist, tau):
    """Assemble the (batch, INPUT_DIM) input tensor for the active feature set."""
    batch = S_hist.shape[0]
    St    = S_hist[:, -1]
    tau_t = torch.full((batch,), tau, device=DEVICE)

    if FEATURE_NAMES == ["moneyness", "tau"]:
        return torch.stack([St / K, tau_t], dim=1)

    # base_relvol_bsdelta
    sig = realized_vol_feat(S_hist)
    return torch.stack([St / K, tau_t, sig, bs_delta_feat(St, sig, tau)], dim=1)


# ── network ────────────────────────────────────────────────────────────────────

@dataclasses.dataclass
class HParams:
    hidden:     int   = 64
    depth:      int   = 3
    lr:         float = 1e-2
    batch_size: int   = 512
    epochs:     int   = 40
    clip_norm:  float = 1.0


class HedgingNet(nn.Module):
    """Feed-forward delta network; sigmoid output keeps a call delta in (0,1)."""
    def __init__(self, hp):
        super().__init__()
        layers = [nn.Linear(INPUT_DIM, hp.hidden), nn.ReLU()]
        for _ in range(hp.depth - 1):
            layers += [nn.Linear(hp.hidden, hp.hidden), nn.ReLU()]
        layers += [nn.Linear(hp.hidden, 1), nn.Sigmoid()]
        self.net     = nn.Sequential(*layers)
        self.premium = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ── forward pass ───────────────────────────────────────────────────────────────

def run_paths(S_batch, net):
    """Roll the hedge forward across REBAL_INDICES; S_batch is (batch, DAILY_STEPS+1)."""
    batch      = S_batch.shape[0]
    currency   = torch.zeros(batch, device=DEVICE)
    underlying = torch.zeros(batch, device=DEVICE)
    prev_delta = torch.zeros(batch, device=DEVICE)

    for daily_idx, tau in zip(REBAL_INDICES, REBAL_TAUS):
        S_hist = S_batch[:, :daily_idx + 1]
        delta  = net(build_state(S_hist, tau))
        trade  = delta - prev_delta
        currency   -= trade * S_hist[:, -1]
        underlying += trade
        prev_delta  = delta

    S_T    = S_batch[:, DAILY_STEPS]
    payoff = torch.clamp(S_T - K, min=0)
    return net.premium + underlying * S_T + currency - payoff


# ── BSM benchmarks ─────────────────────────────────────────────────────────────

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


def realized_vol_np(S_hist_np):
    """Numpy realised vol for use in the practitioner BSM benchmark."""
    if S_hist_np.shape[0] < 2:
        return np.zeros(S_hist_np.shape[1])
    log_ret = np.log(S_hist_np[1:] / S_hist_np[:-1])
    return log_ret.std(axis=0, ddof=0) / (H_DAILY ** 0.5)


def practitioner_bsm_pnl(S, moneynesses, feature_set):
    """Practitioner BSM benchmark — uses the same information as the network.

    Base versions:   fixed sigma=0.2 with average premium across paths.
    Relvol versions: realised vol at each rebalancing date with average premium.
    """
    n_paths    = S.shape[1]
    currency   = np.zeros(n_paths)
    underlying = np.zeros(n_paths)
    prev_delta = np.zeros(n_paths)

    if feature_set == "base":
        sigma_arr = np.full(n_paths, SIGMA_FIXED)
        premium   = float(bsm_call_vec(moneynesses, sigma_arr).mean())
    else:
        # relvol: no history at t=0, use fallback
        sigma_arr = np.full(n_paths, 0.01)
        premium   = float(bsm_call_vec(moneynesses, np.full(n_paths, SIGMA_FIXED)).mean())

    for daily_idx in REBAL_INDICES:
        if feature_set == "base":
            sig_use = sigma_arr
        else:
            rv      = realized_vol_np(S[:daily_idx + 1, :])
            sig_use = np.clip(rv, 0.01, None)

        St    = S[daily_idx, :]
        delta = bsm_delta_vec(St, sig_use, daily_idx * H_DAILY)
        trade = delta - prev_delta
        currency   -= trade * St
        underlying += trade
        prev_delta  = delta

    S_T = S[DAILY_STEPS, :]
    return premium + underlying * S_T + currency - np.maximum(S_T - K, 0)


# ── training ───────────────────────────────────────────────────────────────────

def train_one_config(hp, S_train, S_val):
    """Train one HP config; score by validation P&L std (isolates hedge quality from premium)."""
    net  = HedgingNet(hp).to(DEVICE)
    opt  = optim.Adam(net.parameters(), lr=hp.lr, weight_decay=WEIGHT_DECAY)
    sched = optim.lr_scheduler.StepLR(opt, step_size=STEP_SIZE, gamma=GAMMA)

    S_tensor = torch.tensor(S_train.T, dtype=torch.float32)
    loader   = DataLoader(TensorDataset(S_tensor), batch_size=hp.batch_size, shuffle=True)
    val_t    = torch.tensor(S_val.T, dtype=torch.float32).to(DEVICE)

    best_val = float("inf")
    for epoch in range(hp.epochs):
        net.train()
        for (S_batch,) in loader:
            S_batch = S_batch.to(DEVICE)
            opt.zero_grad()
            loss = (run_paths(S_batch, net) ** 2).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), hp.clip_norm)
            opt.step()
        sched.step()

        net.eval()
        with torch.no_grad():
            val_std = run_paths(val_t, net).std().item()
        best_val = min(best_val, val_std)

    return net, best_val


def confirm_with_seeds(hp, S_train, S_val, n_seeds=5):
    """Retrain the winning config across seeds to confirm the result is not an initialisation artifact."""
    scores = []
    for seed in range(n_seeds):
        torch.manual_seed(seed)
        _, val_std = train_one_config(hp, S_train, S_val)
        print(f"    seed {seed}: val_std={val_std:.5f}")
        scores.append(val_std)
    arr = np.array(scores)
    print(f"    mean={arr.mean():.5f}  std_across_seeds={arr.std():.5f}")
    return arr


# ── search grids ───────────────────────────────────────────────────────────────
# Step-decay schedule (s=30, gamma=0.5) and weight decay=0 are held fixed.
# Daily width is fixed at 128 because the ~21x cost per epoch makes wider searches
# too expensive and 128 was dominant in preliminary work.

MONTHLY_GRID = {
    "hidden":     [64, 128],
    "depth":      [3, 4],
    "lr":         [1e-2, 3e-2],
    "batch_size": [512, 1024],
    "clip_norm":  [1.0, 2.0],
}   # 2^5 = 32 combinations

DAILY_GRID = {
    "hidden":     [128],
    "depth":      [3, 4],
    "lr":         [1e-2, 3e-2],
    "batch_size": [512, 1024],
    "clip_norm":  [0.5, 1.0],
}   # 2^4 = 16 combinations


# ── grid search ────────────────────────────────────────────────────────────────

def grid_search(feature_set, rebalance_freq, epochs=40, n_seeds=5):
    """Full grid search for one (feature_set, rebalance_freq) version."""
    configure_version(feature_set, rebalance_freq)

    space  = MONTHLY_GRID if rebalance_freq == "monthly" else DAILY_GRID
    keys   = list(space.keys())
    combos = list(itertools.product(*[space[k] for k in keys]))

    print(f"\nGenerating train ({N_TRAIN:,}) and validation ({N_VAL:,}) sets...")
    torch.manual_seed(0)
    S_train, _, mon_train = generate_paths(N_TRAIN, seed=0)
    S_val,   _, mon_val   = generate_paths(N_VAL,   seed=VAL_SEED)

    prac_std = practitioner_bsm_pnl(S_val, mon_val, feature_set).std()
    print(f"Practitioner BSM std on validation set: {prac_std:.5f}")
    print(f"Grid has {len(combos)} combinations.\n")

    results = []
    for i, combo in enumerate(combos):
        hp  = HParams(**dict(zip(keys, combo)), epochs=epochs)
        torch.manual_seed(0)
        _, val_std = train_one_config(hp, S_train, S_val)
        gap = val_std - prac_std
        print(f"  [{i + 1:>3}/{len(combos)}] val_std={val_std:.5f}  "
              f"gap_vs_prac={gap:+.5f}  {hp}")
        results.append((hp, val_std, gap))

    results.sort(key=lambda row: row[1])
    print(f"\nTop 5 for '{VERSION_NAME}':")
    for hp, val_std, gap in results[:5]:
        print(f"  val_std={val_std:.5f}  gap={gap:+.5f}  {hp}")

    best_hp = results[0][0]
    print(f"\nConfirming best config across {n_seeds} seeds...")
    confirm_with_seeds(best_hp, S_train, S_val, n_seeds=n_seeds)

    return results


# ── main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    VERSIONS_TO_RUN = [
        ("base",               "monthly"),
        ("base",               "daily"),
        ("base_relvol_bsdelta", "monthly"),
        ("base_relvol_bsdelta", "daily"),
    ]

    all_results = {}
    for feature_set, rebalance_freq in VERSIONS_TO_RUN:
        print(f"\n{'=' * 70}")
        print(f"VERSION: {feature_set}__{rebalance_freq}")
        print(f"{'=' * 70}")
        all_results[f"{feature_set}__{rebalance_freq}"] = grid_search(
            feature_set, rebalance_freq, epochs=40, n_seeds=5
        )
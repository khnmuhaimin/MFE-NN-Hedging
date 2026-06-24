"""Training set size sweep: find the smallest dataset at which validation loss plateaus.

Paths are generated under the methodology's parameter ranges: initial moneyness
sampled uniformly from [0.85, 1.15], effective volatility from [0.1, 0.3], K=1,
r=0, T=1. GBM is always simulated at daily resolution (252 steps); the monthly
variant rebalances at every 21st step while computing features from the full
daily history.

For each candidate training size, a network is trained under fixed default
hyperparameters with early stopping, replicated across NUM_SEEDS independent
seeds. The mean validation P&L standard deviation is plotted with +/- 1 SE
error bars. The smallest size whose mean is within 2 combined SEs of the
largest-size mean is reported as the chosen training set size.

Architecture: the full model is used throughout -- a delta network (feedforward,
sigmoid output) jointly trained with a small premium network (S0/K -> premium).
This is the same architecture used in all subsequent experiments so the plateau
point found here applies directly.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import norm
from torch.utils.data import DataLoader, TensorDataset


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# methodology parameter ranges (fixed)
K, r, T = 1.0, 0.0, 1.0
MONEYNESS_RANGE = (0.85, 1.15)
SIGMA_RANGE     = (0.1, 0.3)
DAILY_STEPS     = 252
H_DAILY         = T / DAILY_STEPS

# sweep settings
TRAINING_SIZES  = [5_000, 10_000, 15_000, 20_000, 50_000]
VALIDATION_SIZE = 5_000
NUM_SEEDS       = 5
VAL_SEED        = 12345
EARLY_STOPPING_PATIENCE  = 10
EARLY_STOPPING_MIN_DELTA = 1e-5

# fixed default hyperparameters -- mid-range of the eventual tuning grids
DEFAULT_HPS = {
    "monthly": dict(hidden=64,  depth=3, lr=1e-2, batch_size=512, clip_norm=1.0),
    "daily":   dict(hidden=128, depth=3, lr=1e-2, batch_size=512, clip_norm=1.0),
}

# rebalancing schedules
REBAL_INDICES = {
    "monthly": list(range(0, DAILY_STEPS, DAILY_STEPS // 12)),
    "daily":   list(range(0, DAILY_STEPS)),
}
REBAL_TAUS = {
    freq: [1.0 - i / DAILY_STEPS for i in indices]
    for freq, indices in REBAL_INDICES.items()
}

# feature dimensions
INPUT_DIM = {
    "base":                2,
    "base_relvol_bsdelta": 4,
}

# premium network architecture (fixed -- no hyperparameter search required;
# see methodology section for justification)
PREMIUM_HIDDEN = 16
PREMIUM_LR_SCALE = 0.5   # premium head trained at lr * this


# ── path generation ────────────────────────────────────────────────────────────

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
    S           = np.concatenate([moneynesses[None, :], np.exp(log_S)], axis=0)
    return S, sigmas, moneynesses


# ── network ────────────────────────────────────────────────────────────────────

class HedgingNet(nn.Module):
    """Delta network with a small premium network conditioned on initial moneyness.

    The delta network outputs a hedge ratio in (0, 1) at each rebalancing date.
    The premium network maps S0/K to an option premium, replacing the single
    scalar used in earlier formulations. Both are trained jointly under the
    MSE loss on terminal P&L.
    """
    def __init__(self, input_dim, hidden, depth):
        super().__init__()
        # delta network
        layers = [nn.Linear(input_dim, hidden), nn.ReLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.ReLU()]
        layers += [nn.Linear(hidden, 1), nn.Sigmoid()]
        self.net = nn.Sequential(*layers)

        # premium network: S0/K -> premium
        # zero-initialised output so the premium is discovered from the gradient
        # signal rather than seeded with a prior value
        self.premium_net = nn.Sequential(
            nn.Linear(1, PREMIUM_HIDDEN), nn.ReLU(),
            nn.Linear(PREMIUM_HIDDEN, 1),
        )
        nn.init.zeros_(self.premium_net[-1].weight)
        nn.init.zeros_(self.premium_net[-1].bias)

    def forward(self, x):
        return self.net(x).squeeze(-1)

    def premium_for(self, S0):
        """Per-path premium as a function of initial price S0 (shape: batch,)."""
        return self.premium_net((S0 / K).unsqueeze(-1)).squeeze(-1)


# ── feature builders ───────────────────────────────────────────────────────────

def realized_vol(S_hist):
    batch, n_obs = S_hist.shape
    if n_obs < 2:
        return torch.zeros(batch, device=DEVICE)
    log_returns = torch.log(S_hist[:, 1:] / S_hist[:, :-1])
    return log_returns.std(dim=1, unbiased=False) / (H_DAILY ** 0.5)


def bs_delta_feat(St, sigma_hat, tau):
    sig = torch.clamp(sigma_hat, min=0.01)
    if tau <= 0:
        return (St > K).float()
    d1 = (torch.log(St / K) + 0.5 * sig ** 2 * tau) / (sig * tau ** 0.5)
    return 0.5 * (1.0 + torch.erf(d1 / (2 ** 0.5)))


def build_state(S_hist, tau, feature_set):
    batch = S_hist.shape[0]
    St    = S_hist[:, -1]
    tau_t = torch.full((batch,), tau, device=DEVICE)
    if feature_set == "base":
        return torch.stack([St / K, tau_t], dim=1)
    else:
        sig = realized_vol(S_hist)
        return torch.stack([St / K, tau_t, sig, bs_delta_feat(St, sig, tau)], dim=1)


# ── forward pass ───────────────────────────────────────────────────────────────

def run_paths(S_batch, net, rebal_indices, rebal_taus, feature_set):
    batch      = S_batch.shape[0]
    currency   = torch.zeros(batch, device=DEVICE)
    underlying = torch.zeros(batch, device=DEVICE)
    prev_delta = torch.zeros(batch, device=DEVICE)

    for daily_idx, tau in zip(rebal_indices, rebal_taus):
        S_hist     = S_batch[:, :daily_idx + 1]
        delta      = net(build_state(S_hist, tau, feature_set))
        trade      = delta - prev_delta
        currency  -= trade * S_hist[:, -1]
        underlying += trade
        prev_delta  = delta

    premium = net.premium_for(S_batch[:, 0])
    S_T     = S_batch[:, DAILY_STEPS]
    payoff  = torch.clamp(S_T - K, min=0)
    return premium + underlying * S_T + currency - payoff


# ── oracle BSM benchmark ───────────────────────────────────────────────────────

def bsm_hedge_pnl(S, sigmas, moneynesses, rebal_indices):
    """Oracle BSM: uses each path's true sigma and a per-path premium."""
    n_paths    = S.shape[1]
    currency   = np.zeros(n_paths)
    underlying = np.zeros(n_paths)
    prev_delta = np.zeros(n_paths)

    d1_0     = (np.log(moneynesses / K) + 0.5 * sigmas ** 2 * T) / (sigmas * np.sqrt(T))
    premiums = moneynesses * norm.cdf(d1_0) - K * norm.cdf(d1_0 - sigmas * np.sqrt(T))

    for daily_idx in rebal_indices:
        St  = S[daily_idx, :]
        tau = T - daily_idx * H_DAILY
        if tau <= 0:
            delta = np.where(St > K, 1.0, 0.0)
        else:
            d1    = (np.log(St / K) + 0.5 * sigmas ** 2 * tau) / (sigmas * np.sqrt(tau))
            delta = norm.cdf(d1)
        trade       = delta - prev_delta
        currency   -= trade * St
        underlying += trade
        prev_delta  = delta

    S_T = S[DAILY_STEPS, :]
    return premiums + underlying * S_T + currency - np.maximum(S_T - K, 0)


# ── training ───────────────────────────────────────────────────────────────────

def set_all_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_with_early_stopping(hp, S_train, S_val, rebal_indices, rebal_taus,
                               feature_set, patience, min_delta):
    input_dim = INPUT_DIM[feature_set]
    net  = HedgingNet(input_dim, hp["hidden"], hp["depth"]).to(DEVICE)

    # separate learning rates: premium head is simpler so trains at half speed
    # to avoid its gradient scale interfering with the delta network early in training
    opt = optim.Adam([
        {"params": net.net.parameters()},
        {"params": net.premium_net.parameters(), "lr": hp["lr"] * PREMIUM_LR_SCALE},
    ], lr=hp["lr"])
    sched = optim.lr_scheduler.StepLR(opt, step_size=30, gamma=0.5)

    S_tensor = torch.tensor(S_train.T, dtype=torch.float32)
    loader   = DataLoader(TensorDataset(S_tensor), batch_size=hp["batch_size"], shuffle=True)
    val_t    = torch.tensor(S_val.T, dtype=torch.float32).to(DEVICE)

    best_val   = float("inf")
    no_improve = 0
    last_epoch = 0

    for epoch in range(200):
        last_epoch = epoch + 1
        net.train()
        for (S_batch,) in loader:
            S_batch = S_batch.to(DEVICE)
            opt.zero_grad()
            loss = (run_paths(S_batch, net, rebal_indices, rebal_taus, feature_set) ** 2).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), hp["clip_norm"])
            opt.step()
        sched.step()

        net.eval()
        with torch.no_grad():
            val_std = run_paths(val_t, net, rebal_indices, rebal_taus, feature_set).std().item()

        if val_std < best_val - min_delta:
            best_val   = val_std
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    return best_val, last_epoch


# ── flatline detection and plotting ───────────────────────────────────────────

def pick_flatline(sizes, means, stderrs, k=2.0):
    target_mean = means[-1]
    target_se   = max(stderrs[-1], 1e-9)
    for size, m, se in zip(sizes, means, stderrs):
        combined = max(np.sqrt(se ** 2 + target_se ** 2), 1e-9)
        if (m - target_mean) <= k * combined:
            return size
    return sizes[-1]


def plot_curve(sizes, results, bsm_floor, version_name, out_path, chosen_size):
    means   = np.array([r["mean"]   for r in results])
    stderrs = np.array([r["stderr"] for r in results])

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    rng = np.random.default_rng(0)
    for size, r_ in zip(sizes, results):
        jitter = size * rng.uniform(0.97, 1.03, size=len(r_["raw"]))
        ax.scatter(jitter, r_["raw"], color="C0", alpha=0.35, s=18, zorder=1)

    n_seeds = len(results[0]["raw"])
    ax.errorbar(sizes, means, yerr=stderrs, marker="o", capsize=4,
                color="C0", linewidth=1.6,
                label=f"NN val P&L std (mean $\\pm$ 1 SE, $n={n_seeds}$ seeds)",
                zorder=3)
    ax.axhline(bsm_floor, ls="--", color="grey",
               label=f"Oracle BSM floor ({bsm_floor:.4f})", zorder=0)
    ax.axvline(chosen_size, ls=":", color="C3", alpha=0.7,
               label=f"chosen: {chosen_size:,} paths", zorder=0)
    ax.set_xlabel("Training set size (number of paths)")
    ax.set_ylabel("Validation P&L standard deviation")
    ax.set_title(f"Training set size sweep — {version_name}")
    ax.set_xscale("log")
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")


# ── main sweep ─────────────────────────────────────────────────────────────────

def run_sweep(feature_set, rebalance_freq, sizes, val_size, num_seeds):
    version_name  = f"{feature_set}__{rebalance_freq}"
    rebal_indices = REBAL_INDICES[rebalance_freq]
    rebal_taus    = REBAL_TAUS[rebalance_freq]
    hp            = DEFAULT_HPS[rebalance_freq]
    max_size      = max(sizes)

    print(f"\n{'=' * 70}")
    print(f"VERSION: {version_name}  |  N={len(rebal_indices)}  |  features={feature_set}")
    print(f"{'=' * 70}")

    S_val, sigmas_val, mon_val = generate_paths(val_size, seed=VAL_SEED)
    bsm_floor = float(bsm_hedge_pnl(S_val, sigmas_val, mon_val, rebal_indices).std())
    print(f"Validation set fixed (seed={VAL_SEED}).  Oracle BSM floor: {bsm_floor:.5f}\n")

    print(f"Generating {num_seeds} independent training pools of {max_size:,} paths each...")
    pools = []
    for seed in range(num_seeds):
        S_pool, _, _ = generate_paths(max_size, seed=seed)
        pools.append(S_pool)

    results = []
    for size in sizes:
        seed_results = []
        for seed in range(num_seeds):
            S_train = pools[seed][:, :size]
            set_all_seeds(seed)
            best_val, stopped_at = train_with_early_stopping(
                hp, S_train, S_val, rebal_indices, rebal_taus, feature_set,
                patience=EARLY_STOPPING_PATIENCE,
                min_delta=EARLY_STOPPING_MIN_DELTA,
            )
            print(f"  [{version_name}] size={size:>6,}  seed={seed}  "
                  f"val_std={best_val:.5f}  gap={best_val - bsm_floor:+.5f}  "
                  f"stopped_at={stopped_at}")
            seed_results.append(best_val)

        mean   = float(np.mean(seed_results))
        stderr = float(np.std(seed_results, ddof=1) / np.sqrt(num_seeds)) if num_seeds > 1 else 0.0
        results.append({"size": size, "raw": seed_results, "mean": mean, "stderr": stderr})
        print(f"  -> size={size:>6,}  mean={mean:.5f} +/- {stderr:.5f}\n")

    means   = [r["mean"]   for r in results]
    stderrs = [r["stderr"] for r in results]
    chosen  = pick_flatline(sizes, means, stderrs)

    print(f"\n=== {version_name} summary ===")
    for r_ in results:
        print(f"  size={r_['size']:>6,}  mean={r_['mean']:.5f} +/- {r_['stderr']:.5f}  "
              f"gap_to_floor={r_['mean'] - bsm_floor:+.5f}")
    print(f"\n  chosen training size: {chosen:,}  (smallest mean within 2 SE of size {sizes[-1]:,})")

    out_dir = Path("results/dataset_size")
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_curve(sizes, results, bsm_floor, version_name,
               out_dir / f"{version_name}.png", chosen)

    summary = {
        "version":              version_name,
        "feature_set":          feature_set,
        "rebalance_freq":       rebalance_freq,
        "moneyness_range":      MONEYNESS_RANGE,
        "sigma_range":          SIGMA_RANGE,
        "daily_steps":          DAILY_STEPS,
        "num_seeds":            num_seeds,
        "val_seed":             VAL_SEED,
        "validation_size":      val_size,
        "bsm_floor":            bsm_floor,
        "chosen_training_size": chosen,
        "results":              results,
        "hyperparameters":      hp,
        "premium_architecture": {
            "type":          "premium_net",
            "hidden":        PREMIUM_HIDDEN,
            "lr_scale":      PREMIUM_LR_SCALE,
        },
    }
    with open(out_dir / f"{version_name}.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


# ── quick check ───────────────────────────────────────────────────────────────

QUICK_CHECK_HPS = [
    dict(hidden=64,  depth=3, lr=1e-2, batch_size=512, clip_norm=1.0),
    dict(hidden=128, depth=3, lr=1e-2, batch_size=512, clip_norm=1.0),
    dict(hidden=64,  depth=3, lr=3e-2, batch_size=512, clip_norm=1.0),
    dict(hidden=128, depth=4, lr=1e-2, batch_size=512, clip_norm=0.5),
    dict(hidden=64,  depth=3, lr=3e-3, batch_size=512, clip_norm=2.0),
]

QUICK_CHECK_SIZE   = 20_000
QUICK_CHECK_EPOCHS = 80


def run_quick_check(feature_set, rebalance_freq):
    version_name  = f"{feature_set}__{rebalance_freq}"
    rebal_indices = REBAL_INDICES[rebalance_freq]
    rebal_taus    = REBAL_TAUS[rebalance_freq]

    print(f"\n{'=' * 70}")
    print(f"QUICK CHECK: {version_name}")
    print(f"  training size={QUICK_CHECK_SIZE:,}  epochs={QUICK_CHECK_EPOCHS}  "
          f"n_hp_combos={len(QUICK_CHECK_HPS)}")
    print(f"{'=' * 70}")

    S_val, sigmas_val, mon_val = generate_paths(VALIDATION_SIZE, seed=VAL_SEED)
    bsm_floor = float(bsm_hedge_pnl(S_val, sigmas_val, mon_val, rebal_indices).std())
    print(f"Oracle BSM floor (validation): {bsm_floor:.5f}\n")

    S_train, _, _ = generate_paths(QUICK_CHECK_SIZE, seed=0)
    val_t         = torch.tensor(S_val.T, dtype=torch.float32).to(DEVICE)

    results = []
    for i, hp in enumerate(QUICK_CHECK_HPS):
        set_all_seeds(0)
        net = HedgingNet(INPUT_DIM[feature_set], hp["hidden"], hp["depth"]).to(DEVICE)
        opt = optim.Adam([
            {"params": net.net.parameters()},
            {"params": net.premium_net.parameters(), "lr": hp["lr"] * PREMIUM_LR_SCALE},
        ], lr=hp["lr"])
        sched    = optim.lr_scheduler.StepLR(opt, step_size=30, gamma=0.5)
        S_tensor = torch.tensor(S_train.T, dtype=torch.float32)
        loader   = DataLoader(TensorDataset(S_tensor), batch_size=hp["batch_size"], shuffle=True)

        best_val   = float("inf")
        no_improve = 0

        for epoch in range(QUICK_CHECK_EPOCHS):
            net.train()
            for (S_batch,) in loader:
                S_batch = S_batch.to(DEVICE)
                opt.zero_grad()
                loss = (run_paths(S_batch, net, rebal_indices, rebal_taus, feature_set) ** 2).mean()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), hp["clip_norm"])
                opt.step()
            sched.step()

            net.eval()
            with torch.no_grad():
                val_std = run_paths(val_t, net, rebal_indices, rebal_taus, feature_set).std().item()

            if val_std < best_val - EARLY_STOPPING_MIN_DELTA:
                best_val   = val_std
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= QUICK_CHECK_EPOCHS // 8:
                    break

        gap    = best_val - bsm_floor
        status = "OK" if best_val < bsm_floor * 5 else "POOR"
        print(f"  [{i+1}/{len(QUICK_CHECK_HPS)}] {status}  "
              f"val_std={best_val:.5f}  gap={gap:+.5f}  "
              f"stopped_at={epoch+1}  hp={hp}")
        results.append({"hp": hp, "val_std": best_val, "gap": gap})

    results.sort(key=lambda x: x["val_std"])
    best      = results[0]
    val_stds  = [r["val_std"] for r in results]
    hp_spread = max(val_stds) - min(val_stds)
    print(f"\n  Best HP combo:  val_std={best['val_std']:.5f}  gap={best['gap']:+.5f}")
    print(f"  {best['hp']}")
    print(f"\n  Recommendation:")
    if hp_spread < 0.002:
        print(f"  All combos converged (spread={hp_spread:.5f}). HPs are not the bottleneck.")
        print(f"  Proceed with the full sweep.")
    elif best["val_std"] < bsm_floor * 2:
        print(f"  Network is learning well. Update DEFAULT_HPS['{rebalance_freq}'] to the")
        print(f"  best combo above and proceed with the full sweep.")
    else:
        print(f"  Large spread ({hp_spread:.5f}). Consider expanding the search grid.")

    return results


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Training set size sweep")
    parser.add_argument("--feature-set",    default="base",
                        choices=["base", "base_relvol_bsdelta"])
    parser.add_argument("--rebalance-freq", default="monthly",
                        choices=["monthly", "daily"])
    parser.add_argument("--num-seeds", type=int, default=NUM_SEEDS)
    parser.add_argument("--all", action="store_true",
                        help="run all four versions in sequence")
    parser.add_argument("--quick-check", action="store_true",
                        help="run a small HP sanity check before the full sweep")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.quick_check:
        run_quick_check(args.feature_set, args.rebalance_freq)
    elif args.all:
        for fs in ["base", "base_relvol_bsdelta"]:
            for rf in ["monthly", "daily"]:
                run_sweep(fs, rf, TRAINING_SIZES, VALIDATION_SIZE, args.num_seeds)
    else:
        run_sweep(args.feature_set, args.rebalance_freq,
                  TRAINING_SIZES, VALIDATION_SIZE, args.num_seeds)
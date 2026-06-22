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

This experiment is run *before* hyperparameter tuning, so the defaults must
not depend on the tuning results.
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
SIGMA_RANGE = (0.1, 0.3)
DAILY_STEPS = 252
H_DAILY = T / DAILY_STEPS

# sweep settings
TRAINING_SIZES = [5_000, 10_000, 15_000, 20_000, 50_000]
VALIDATION_SIZE = 5_000
NUM_SEEDS = 5
VAL_SEED = 12345
EARLY_STOPPING_PATIENCE = 10
EARLY_STOPPING_MIN_DELTA = 1e-5

# fixed default hyperparameters — mid-range of the eventual tuning grids
DEFAULT_HPS = {
    "monthly": dict(hidden=64,  depth=3, lr=1e-2, batch_size=512,  clip_norm=1.0),
    "daily":   dict(hidden=128, depth=3, lr=1e-2, batch_size=512,  clip_norm=1.0),
}

# rebalancing schedules
REBAL_INDICES = {
    "monthly": list(range(0, DAILY_STEPS, DAILY_STEPS // 12)),   # every 21st daily step
    "daily":   list(range(0, DAILY_STEPS)),                       # every daily step
}
REBAL_TAUS = {
    freq: [1.0 - i / DAILY_STEPS for i in indices]
    for freq, indices in REBAL_INDICES.items()
}

# feature dimensions
INPUT_DIM = {
    "base":               2,
    "base_relvol_bsdelta": 4,
}


# ── path generation ────────────────────────────────────────────────────────────

def generate_paths(n_paths, seed=None):
    """GBM paths with random moneyness in [0.85,1.15] and sigma in [0.1,0.3].

    Returns S of shape (DAILY_STEPS+1, n_paths), sigmas (n_paths,), moneynesses (n_paths,).
    """
    rng = np.random.default_rng(seed)
    moneynesses = rng.uniform(*MONEYNESS_RANGE, n_paths)
    sigmas = rng.uniform(*SIGMA_RANGE, n_paths)

    Z = rng.standard_normal((DAILY_STEPS, n_paths))
    W = np.cumsum(np.sqrt(H_DAILY) * Z, axis=0)
    t_grid = np.arange(1, DAILY_STEPS + 1)[:, None] * H_DAILY
    log_S = (np.log(moneynesses)[None, :]
             - 0.5 * sigmas[None, :] ** 2 * t_grid
             + sigmas[None, :] * W)
    S = np.concatenate([moneynesses[None, :], np.exp(log_S)], axis=0)
    return S, sigmas, moneynesses


# ── network ────────────────────────────────────────────────────────────────────

class HedgingNet(nn.Module):
    """Feed-forward delta net; sigmoid output keeps a call delta in (0, 1)."""
    def __init__(self, input_dim, hidden, depth):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden), nn.ReLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.ReLU()]
        layers += [nn.Linear(hidden, 1), nn.Sigmoid()]
        self.net = nn.Sequential(*layers)
        self.premium = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ── feature builders ───────────────────────────────────────────────────────────

def realized_vol(S_hist):
    """Annualised realised vol from the full daily history available at this rebal date."""
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
    St = S_hist[:, -1]
    tau_t = torch.full((batch,), tau, device=DEVICE)
    if feature_set == "base":
        return torch.stack([St / K, tau_t], dim=1)
    else:
        sig = realized_vol(S_hist)
        return torch.stack([St / K, tau_t, sig, bs_delta_feat(St, sig, tau)], dim=1)


# ── forward pass ───────────────────────────────────────────────────────────────

def run_paths(S_batch, net, rebal_indices, rebal_taus, feature_set):
    """Roll the hedge forward across rebal_indices; S_batch is (batch, DAILY_STEPS+1)."""
    batch = S_batch.shape[0]
    currency   = torch.zeros(batch, device=DEVICE)
    underlying = torch.zeros(batch, device=DEVICE)
    prev_delta = torch.zeros(batch, device=DEVICE)

    for daily_idx, tau in zip(rebal_indices, rebal_taus):
        S_hist = S_batch[:, :daily_idx + 1]
        delta = net(build_state(S_hist, tau, feature_set))
        trade = delta - prev_delta
        currency   -= trade * S_hist[:, -1]
        underlying += trade
        prev_delta  = delta

    S_T    = S_batch[:, DAILY_STEPS]
    payoff = torch.clamp(S_T - K, min=0)
    return net.premium + underlying * S_T + currency - payoff


# ── oracle BSM benchmark ───────────────────────────────────────────────────────

def bsm_hedge_pnl(S, sigmas, moneynesses, rebal_indices):
    """Oracle BSM hedge using each path's TRUE sigma."""
    n_paths    = S.shape[1]
    currency   = np.zeros(n_paths)
    underlying = np.zeros(n_paths)
    prev_delta = np.zeros(n_paths)

    d1_0 = (np.log(moneynesses / K) + 0.5 * sigmas ** 2 * T) / (sigmas * np.sqrt(T))
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
    opt  = optim.Adam(net.parameters(), lr=hp["lr"])
    sched = optim.lr_scheduler.StepLR(opt, step_size=30, gamma=0.5)

    S_tensor = torch.tensor(S_train.T, dtype=torch.float32)
    loader   = DataLoader(TensorDataset(S_tensor), batch_size=hp["batch_size"], shuffle=True)
    val_t    = torch.tensor(S_val.T,   dtype=torch.float32).to(DEVICE)

    best_val = float("inf")
    no_improve = 0
    last_epoch = 0

    for epoch in range(200):        # hard cap; early stopping does the real work
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
    """Smallest size within k combined SEs of the largest-size mean."""
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

    # fixed validation set — same for every seed and size
    S_val, sigmas_val, mon_val = generate_paths(val_size, seed=VAL_SEED)
    bsm_floor = float(bsm_hedge_pnl(S_val, sigmas_val, mon_val, rebal_indices).std())
    print(f"Validation set fixed (seed={VAL_SEED}).  Oracle BSM floor: {bsm_floor:.5f}\n")

    # one large training pool per seed, sliced down for each size
    print(f"Generating {num_seeds} independent training pools of {max_size:,} paths each...")
    pools = []
    for seed in range(num_seeds):
        S_pool, _, _ = generate_paths(max_size, seed=seed)
        pools.append(S_pool)

    val_t = torch.tensor(S_val.T, dtype=torch.float32).to(DEVICE)

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
        "version":               version_name,
        "feature_set":           feature_set,
        "rebalance_freq":        rebalance_freq,
        "moneyness_range":       MONEYNESS_RANGE,
        "sigma_range":           SIGMA_RANGE,
        "daily_steps":           DAILY_STEPS,
        "num_seeds":             num_seeds,
        "val_seed":              VAL_SEED,
        "validation_size":       val_size,
        "bsm_floor":             bsm_floor,
        "chosen_training_size":  chosen,
        "results":               results,
        "hyperparameters":       hp,
    }
    with open(out_dir / f"{version_name}.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


# ── quick check ───────────────────────────────────────────────────────────────

# HP grid for the quick check: a small set of combinations spanning the
# important axes (learning rate, width, clip norm). Run time is
# len(QUICK_CHECK_HPS) * 2 training runs per (feature_set, rebalance_freq).
QUICK_CHECK_HPS = [
    dict(hidden=64,  depth=3, lr=1e-2,  batch_size=512, clip_norm=1.0),  # default monthly
    dict(hidden=128, depth=3, lr=1e-2,  batch_size=512, clip_norm=1.0),  # default daily
    dict(hidden=64,  depth=3, lr=3e-2,  batch_size=512, clip_norm=1.0),  # higher lr
    dict(hidden=128, depth=4, lr=1e-2,  batch_size=512, clip_norm=0.5),  # deeper, tighter clip
    dict(hidden=64,  depth=3, lr=3e-3,  batch_size=512, clip_norm=2.0),  # lower lr, loose clip
]

QUICK_CHECK_SIZE   = 20_000   # single training size — enough to see convergence
QUICK_CHECK_EPOCHS = 80       # more epochs than the sweep default so slow configs get a chance


def run_quick_check(feature_set, rebalance_freq):
    """Train a small set of HP combinations on a fixed 20k pool and report val std vs BSM floor.

    Use this before the full sweep to confirm that at least one HP combination
    is actually learning under the methodology-aligned generator (varying moneyness
    and sigma). If every combination flatlines well above the BSM floor, the HPs
    need revisiting before the full sweep is worth running.
    """
    version_name  = f"{feature_set}__{rebalance_freq}"
    rebal_indices = REBAL_INDICES[rebalance_freq]
    rebal_taus    = REBAL_TAUS[rebalance_freq]

    print(f"\n{'=' * 70}")
    print(f"QUICK CHECK: {version_name}")
    print(f"  training size={QUICK_CHECK_SIZE:,}  epochs={QUICK_CHECK_EPOCHS}  "
          f"n_hp_combos={len(QUICK_CHECK_HPS)}")
    print(f"{'=' * 70}")

    # fixed validation set
    S_val, sigmas_val, mon_val = generate_paths(VALIDATION_SIZE, seed=VAL_SEED)
    bsm_floor = float(bsm_hedge_pnl(S_val, sigmas_val, mon_val, rebal_indices).std())
    print(f"Oracle BSM floor (validation): {bsm_floor:.5f}\n")

    # single training pool — same data for every HP combo so we compare HPs not data
    S_train, _, _ = generate_paths(QUICK_CHECK_SIZE, seed=0)

    results = []
    for i, hp in enumerate(QUICK_CHECK_HPS):
        # use a high-epoch version of the training function
        hp_with_epochs = {**hp}
        set_all_seeds(0)
        net   = HedgingNet(INPUT_DIM[feature_set], hp["hidden"], hp["depth"]).to(DEVICE)
        opt   = optim.Adam(net.parameters(), lr=hp["lr"])
        sched = optim.lr_scheduler.StepLR(opt, step_size=30, gamma=0.5)

        S_tensor = torch.tensor(S_train.T, dtype=torch.float32)
        loader   = DataLoader(TensorDataset(S_tensor),
                              batch_size=hp["batch_size"], shuffle=True)
        val_t    = torch.tensor(S_val.T, dtype=torch.float32).to(DEVICE)

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
                if no_improve >= EARLY_STOPPING_PATIENCE:
                    break

        gap = best_val - bsm_floor
        status = "OK" if best_val < bsm_floor * 5 else "POOR"
        print(f"  [{i+1}/{len(QUICK_CHECK_HPS)}] {status}  "
              f"val_std={best_val:.5f}  gap={gap:+.5f}  "
              f"stopped_at={epoch+1}  hp={hp}")
        results.append({"hp": hp, "val_std": best_val, "gap": gap})

    results.sort(key=lambda x: x["val_std"])
    best = results[0]
    print(f"\n  Best HP combo:  val_std={best['val_std']:.5f}  gap={best['gap']:+.5f}")
    print(f"  {best['hp']}")
    val_stds  = [r["val_std"] for r in results]
    hp_spread = max(val_stds) - min(val_stds)
    print(f"\n  Recommendation:")
    if hp_spread < 0.002:
        print(f"  All combos converged to nearly the same val std (spread={hp_spread:.5f}).")
        print(f"  This means HPs are not the bottleneck — the network has saturated given")
        print(f"  the information available. The gap to the oracle floor ({best['val_std'] - bsm_floor:.4f})")
        print(f"  is structural, not a training problem. Proceed with the full sweep using")
        print(f"  the best combo above — the flatline will likely appear at a small training size.")
    elif best["val_std"] < bsm_floor * 2:
        print(f"  Network is learning well. Update DEFAULT_HPS['{rebalance_freq}'] to the")
        print(f"  best combo above and proceed with the full sweep.")
    else:
        print(f"  Large spread across combos ({hp_spread:.5f}) suggests HPs matter.")
        print(f"  Consider expanding the search grid before running the full sweep.")

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
                        help="run a small HP sanity check before committing to the full sweep")
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
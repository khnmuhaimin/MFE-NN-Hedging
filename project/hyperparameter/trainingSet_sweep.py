import argparse
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import norm
from torch.utils.data import DataLoader, TensorDataset
from project.hyperparameter.hyperparam_tuning import (
    DEVICE, K, T, DAILY_STEPS, H_DAILY,
    MONEYNESS_RANGE, SIGMA_RANGE,
    make_paths, realized_vol_feat, bs_delta_feat,
)

# variables, settings and hyperparameters for the premium network
TRAINING_SIZES = [5_000, 10_000, 15_000, 20_000, 50_000]
VALIDATION_SIZE = 5_000
NUM_SEEDS = 5
VAL_SEED = 12345
EARLY_STOPPING_PATIENCE = 10
EARLY_STOPPING_MIN_DELTA = 1e-5
PREMIUM_HIDDEN = 16
PREMIUM_LR_SCALE = 0.5
DEFAULT_HPS = {
    "monthly": dict(hidden=64, depth=3, lr=1e-2, batch_size=512, clip_norm=1.0),
    "daily": dict(hidden=128, depth=3, lr=1e-2, batch_size=512, clip_norm=1.0),
}
REBAL_INDICES = {
    "monthly": list(range(0, DAILY_STEPS, DAILY_STEPS // 12)),
    "daily": list(range(0, DAILY_STEPS)),
}
REBAL_TAUS = {
    freq: [1.0 - i / DAILY_STEPS for i in indices]
    for freq, indices in REBAL_INDICES.items()
}
INPUT_DIM = {"base": 2, "base_relvol_bsdelta": 4}

# delta network + premium network; defined locally because the tuning script uses a scalar premium
class HedgingNet(nn.Module):
    def __init__(self, input_dim, hidden, depth):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden), nn.ReLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.ReLU()]
        layers += [nn.Linear(hidden, 1), nn.Sigmoid()]
        self.net = nn.Sequential(*layers)
        self.premium_net = nn.Sequential(
            nn.Linear(1, PREMIUM_HIDDEN), nn.ReLU(),
            nn.Linear(PREMIUM_HIDDEN, 1),
        )
        nn.init.zeros_(self.premium_net[-1].weight)
        nn.init.zeros_(self.premium_net[-1].bias)

    def forward(self, x):
        return self.net(x).squeeze(-1)

    def premium_for(self, S0):
        return self.premium_net((S0 / K).unsqueeze(-1)).squeeze(-1)

def build_state(S_hist, tau, feature_set):

    batch = S_hist.shape[0]
    St = S_hist[:, -1]
    tau_t = torch.full((batch,), tau, device=DEVICE)
    if feature_set == "base":
        return torch.stack([St / K, tau_t], dim=1)
    sig = realized_vol_feat(S_hist)

    return torch.stack([St / K, tau_t, sig, bs_delta_feat(St, sig, tau)], dim=1)

# roll the hedge forward and return terminal P&L using the premium network
def run_paths(S_batch, net, rebal_indices, rebal_taus, feature_set):

    batch = S_batch.shape[0]
    currency = torch.zeros(batch, device=DEVICE)
    underlying = torch.zeros(batch, device=DEVICE)
    prev_delta = torch.zeros(batch, device=DEVICE)
    for daily_idx, tau in zip(rebal_indices, rebal_taus):
        S_hist = S_batch[:, :daily_idx + 1]
        delta = net(build_state(S_hist, tau, feature_set))
        trade = delta - prev_delta
        currency -= trade * S_hist[:, -1]
        underlying += trade
        prev_delta = delta
    premium = net.premium_for(S_batch[:, 0])
    S_T = S_batch[:, DAILY_STEPS]
    payoff = torch.clamp(S_T - K, min=0)

    return premium + underlying * S_T + currency - payoff

# oracle BSM floor using true per-path sigma; used as the benchmark floor in the sweep
def bsm_hedge_pnl(S, sigmas, moneynesses, rebal_indices):

    n_paths = S.shape[1]
    currency = np.zeros(n_paths)
    underlying = np.zeros(n_paths)
    prev_delta = np.zeros(n_paths)
    d1_0 = (np.log(moneynesses / K) + 0.5 * sigmas**2 * T) / (sigmas * np.sqrt(T))
    premiums = moneynesses * norm.cdf(d1_0) - K * norm.cdf(d1_0 - sigmas * np.sqrt(T))
    for daily_idx in rebal_indices:
        St = S[daily_idx, :]
        tau = T - daily_idx * H_DAILY
        d1 = (np.log(St / K) + 0.5 * sigmas**2 * tau) / (sigmas * np.sqrt(max(tau, 1e-8)))
        delta = np.where(tau <= 0, np.where(St > K, 1.0, 0.0), norm.cdf(d1))
        trade = delta - prev_delta
        currency -= trade * St
        underlying += trade
        prev_delta = delta
    S_T = S[DAILY_STEPS, :]

    return premiums + underlying * S_T + currency - np.maximum(S_T - K, 0)

def set_all_seeds(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

# train one config with early stopping; return best validation P&L std
def train_with_early_stopping(hp, S_train, S_val, rebal_indices, rebal_taus,
                               feature_set, patience, min_delta):

    net = HedgingNet(INPUT_DIM[feature_set], hp["hidden"], hp["depth"]).to(DEVICE)
    opt = optim.Adam([
        {"params": net.net.parameters()},
        {"params": net.premium_net.parameters(), "lr": hp["lr"] * PREMIUM_LR_SCALE},
    ], lr=hp["lr"])
    sched = optim.lr_scheduler.StepLR(opt, step_size=30, gamma=0.5)
    loader = DataLoader(TensorDataset(torch.tensor(S_train.T, dtype=torch.float32)),
                        batch_size=hp["batch_size"], shuffle=True)
    val_t = torch.tensor(S_val.T, dtype=torch.float32).to(DEVICE)
    best_val, no_improve, last_epoch = float("inf"), 0, 0
    for epoch in range(200):
        last_epoch = epoch + 1
        net.train()
        for (S_batch,) in loader:
            S_batch = S_batch.to(DEVICE)
            opt.zero_grad()
            loss = (run_paths(S_batch, net, rebal_indices, rebal_taus, feature_set)**2).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), hp["clip_norm"])
            opt.step()
        sched.step()
        net.eval()
        with torch.no_grad():
            val_std = run_paths(val_t, net, rebal_indices, rebal_taus, feature_set).std().item()
        if val_std < best_val - min_delta:
            best_val, no_improve = val_std, 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    return best_val, last_epoch

# smallest size whose mean is within k combined SEs of the largest-size mean
def pick_flatline(sizes, means, stderrs, k=2.0):

    target_mean, target_se = means[-1], max(stderrs[-1], 1e-9)
    for size, m, se in zip(sizes, means, stderrs):
        if (m - target_mean) <= k*max(np.sqrt(se**2 + target_se**2), 1e-9):
            return size

    return sizes[-1]


def run_sweep(feature_set, rebalance_freq, sizes, val_size, num_seeds):

    version_name = f"{feature_set}__{rebalance_freq}"
    rebal_indices = REBAL_INDICES[rebalance_freq]
    rebal_taus = REBAL_TAUS[rebalance_freq]
    hp = DEFAULT_HPS[rebalance_freq]
    print(f"\n{'='*70}")
    print(f"VERSION: {version_name}  |  N={len(rebal_indices)}  |  features={feature_set}")
    print(f"{'='*70}")
    S_val, sigmas_val, mon_val = make_paths(val_size, seed=VAL_SEED)
    bsm_floor = float(bsm_hedge_pnl(S_val, sigmas_val, mon_val, rebal_indices).std())
    print(f"Validation set fixed (seed={VAL_SEED}).  Oracle BSM floor: {bsm_floor:.5f}\n")
    pools = [make_paths(max(sizes), seed=s)[0] for s in range(num_seeds)]

    results = []
    for size in sizes:
        seed_results = []
        for seed in range(num_seeds):
            set_all_seeds(seed)
            best_val, stopped_at = train_with_early_stopping(
                hp, pools[seed][:, :size], S_val, rebal_indices, rebal_taus, feature_set,
                EARLY_STOPPING_PATIENCE, EARLY_STOPPING_MIN_DELTA,
            )
            print(f"  [{version_name}] size={size:>6,}  seed={seed}  "
                  f"val_std={best_val:.5f}  gap={best_val - bsm_floor:+.5f}  "
                  f"stopped_at={stopped_at}")
            seed_results.append(best_val)
        mean = float(np.mean(seed_results))
        stderr = float(np.std(seed_results, ddof=1) / np.sqrt(num_seeds)) if num_seeds > 1 else 0.0
        results.append({"size": size, "raw": seed_results, "mean": mean, "stderr": stderr})
        print(f"  -> size={size:>6,}  mean={mean:.5f} +/- {stderr:.5f}\n")

    means = [r["mean"] for r in results]
    stderrs = [r["stderr"] for r in results]
    chosen = pick_flatline(sizes, means, stderrs)

    print(f"\n=== {version_name} summary ===")
    for r_ in results:
        print(f"  size={r_['size']:>6,}  mean={r_['mean']:.5f} +/- {r_['stderr']:.5f}  "
              f"gap_to_floor={r_['mean'] - bsm_floor:+.5f}")
    print(f"\n  chosen training size: {chosen:,}  (smallest mean within 2 SE of size {sizes[-1]:,})")

    out_dir = Path("results/dataset_size")
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "version": version_name,
        "feature_set": feature_set,
        "rebalance_freq": rebalance_freq,
        "num_seeds": num_seeds,
        "val_seed": VAL_SEED,
        "validation_size": val_size,
        "bsm_floor": bsm_floor,
        "chosen_training_size": chosen,
        "results": results,
        "hyperparameters": hp,
        "premium_architecture": {"type": "premium_net", "hidden": PREMIUM_HIDDEN,
                                 "lr_scale": PREMIUM_LR_SCALE},
    }
    with open(out_dir / f"{version_name}.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary

def parse_args():
    parser = argparse.ArgumentParser(description="Training set size sweep")
    parser.add_argument("--feature-set", default="base",
                        choices=["base", "base_relvol_bsdelta"])
    parser.add_argument("--rebalance-freq", default="monthly",
                        choices=["monthly", "daily"])
    parser.add_argument("--num-seeds", type=int, default=NUM_SEEDS)
    parser.add_argument("--all", action="store_true", help="run all four versions")
    return parser.parse_args()

# main programme run
if __name__ == "__main__":
    args = parse_args()
    if args.all:
        for fs in ["base", "base_relvol_bsdelta"]:
            for rf in ["monthly", "daily"]:
                run_sweep(fs, rf, TRAINING_SIZES, VALIDATION_SIZE, args.num_seeds)
    else:
        run_sweep(args.feature_set, args.rebalance_freq,
                  TRAINING_SIZES, VALIDATION_SIZE, args.num_seeds)
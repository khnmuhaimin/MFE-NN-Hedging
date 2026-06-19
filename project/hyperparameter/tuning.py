"""Hyperparameter tuning for the deep hedging network: 2 feature sets x 2 rebalancing frequencies."""

import dataclasses
import itertools
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import norm

from project.stock.generators import generate_gbm

# fixed problem parameters, never tuned
S0, K, sigma, r, T = 1, 1, 0.1, 0, 1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# feature sets: name -> the input columns the network sees
FEATURE_CONFIGS = {
    "base": ["moneyness", "tau"],
    "base_relvol_bsdelta": ["moneyness", "tau", "realized_vol", "bs_delta_feature"],
}

# rebalancing frequency: name -> number of rebalancing dates over [0, T]
REBALANCE_CONFIGS = {
    "monthly": 12,
    "daily": 252,
}


def configure_version(feature_set, rebalance_freq):
    """Switch the whole module to one (feature set, rebalancing frequency) combination."""
    global N, h, FEATURE_NAMES, INPUT_DIM, VERSION_NAME

    if feature_set not in FEATURE_CONFIGS:
        raise ValueError(f"Unknown feature_set '{feature_set}'. Pick from {list(FEATURE_CONFIGS)}.")
    if rebalance_freq not in REBALANCE_CONFIGS:
        raise ValueError(f"Unknown rebalance_freq '{rebalance_freq}'. Pick from {list(REBALANCE_CONFIGS)}.")

    FEATURE_NAMES = FEATURE_CONFIGS[feature_set]
    INPUT_DIM = len(FEATURE_NAMES)
    N = REBALANCE_CONFIGS[rebalance_freq]
    h = T / N

    VERSION_NAME = f"{feature_set}__{rebalance_freq}"
    print(f"[configure_version] {VERSION_NAME}:  features={FEATURE_NAMES}  N={N}  h={h:.5f}")
    return VERSION_NAME


# default so the module is usable before configure_version is called explicitly
configure_version("base", "monthly")


def realized_vol_feature(S_hist):
    """Annualized realized vol from the path's whole history so far (all-time window, no look-ahead)."""
    batch, n_obs = S_hist.shape
    if n_obs < 2:
        return torch.zeros(batch, device=DEVICE)
    log_returns = torch.log(S_hist[:, 1:] / S_hist[:, :-1])
    return log_returns.std(dim=1, unbiased=False) / (h ** 0.5)


def bs_delta_feature(St, sigma_hat, tau):
    """BS delta built from the estimated vol sigma_hat, never the true simulation sigma."""
    sigma_safe = torch.clamp(sigma_hat, min=0.01)
    if tau <= 0:
        return (St > K).float()
    d1 = (torch.log(St / K) + (r + 0.5 * sigma_safe ** 2) * tau) / (sigma_safe * tau ** 0.5)
    return 0.5 * (1.0 + torch.erf(d1 / (2 ** 0.5)))


def build_state(S_hist, t):
    """Assemble the (batch, INPUT_DIM) network input for the active feature set."""
    batch = S_hist.shape[0]
    St = S_hist[:, -1]
    tau = 1.0 - t / N
    sigma_hat = None

    cols = []
    for name in FEATURE_NAMES:
        if name == "moneyness":
            cols.append(St / K)
        elif name == "tau":
            cols.append(torch.full((batch,), tau, device=DEVICE))
        elif name == "realized_vol":
            sigma_hat = realized_vol_feature(S_hist) if sigma_hat is None else sigma_hat
            cols.append(sigma_hat)
        elif name == "bs_delta_feature":
            sigma_hat = realized_vol_feature(S_hist) if sigma_hat is None else sigma_hat
            cols.append(bs_delta_feature(St, sigma_hat, tau))
        else:
            raise ValueError(f"Unknown feature name: {name!r}")

    return torch.stack(cols, dim=1)


@dataclasses.dataclass
class HParams:
    """One hyperparameter configuration."""
    hidden: int = 64
    depth: int = 3
    lr: float = 1e-3
    batch_size: int = 2048
    epochs: int = 40
    step_size: int = 30
    gamma: float = 0.5
    clip_norm: float = 1.0
    weight_decay: float = 0.0


class HedgingNet(nn.Module):
    """Feed-forward delta network; sigmoid output keeps a call delta in [0, 1]."""
    def __init__(self, hp, input_dim):
        super().__init__()
        layers = [nn.Linear(input_dim, hp.hidden), nn.ReLU()]
        for _ in range(hp.depth - 1):
            layers += [nn.Linear(hp.hidden, hp.hidden), nn.ReLU()]
        layers += [nn.Linear(hp.hidden, 1)]
        self.net = nn.Sequential(*layers)
        self.out_act = nn.Sigmoid()
        self.premium = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        return self.out_act(self.net(x)).squeeze(-1)


def run_paths(S_batch, hedging_net):
    """Roll the hedge forward over all N steps and return terminal P&L per path."""
    batch = S_batch.shape[0]
    currency = torch.zeros(batch, device=DEVICE)
    underlying = torch.zeros(batch, device=DEVICE)
    prev_delta = torch.zeros(batch, device=DEVICE)

    for t in range(N):
        S_hist = S_batch[:, :t + 1]          # only info available at t
        St = S_hist[:, -1]
        delta = hedging_net(build_state(S_hist, t))

        trade = delta - prev_delta
        currency -= trade * St
        underlying += trade
        prev_delta = delta

    S_T = S_batch[:, N]
    payoff = torch.clamp(S_T - K, min=0)
    return hedging_net.premium + underlying * S_T + currency - payoff


def bsm_call(S0, K, r, sigma, T):
    """Black-Scholes call price."""
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bsm_delta(S, K, r, sigma, t, T):
    """Black-Scholes call delta at time t."""
    tau = T - t
    if tau <= 0:
        return np.where(S > K, 1.0, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * tau) / (sigma * np.sqrt(tau))
    return norm.cdf(d1)


def bsm_hedge_pnl(S):
    """Discrete delta-hedge P&L under the exact BS delta: the benchmark floor."""
    n_paths = S.shape[1]
    currency = np.zeros(n_paths)
    underlying = np.zeros(n_paths)
    prev_delta = np.zeros(n_paths)
    premium = bsm_call(S0, K, r, sigma, T)

    for t in range(N):
        St = S[t, :]
        delta = bsm_delta(St, K, r, sigma, t * h, T)
        trade = delta - prev_delta
        currency -= trade * St
        underlying += trade
        currency *= np.exp(r * h)
        prev_delta = delta

    S_T = S[N, :]
    payoff = np.maximum(S_T - K, 0)
    return premium + underlying * S_T + currency - payoff


def train_one_config(hp, S_train, S_val):
    """Train one config on the given paths; score by validation P&L std (isolates hedging from the premium)."""
    net = HedgingNet(hp, INPUT_DIM).to(DEVICE)
    opt = optim.Adam(net.parameters(), lr=hp.lr, weight_decay=hp.weight_decay)
    sched = optim.lr_scheduler.StepLR(opt, step_size=hp.step_size, gamma=hp.gamma)

    S_tensor = torch.tensor(S_train.T, dtype=torch.float32)
    loader = DataLoader(TensorDataset(S_tensor), batch_size=hp.batch_size, shuffle=True)
    val_tensor = torch.tensor(S_val.T, dtype=torch.float32).to(DEVICE)

    best_val_std = float("inf")
    history = []

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
            val_std = run_paths(val_tensor, net).std().item()
        history.append(val_std)
        best_val_std = min(best_val_std, val_std)

    return net, best_val_std, history


# wide random-search grid; lr range pushed up because sigmoid gradients are weaker than tanh
DEFAULT_SEARCH_SPACE = {
    "hidden": [32, 64, 128],
    "depth": [2, 3, 4],
    "lr": [3e-4, 1e-3, 3e-3, 1e-2, 3e-2],
    "batch_size": [512, 1024, 2048],
    "step_size": [20, 30, 50],
    "gamma": [0.3, 0.5, 0.7],
    "clip_norm": [0.5, 1.0, 2.0],
}


def random_search(n_trials=20, seed=0, epochs=40, space=None):
    """Score n_trials random configs from space against the BSM floor, under the active version."""
    space = space or DEFAULT_SEARCH_SPACE
    rng = random.Random(seed)

    print(f"Simulating shared train/val path sets for '{VERSION_NAME}' (N={N})...")
    S_train = generate_gbm(S0, r, sigma, h, 8_000, N + 1)
    S_val = generate_gbm(S0, r, sigma, h, 2_000, N + 1)
    bsm_std = bsm_hedge_pnl(S_val).std()
    print(f"BSM benchmark std on the validation set: {bsm_std:.4f}\n")

    results = []
    for i in range(n_trials):
        sampled = {k: rng.choice(v) for k, v in space.items()}
        hp = HParams(**sampled, epochs=epochs)

        torch.manual_seed(0)        # same init every trial so we compare hyperparameters, not noise
        _, val_std, _ = train_one_config(hp, S_train, S_val)
        gap = val_std - bsm_std

        print(f"[{i + 1}/{n_trials}] val_std={val_std:.4f}  gap_vs_bsm={gap:+.4f}  {hp}")
        results.append((hp, val_std, gap))

    results.sort(key=lambda row: row[1])
    print(f"\nTop 5 for '{VERSION_NAME}' by validation std:")
    for hp, val_std, gap in results[:5]:
        print(f"  val_std={val_std:.4f}  gap={gap:+.4f}  {hp}")

    return results


# narrowed grids, one per rebalancing frequency (32 / 16 combos); lr does most of the discriminating
MONTHLY_GRID_SEARCH_SPACE = {
    "hidden": [64, 128],
    "depth": [3, 4],
    "lr": [1e-2, 3e-2],
    "batch_size": [512, 1024],
    "step_size": [30],
    "gamma": [0.5],
    "clip_norm": [1.0, 2.0],
}

DAILY_GRID_SEARCH_SPACE = {
    "hidden": [128],            # fixed: ~21x cost per epoch and clearly dominant in the daily log
    "depth": [3, 4],
    "lr": [1e-2, 3e-2],
    "batch_size": [512, 1024],
    "step_size": [30],
    "gamma": [0.5],
    "clip_norm": [0.5, 2.0],
}


def grid_search(space, epochs=40):
    """Exhaustive search over an explicitly passed grid; same scoring and fixed init as random_search."""
    keys = list(space.keys())
    combos = list(itertools.product(*[space[k] for k in keys]))

    print(f"Simulating shared train/val path sets for '{VERSION_NAME}' (N={N})...")
    S_train = generate_gbm(S0, r, sigma, h, 8_000, N + 1)
    S_val = generate_gbm(S0, r, sigma, h, 2_000, N + 1)
    bsm_std = bsm_hedge_pnl(S_val).std()
    print(f"BSM benchmark std on the validation set: {bsm_std:.4f}")
    print(f"Grid has {len(combos)} combinations.\n")

    results = []
    for i, combo in enumerate(combos):
        hp = HParams(**dict(zip(keys, combo)), epochs=epochs)

        torch.manual_seed(0)
        _, val_std, _ = train_one_config(hp, S_train, S_val)
        gap = val_std - bsm_std

        print(f"[{i + 1}/{len(combos)}] val_std={val_std:.4f}  gap_vs_bsm={gap:+.4f}  {hp}")
        results.append((hp, val_std, gap))

    results.sort(key=lambda row: row[1])
    print(f"\nTop 5 for '{VERSION_NAME}' by validation std:")
    for hp, val_std, gap in results[:5]:
        print(f"  val_std={val_std:.4f}  gap={gap:+.4f}  {hp}")

    return results


def optuna_search(n_trials=30, epochs=40):
    """Optional Optuna search, more sample-efficient than the random/grid drivers."""
    import optuna  # pip install optuna --break-system-packages

    S_train = generate_gbm(S0, r, sigma, h, 8_000, N + 1)
    S_val = generate_gbm(S0, r, sigma, h, 2_000, N + 1)
    bsm_std = bsm_hedge_pnl(S_val).std()

    def objective(trial):
        hp = HParams(
            hidden=trial.suggest_categorical("hidden", [32, 64, 128]),
            depth=trial.suggest_int("depth", 2, 4),
            lr=trial.suggest_float("lr", 1e-4, 5e-2, log=True),
            batch_size=trial.suggest_categorical("batch_size", [512, 1024, 2048]),
            step_size=trial.suggest_int("step_size", 15, 50),
            gamma=trial.suggest_float("gamma", 0.2, 0.8),
            clip_norm=trial.suggest_float("clip_norm", 0.3, 2.0),
            epochs=epochs,
        )
        torch.manual_seed(0)
        _, val_std, _ = train_one_config(hp, S_train, S_val)
        return val_std

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    print(f"'{VERSION_NAME}' BSM benchmark std: {bsm_std:.4f}")
    print(f"Best val_std found: {study.best_value:.4f}")
    print("Best hyperparameters:", study.best_params)
    return study


def confirm_with_multiple_seeds(hp, n_seeds=5):
    """Retrain the winning config across seeds to check the win is real, not luck."""
    S_train = generate_gbm(S0, r, sigma, h, 8_000, N + 1)
    S_val = generate_gbm(S0, r, sigma, h, 2_000, N + 1)

    scores = []
    for seed in range(n_seeds):
        torch.manual_seed(seed)
        _, val_std, _ = train_one_config(hp, S_train, S_val)
        print(f"  seed {seed}: val_std={val_std:.4f}")
        scores.append(val_std)

    scores = np.array(scores)
    print(f"mean={scores.mean():.4f}  std_across_seeds={scores.std():.4f}")
    return scores


def run_all_versions(n_trials_per_version=20, epochs=40):
    """Independent random search for every (feature set, rebalancing frequency) pair."""
    all_results = {}
    for feature_set in FEATURE_CONFIGS:
        for rebalance_freq in REBALANCE_CONFIGS:
            version_name = configure_version(feature_set, rebalance_freq)
            print(f"\n{'=' * 70}\nVERSION: {version_name}\n{'=' * 70}")
            all_results[version_name] = random_search(n_trials=n_trials_per_version, epochs=epochs)
    return all_results


if __name__ == "__main__":
    # versions to sweep this run: (feature_set, rebalance_freq)
    VERSIONS_TO_RUN = [
        ("base", "monthly"),
        ("base", "daily"),
        ("base_relvol_bsdelta", "monthly"),
        ("base_relvol_bsdelta", "daily"),
    ]

    for feature_set, rebalance_freq in VERSIONS_TO_RUN:
        configure_version(feature_set, rebalance_freq)
        print(f"\n{'=' * 70}\nVERSION: {VERSION_NAME}\n{'=' * 70}")

        space = MONTHLY_GRID_SEARCH_SPACE if rebalance_freq == "monthly" else DAILY_GRID_SEARCH_SPACE
        results = grid_search(space=space)
        best_hp, _, _ = results[0]

        print(f"\nConfirming the top config for '{VERSION_NAME}' across seeds...")
        confirm_with_multiple_seeds(best_hp, n_seeds=5)

    # run_all_versions(n_trials_per_version=20)  # rerun every version from scratch in one call
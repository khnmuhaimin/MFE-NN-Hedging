"""
Hyperparameter tuning for the Deep Hedging network.

This refactors the core pieces of deep_hedging.py so that the things you'd
want to tune (network width/depth, learning rate, batch size, LR schedule,
gradient clipping) are arguments instead of hardcoded constants, adds a
genuine held-out validation set, and provides two search drivers: a plain
random search (no extra dependencies) and an Optuna-based search (more
sample-efficient, optional dependency).

Output activation is Sigmoid rather than Tanh, matching a vanilla call's
delta range of [0,1]. Sigmoid's gradient is weaker than Tanh's away from
its midpoint (peak 0.25 vs. 1.0), so the `lr` search space is widened
upward versus a Tanh search to give the optimizer a fair chance to find
where this activation actually converges well.

Drop this file alongside deep_hedging.py in the same project package so the
`project.stock.generators` / `project.helpers.path_helpers` imports resolve
the same way they do there.
"""

import dataclasses
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import norm

from project.stock.generators import generate_gbm
from project.helpers.path_helpers import project_path

# Same market / option setup as the main script — these define the problem,
# not the model, so they are not search candidates.
S0, K, sigma, r, N, T = 1, 1, 0.1, 0, 100, 1
h = T / N

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ───────────────────────── hyperparameters ─────────────────────────

@dataclasses.dataclass
class HParams:
    hidden: int = 64
    depth: int = 3          # number of hidden layers
    lr: float = 1e-3
    batch_size: int = 2048
    epochs: int = 40        # kept short during search; lengthen for the final run
    step_size: int = 30
    gamma: float = 0.5
    clip_norm: float = 1.0
    weight_decay: float = 0.0


class HedgingNet(nn.Module):
    def __init__(self, hp: HParams):
        super().__init__()
        layers = [nn.Linear(2, hp.hidden), nn.ReLU()]
        for _ in range(hp.depth - 1):
            layers += [nn.Linear(hp.hidden, hp.hidden), nn.ReLU()]
        layers += [nn.Linear(hp.hidden, 1)]
        self.net = nn.Sequential(*layers)
        self.out_act = nn.Sigmoid()  # call delta lives in [0,1]; was nn.Tanh() ([-1,1])
        self.premium = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        return self.out_act(self.net(x)).squeeze(-1)


# ───────────────────────── rollout (unchanged logic) ─────────────────────────

def run_paths(S_batch: torch.Tensor, hedging_net: HedgingNet) -> torch.Tensor:
    batch = S_batch.shape[0]
    currency = torch.zeros(batch, device=DEVICE)
    underlying = torch.zeros(batch, device=DEVICE)
    prev_delta = torch.zeros(batch, device=DEVICE)

    for t in range(N):
        St = S_batch[:, t]
        tau = 1.0 - t / N
        state = torch.stack(
            [St / K, torch.full((batch,), tau, device=DEVICE)], dim=1
        )
        delta = hedging_net(state)
        trade = delta - prev_delta
        currency -= trade * St
        underlying += trade
        prev_delta = delta

    S_T = S_batch[:, N]
    payoff = torch.clamp(S_T - K, min=0)
    return hedging_net.premium + underlying * S_T + currency - payoff


# ───────────────────────── BSM benchmark (the floor for comparison) ─────────────────────────

def bsm_call(S0, K, r, sigma, T):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bsm_delta(S, K, r, sigma, t, T):
    tau = T - t
    if tau <= 0:
        return np.where(S > K, 1.0, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    return norm.cdf(d1)


def bsm_hedge_pnl(S: np.ndarray) -> np.ndarray:
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


# ───────────────────────── train + validate one config ─────────────────────────

def train_one_config(hp: HParams, S_train: np.ndarray, S_val: np.ndarray):
    """
    Train a single hyperparameter configuration and score it on held-out paths.

    Scoring uses the standard deviation of validation P&L rather than the raw
    loss. The mean of P&L is largely controlled by the learned `premium`
    parameter regardless of hyperparameters, so std isolates how good the
    *hedging* is — which is what you actually want to compare across configs.
    """
    net = HedgingNet(hp).to(DEVICE)
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
            pnl = run_paths(S_batch, net)
            loss = (pnl ** 2).mean()
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


# ───────────────────────── random search ─────────────────────────

def random_search(n_trials: int = 20, seed: int = 0):
    rng = random.Random(seed)
    space = {
        "hidden": [32, 64, 128],
        "depth": [2, 3, 4],
        # lr grid widened upward vs. the Tanh search: Sigmoid's gradient peaks
        # at 0.25 (vs. Tanh's 1.0), so the old 3e-3 ceiling may sit below
        # where Sigmoid actually converges well.
        "lr": [3e-4, 1e-3, 3e-3, 1e-2, 3e-2],
        "batch_size": [512, 1024, 2048],
        "step_size": [20, 30, 50],
        "gamma": [0.3, 0.5, 0.7],
        "clip_norm": [0.5, 1.0, 2.0],
    }

    print("Simulating shared train / validation path sets...")
    S_train = generate_gbm(S0, r, sigma, h, 8_000, N + 1)
    S_val = generate_gbm(S0, r, sigma, h, 2_000, N + 1)
    bsm_std = bsm_hedge_pnl(S_val).std()
    print(f"BSM benchmark std on the validation set: {bsm_std:.4f}\n")

    results = []
    for i in range(n_trials):
        sampled = {k: rng.choice(v) for k, v in space.items()}
        hp = HParams(**sampled, epochs=40)

        torch.manual_seed(0)  # fix init across trials so the search isn't measuring init noise
        _, val_std, _ = train_one_config(hp, S_train, S_val)
        gap = val_std - bsm_std

        print(f"[{i + 1}/{n_trials}] val_std={val_std:.4f}  gap_vs_bsm={gap:+.4f}  {hp}")
        results.append((hp, val_std, gap))

    results.sort(key=lambda r: r[1])
    print("\nTop 5 configs by validation std:")
    for hp, val_std, gap in results[:5]:
        print(f"  val_std={val_std:.4f}  gap={gap:+.4f}  {hp}")

    return results


# ───────────────────────── optuna search (optional, more sample-efficient) ─────────────────────────

def optuna_search(n_trials: int = 30):
    import optuna  # pip install optuna --break-system-packages

    S_train = generate_gbm(S0, r, sigma, h, 8_000, N + 1)
    S_val = generate_gbm(S0, r, sigma, h, 2_000, N + 1)
    bsm_std = bsm_hedge_pnl(S_val).std()

    def objective(trial: "optuna.Trial") -> float:
        hp = HParams(
            hidden=trial.suggest_categorical("hidden", [32, 64, 128]),
            depth=trial.suggest_int("depth", 2, 4),
            lr=trial.suggest_float("lr", 1e-4, 5e-2, log=True),
            batch_size=trial.suggest_categorical("batch_size", [512, 1024, 2048]),
            step_size=trial.suggest_int("step_size", 15, 50),
            gamma=trial.suggest_float("gamma", 0.2, 0.8),
            clip_norm=trial.suggest_float("clip_norm", 0.3, 2.0),
            epochs=40,
        )
        torch.manual_seed(0)
        _, val_std, _ = train_one_config(hp, S_train, S_val)
        return val_std

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    print(f"BSM benchmark std: {bsm_std:.4f}")
    print(f"Best val_std found: {study.best_value:.4f}")
    print("Best hyperparameters:", study.best_params)
    return study


# ───────────────────────── robustness check for the winning config ─────────────────────────

def confirm_with_multiple_seeds(hp: HParams, n_seeds: int = 5):
    """
    Retrain the winning config under several random seeds. A single search
    result can look good purely by luck — both the network's random
    initialization and the finite validation sample add noise. This checks
    the win is real before you commit to a config.
    """
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


if __name__ == "__main__":
    results = random_search(n_trials=20)
    best_hp, _, _ = results[0]

    print("\nConfirming the top config across seeds...")
    confirm_with_multiple_seeds(best_hp, n_seeds=5)
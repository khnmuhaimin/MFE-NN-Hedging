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

NEW: the network's input features and the rebalancing frequency N are now
both switchable "versions" rather than fixed constants. The team has
settled on exactly two feature sets -- `base` and `base_relvol_bsdelta`
-- crossed with two rebalancing frequencies (`monthly`, `daily`), giving
four versions total. See `configure_version()` below -- that single
function call is the one place you need to touch to point an entire
sweep at a different combination. `run_all_versions()` loops over all
four automatically if you want the whole grid in one run.

Drop this file alongside deep_hedging.py in the same project package so the
`project.stock.generators` / `project.helpers.path_helpers` imports resolve
the same way they do there.
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

from project.stock.generators import generate_gbm
from project.helpers.path_helpers import project_path

# Option/market parameters -- these define the problem, not the model, so
# they are never search candidates and never vary by "version".
S0, K, sigma, r, T = 1, 1, 0.1, 0, 1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ════════════════════════════════════════════════════════════════════
#  VERSION CONFIGURATION
#  This is the one section you touch to change what a sweep tests.
# ════════════════════════════════════════════════════════════════════

# Each entry: a name -> the ordered list of features that name turns on.
# The network's input dimension is just len(...) of whichever list is active.
# Exactly two feature sets, per the team's confirmed design:
#   "base"                -- the original two-input network
#   "base_relvol_bsdelta"  -- the input-feature-engineering version: adds a
#                            trailing realized-volatility estimate and a BS
#                            delta computed from that estimate (not the
#                            true sigma -- see realized_vol_feature() /
#                            bs_delta_feature() below).
FEATURE_CONFIGS = {
    "base":                ["moneyness", "tau"],
    "base_relvol_bsdelta": ["moneyness", "tau", "realized_vol", "bs_delta_feature"],
}

# Each entry: a name -> the number of rebalancing dates N over the option's
# 1-year life. 12 = monthly, 252 = daily (matching trading-day convention).
REBALANCE_CONFIGS = {
    "monthly": 12,
    "daily":   252,
}

# Trailing window for the realized-vol feature, expressed in *calendar
# time* (months) rather than a fixed step count, so it represents a
# comparable historical window regardless of which REBALANCE_CONFIGS entry
# is active. See configure_version() for the conversion to a step count.
REALIZED_VOL_WINDOW_MONTHS = 1.0


def configure_version(feature_set: str, rebalance_freq: str) -> str:
    """
    Point every downstream function at a new (feature set, rebalancing
    frequency) combination. Call this once before generating paths or
    running a search -- everything else (network input size, the rollout's
    time grid, the realized-vol window) is derived from these two choices
    and stored in module-level globals that run_paths / train_one_config /
    the BSM benchmark all read at call time.

    feature_set    : a key of FEATURE_CONFIGS, e.g. "base_relvol_bsdelta"
    rebalance_freq : a key of REBALANCE_CONFIGS, e.g. "daily"
    """
    global N, h, FEATURE_NAMES, INPUT_DIM, VOL_WINDOW_STEPS, VERSION_NAME

    if feature_set not in FEATURE_CONFIGS:
        raise ValueError(f"Unknown feature_set '{feature_set}'. "
                          f"Choose from {list(FEATURE_CONFIGS)}.")
    if rebalance_freq not in REBALANCE_CONFIGS:
        raise ValueError(f"Unknown rebalance_freq '{rebalance_freq}'. "
                          f"Choose from {list(REBALANCE_CONFIGS)}.")

    FEATURE_NAMES = FEATURE_CONFIGS[feature_set]
    INPUT_DIM = len(FEATURE_NAMES)

    N = REBALANCE_CONFIGS[rebalance_freq]
    h = T / N

    steps_per_month = N / 12
    VOL_WINDOW_STEPS = max(1, round(REALIZED_VOL_WINDOW_MONTHS * steps_per_month))

    VERSION_NAME = f"{feature_set}__{rebalance_freq}"
    print(f"[configure_version] {VERSION_NAME}:  "
          f"features={FEATURE_NAMES}  N={N}  h={h:.5f}  "
          f"vol_window_steps={VOL_WINDOW_STEPS}")
    return VERSION_NAME


# Sensible default so the module doesn't crash if something below is
# imported or called before configure_version() is explicitly invoked.
configure_version(feature_set="base", rebalance_freq="monthly")


# ════════════════════════════════════════════════════════════════════
#  Feature engineering
# ════════════════════════════════════════════════════════════════════

def realized_vol_feature(S_hist: torch.Tensor) -> torch.Tensor:
    """
    Trailing annualized realized volatility, estimated only from this
    path's own simulated history up to and including the current step
    (no look-ahead). Window length is VOL_WINDOW_STEPS, set by
    configure_version() to represent a fixed calendar-time window
    regardless of N.

    S_hist : (batch, i+1) -- prices from t_0 up to and including t_i.
    """
    batch, n_obs = S_hist.shape
    if n_obs < 2:
        # No return observed yet at t_0 -- no information available.
        return torch.zeros(batch, device=DEVICE)

    window = min(VOL_WINDOW_STEPS, n_obs - 1)
    log_returns = torch.log(S_hist[:, -window:] / S_hist[:, -window - 1:-1])
    return log_returns.std(dim=1, unbiased=False) / (h ** 0.5)


def bs_delta_feature(St: torch.Tensor, sigma_hat: torch.Tensor, tau: float) -> torch.Tensor:
    """
    Black-Scholes delta computed from the *estimated* volatility
    `sigma_hat` (realized_vol_feature's output), never the true simulation
    `sigma`. Using the true value here would hand the network information
    it could never have outside a simulation study; using the estimate
    keeps the feature realistic and reframes the network's job as
    learning a correction to an already-reasonable proxy.
    """
    sigma_safe = torch.clamp(sigma_hat, min=0.01)  # guards t_0, where sigma_hat==0
    if tau <= 0:
        return (St > K).float()
    d1 = (torch.log(St / K) + (r + 0.5 * sigma_safe ** 2) * tau) / (sigma_safe * tau ** 0.5)
    return 0.5 * (1.0 + torch.erf(d1 / (2 ** 0.5)))  # standard normal CDF


def build_state(S_hist: torch.Tensor, t: int) -> torch.Tensor:
    """
    Construct the (batch, INPUT_DIM) input tensor for whichever
    FEATURE_NAMES are active (set by configure_version()).

    S_hist : (batch, t+1) -- price history up to and including step t.
    """
    batch = S_hist.shape[0]
    St = S_hist[:, -1]
    tau = 1.0 - t / N

    # Computed lazily, only if a feature that needs it is actually active.
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


# ════════════════════════════════════════════════════════════════════
#  hyperparameters
# ════════════════════════════════════════════════════════════════════

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
    def __init__(self, hp: HParams, input_dim: int):
        super().__init__()
        layers = [nn.Linear(input_dim, hp.hidden), nn.ReLU()]
        for _ in range(hp.depth - 1):
            layers += [nn.Linear(hp.hidden, hp.hidden), nn.ReLU()]
        layers += [nn.Linear(hp.hidden, 1)]
        self.net = nn.Sequential(*layers)
        self.out_act = nn.Sigmoid()  # call delta lives in [0,1]
        self.premium = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        return self.out_act(self.net(x)).squeeze(-1)


# ════════════════════════════════════════════════════════════════════
#  rollout
# ════════════════════════════════════════════════════════════════════

def run_paths(S_batch: torch.Tensor, hedging_net: HedgingNet) -> torch.Tensor:
    batch = S_batch.shape[0]
    currency = torch.zeros(batch, device=DEVICE)
    underlying = torch.zeros(batch, device=DEVICE)
    prev_delta = torch.zeros(batch, device=DEVICE)

    for t in range(N):
        S_hist = S_batch[:, :t + 1]   # only information available at t -- no look-ahead
        St = S_hist[:, -1]

        state = build_state(S_hist, t)        # (batch, INPUT_DIM)
        delta = hedging_net(state)            # (batch,)

        trade = delta - prev_delta
        currency -= trade * St
        underlying += trade
        prev_delta = delta

    S_T = S_batch[:, N]
    payoff = torch.clamp(S_T - K, min=0)
    return hedging_net.premium + underlying * S_T + currency - payoff


# ════════════════════════════════════════════════════════════════════
#  BSM benchmark (the floor for comparison -- unaffected by feature
#  engineering, but N-dependent, so it re-reads the module-level N/h set
#  by configure_version())
# ════════════════════════════════════════════════════════════════════

def bsm_call(S0, K, r, sigma, T):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bsm_delta(S, K, r, sigma, t, T):
    tau = T - t
    if tau <= 0:
        return np.where(S > K, 1.0, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * tau) / (sigma * np.sqrt(tau))
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


# ════════════════════════════════════════════════════════════════════
#  train + validate one config
# ════════════════════════════════════════════════════════════════════

def train_one_config(hp: HParams, S_train: np.ndarray, S_val: np.ndarray):
    """
    Train a single hyperparameter configuration under whichever version
    configure_version() last set, and score it on held-out paths.

    Scoring uses the standard deviation of validation P&L rather than the
    raw loss. The mean of P&L is largely controlled by the learned
    `premium` parameter regardless of hyperparameters, so std isolates how
    good the *hedging* is -- which is what you actually want to compare
    across configs.
    """
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


# ════════════════════════════════════════════════════════════════════
#  random search
# ════════════════════════════════════════════════════════════════════

DEFAULT_SEARCH_SPACE = {
    "hidden": [32, 64, 128],
    "depth": [2, 3, 4],
    # Widened upward vs. a Tanh search (see module docstring). If a
    # version's results pile up at one edge of this grid, widen it
    # further for that version specifically rather than assuming the
    # same range suits every (feature set, N) combination -- exactly the
    # lesson from the Tanh -> Sigmoid switch.
    "lr": [3e-4, 1e-3, 3e-3, 1e-2, 3e-2],
    "batch_size": [512, 1024, 2048],
    "step_size": [20, 30, 50],
    "gamma": [0.3, 0.5, 0.7],
    "clip_norm": [0.5, 1.0, 2.0],
}


def random_search(n_trials: int = 20, seed: int = 0, epochs: int = 40, space: dict = None):
    """
    Runs entirely under whichever version configure_version() last set.
    Pass a custom `space` if a given version needs a different grid (e.g.
    a daily-rebalancing version that the default grid under- or
    over-shoots for).
    """
    space = space or DEFAULT_SEARCH_SPACE
    rng = random.Random(seed)

    print(f"Simulating shared train / validation path sets for version "
          f"'{VERSION_NAME}' (N={N})...")
    S_train = generate_gbm(S0, r, sigma, h, 8_000, N + 1)
    S_val = generate_gbm(S0, r, sigma, h, 2_000, N + 1)
    bsm_std = bsm_hedge_pnl(S_val).std()
    print(f"BSM benchmark std on the validation set: {bsm_std:.4f}\n")

    results = []
    for i in range(n_trials):
        sampled = {k: rng.choice(v) for k, v in space.items()}
        hp = HParams(**sampled, epochs=epochs)

        torch.manual_seed(0)  # fix init across trials so the search isn't measuring init noise
        _, val_std, _ = train_one_config(hp, S_train, S_val)
        gap = val_std - bsm_std

        print(f"[{i + 1}/{n_trials}] val_std={val_std:.4f}  gap_vs_bsm={gap:+.4f}  {hp}")
        results.append((hp, val_std, gap))

    results.sort(key=lambda r: r[1])
    print(f"\nTop 5 configs for version '{VERSION_NAME}' by validation std:")
    for hp, val_std, gap in results[:5]:
        print(f"  val_std={val_std:.4f}  gap={gap:+.4f}  {hp}")

    return results


# ════════════════════════════════════════════════════════════════════
#  exhaustive grid search over a narrowed space
# ════════════════════════════════════════════════════════════════════

# Two version-specific grids, each narrowed from that version's *own*
# full random-search log (base__monthly's earlier 20-trial run; the
# base__daily 20-trial run pasted into chat) rather than sharing one grid
# across rebalancing frequencies. Both logs agree that lr does almost all
# the discriminating (0.01/0.03 consistently best, 3e-4 consistently
# worst) and that depth=3, hidden=128 sit in literally the best
# configuration found in both logs -- the exact same hyperparameters won
# both searches despite N changing from 12 to 252. They disagree on two
# points: daily's log showed a clear batch_size=2048 penalty at lr=3e-3
# that monthly's didn't show as strongly, and hidden=128 was even more
# dominant among daily's top results (4 of the top 5) than monthly's,
# which combined with daily costing ~21x more per epoch is reason enough
# to just fix hidden=128 there instead of spending trials confirming it.

MONTHLY_GRID_SEARCH_SPACE = {
    "hidden": [64, 128],
    "depth": [3, 4],
    "lr": [1e-2, 3e-2],
    "batch_size": [512, 1024],
    "step_size": [30],
    "gamma": [0.5],
    "clip_norm": [1.0, 2.0],
}
# 2 x 2 x 2 x 2 x 1 x 1 x 2 = 32 combinations. Cheap at N=12, so kept a
# little more open than the daily grid below.

DAILY_GRID_SEARCH_SPACE = {
    "hidden": [128],
    "depth": [3, 4],
    "lr": [1e-2, 3e-2],
    "batch_size": [512, 1024],
    "step_size": [30],
    "gamma": [0.5],
    "clip_norm": [0.5, 2.0],
}
# 1 x 2 x 2 x 2 x 1 x 1 x 2 = 16 combinations. hidden fixed at 128 (cost
# asymmetry + dominance in the log); clip_norm is {0.5, 2.0} rather than
# monthly's {1.0, 2.0} because clip=1.0 didn't make daily's top 5 at all
# (it scored 0.0036-0.0038 there, solid but a clear step behind the
# 0.0024-0.0027 of the actual top 5, all of which used 0.5 or 2.0).


def grid_search(space: dict, epochs: int = 40):
    """
    Exhaustive search: every combination in `space` is trained and scored,
    rather than a random sample of n_trials as in random_search(). Only
    sensible once the space has been narrowed enough that "every
    combination" is a small, tractable number -- see
    MONTHLY_GRID_SEARCH_SPACE / DAILY_GRID_SEARCH_SPACE above.

    `space` is required (no default) deliberately: the whole point of
    narrowing per version is that the same grid should not be silently
    reused across rebalancing frequencies, so the caller has to pick one
    explicitly each time.

    Runs entirely under whichever version configure_version() last set,
    using the same scoring (validation std vs. the BSM floor) and the
    same fixed-init-across-configs policy as random_search, so results
    from the two are directly comparable.
    """
    keys = list(space.keys())
    combos = list(itertools.product(*[space[k] for k in keys]))

    print(f"Simulating shared train / validation path sets for version "
          f"'{VERSION_NAME}' (N={N})...")
    S_train = generate_gbm(S0, r, sigma, h, 8_000, N + 1)
    S_val = generate_gbm(S0, r, sigma, h, 2_000, N + 1)
    bsm_std = bsm_hedge_pnl(S_val).std()
    print(f"BSM benchmark std on the validation set: {bsm_std:.4f}")
    print(f"Grid has {len(combos)} combinations.\n")

    results = []
    for i, combo in enumerate(combos):
        sampled = dict(zip(keys, combo))
        hp = HParams(**sampled, epochs=epochs)

        torch.manual_seed(0)  # same reasoning as random_search: isolate hyperparameter effects
        _, val_std, _ = train_one_config(hp, S_train, S_val)
        gap = val_std - bsm_std

        print(f"[{i + 1}/{len(combos)}] val_std={val_std:.4f}  gap_vs_bsm={gap:+.4f}  {hp}")
        results.append((hp, val_std, gap))

    results.sort(key=lambda r: r[1])
    print(f"\nTop 5 configs for version '{VERSION_NAME}' by validation std:")
    for hp, val_std, gap in results[:5]:
        print(f"  val_std={val_std:.4f}  gap={gap:+.4f}  {hp}")

    return results


# ════════════════════════════════════════════════════════════════════
#  optuna search (optional, more sample-efficient)
# ════════════════════════════════════════════════════════════════════

def optuna_search(n_trials: int = 30, epochs: int = 40):
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
            epochs=epochs,
        )
        torch.manual_seed(0)
        _, val_std, _ = train_one_config(hp, S_train, S_val)
        return val_std

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    print(f"Version '{VERSION_NAME}'  BSM benchmark std: {bsm_std:.4f}")
    print(f"Best val_std found: {study.best_value:.4f}")
    print("Best hyperparameters:", study.best_params)
    return study


# ════════════════════════════════════════════════════════════════════
#  robustness check for the winning config
# ════════════════════════════════════════════════════════════════════

def confirm_with_multiple_seeds(hp: HParams, n_seeds: int = 5):
    """
    Retrain the winning config under several random seeds, under whichever
    version configure_version() last set. A single search result can look
    good purely by luck -- both the network's random initialization and
    the finite validation sample add noise. This checks the win is real
    before you commit to a config.
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


# ════════════════════════════════════════════════════════════════════
#  run every (feature set, rebalancing frequency) combination
# ════════════════════════════════════════════════════════════════════

def run_all_versions(n_trials_per_version: int = 20, epochs: int = 40):
    """
    Full 2x2 grid, fully automated: both feature sets crossed with both
    rebalancing frequencies, each as an independent full random search.
    With only four versions total, there's no need for the tiered
    full-search/confirmation-sweep split that would make sense for a
    larger grid -- running all four independently is cheap enough.

    NOTE on cost: training time per epoch scales roughly linearly with N,
    so the two "daily" (N=252) versions below cost ~21x the per-epoch
    wall-clock time of the two "monthly" (N=12) versions at the same
    epochs/trials. Lower `n_trials_per_version` or `epochs` for a first
    pass if that's prohibitive, then re-run just the promising versions
    at full budget.
    """
    all_results = {}
    for feature_set in FEATURE_CONFIGS:
        for rebalance_freq in REBALANCE_CONFIGS:
            version_name = configure_version(feature_set, rebalance_freq)
            print(f"\n{'=' * 70}\nVERSION: {version_name}\n{'=' * 70}")
            all_results[version_name] = random_search(
                n_trials=n_trials_per_version, epochs=epochs
            )
    return all_results


if __name__ == "__main__":
    # ════════════════════════════════════════════════════════════════
    #  VERSIONS TO SWEEP THIS RUN.
    #  base/monthly and base/daily are both done now (see Chapter 4) --
    #  these are the two remaining, both on the extended feature set.
    #  Add/remove tuples to control what this run covers; each tuple is
    #  (feature_set, rebalance_freq).
    #
    #  COST NOTE: base_relvol_bsdelta/daily costs roughly 21x the
    #  per-epoch wall-clock time of base_relvol_bsdelta/monthly at the
    #  same epochs. If that's prohibitive in one sitting, comment out
    #  whichever entry you want to defer and run them separately.
    # ════════════════════════════════════════════════════════════════
    VERSIONS_TO_RUN = [
        ("base", "monthly"),
        ("base", "daily"),
        ("base_relvol_bsdelta", "monthly"),
        ("base_relvol_bsdelta", "daily"),
    ]
    # ════════════════════════════════════════════════════════════════

    for feature_set, rebalance_freq in VERSIONS_TO_RUN:
        configure_version(feature_set, rebalance_freq)
        print(f"\n{'=' * 70}\nVERSION: {VERSION_NAME}\n{'=' * 70}")

        # Each rebalancing frequency gets its own grid -- see the comment
        # above MONTHLY_GRID_SEARCH_SPACE / DAILY_GRID_SEARCH_SPACE for
        # why they differ. Swap to `random_search(n_trials=20)` here if
        # you'd rather sample DEFAULT_SEARCH_SPACE instead for either.
        space = MONTHLY_GRID_SEARCH_SPACE if rebalance_freq == "monthly" else DAILY_GRID_SEARCH_SPACE
        results = grid_search(space=space)
        best_hp, _, _ = results[0]

        print(f"\nConfirming the top config for '{VERSION_NAME}' across seeds...")
        confirm_with_multiple_seeds(best_hp, n_seeds=5)

    # To rerun every version from scratch (including base/monthly and
    # base/daily again) in one call instead of the loop above, use:
    # run_all_versions(n_trials_per_version=20)
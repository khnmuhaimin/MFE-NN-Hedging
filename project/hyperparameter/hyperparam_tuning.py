import dataclasses
import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import norm

DEVICE = torch.device("cpu")

# ------------------ some necessary variables and such
K, r, T = 1.0, 0.0, 1.0
MONEYNESS_RANGE = (0.85, 1.15)
SIGMA_RANGE = (0.1, 0.3)
SIGMA_FIXED = 0.2
DAILY_STEPS = 252
H_DAILY = T/DAILY_STEPS
STEP_SIZE = 30
GAMMA = 0.5
WEIGHT_DECAY = 0.0
N_TRAIN = 10_000
N_VAL = 20_000
VAL_SEED = 12345
FEATURE_CONFIGS = {
    "base": ["moneyness", "tau"],
    "base_relvol_bsdelta": ["moneyness", "tau", "realized_vol", "bs_delta_feature"],
}
REBALANCE_CONFIGS = {
    "monthly": 12,
    "daily": 252,
}
N = 12
REBAL_INDICES = list(range(0, DAILY_STEPS, DAILY_STEPS // 12))
REBAL_TAUS = [1.0 - i / DAILY_STEPS for i in REBAL_INDICES]
FEATURE_NAMES = FEATURE_CONFIGS["base"]
INPUT_DIM = 2
VERSION_NAME = "base__monthly"

# -------------------- the grids for the search
MONTHLY_GRID = {
    "hidden": [64, 128],
    "depth": [3, 4],
    "lr": [1e-2, 3e-2],
    "batch_size": [512, 1024],
    "clip_norm": [1.0, 2.0],
}

DAILY_GRID = {
    "hidden": [128],
    "depth": [3, 4],
    "lr": [1e-2, 3e-2],
    "batch_size": [512, 1024],
    "clip_norm": [0.5, 1.0],
}

# set the global config for one (feature set, rebalancing frequency) combo
def configure_version(feature_set, rebalance_freq):
    global N, REBAL_INDICES, REBAL_TAUS, FEATURE_NAMES, INPUT_DIM, VERSION_NAME

    N = REBALANCE_CONFIGS[rebalance_freq]
    stride = DAILY_STEPS // N
    REBAL_INDICES = list(range(0, DAILY_STEPS, stride))
    REBAL_TAUS = [1.0 - i / DAILY_STEPS for i in REBAL_INDICES]
    FEATURE_NAMES = FEATURE_CONFIGS[feature_set]
    INPUT_DIM = len(FEATURE_NAMES)
    VERSION_NAME = f"{feature_set}__{rebalance_freq}"
    print(f"[configure_version] {VERSION_NAME}:  features={FEATURE_NAMES}  "
          f"N={N}  rebal_stride={stride}")

    return VERSION_NAME

# simulate GBM paths with random moneyness in [0.85,1.15] and sigma in [0.1,0.3]
def make_paths(n_paths, seed=None):

    rng = np.random.default_rng(seed)
    moneynesses = rng.uniform(*MONEYNESS_RANGE, n_paths)
    sigmas = rng.uniform(*SIGMA_RANGE, n_paths)
    Z = rng.standard_normal((DAILY_STEPS, n_paths))
    W = np.cumsum(np.sqrt(H_DAILY)*Z, axis=0)
    t_grid = np.arange(1, DAILY_STEPS + 1)[:, None]*H_DAILY
    log_S = (np.log(moneynesses)[None, :]
             - 0.5*sigmas[None, :]**2*t_grid
             + sigmas[None, :]*W)
    S_full = np.concatenate([moneynesses[None, :], np.exp(log_S)], axis=0)

    return S_full, sigmas, moneynesses

# annualised realised vol from the price history so far (torch, for the network)
def realized_vol_feat(S_hist):

    batch, n_obs = S_hist.shape
    if n_obs < 2:
        return torch.zeros(batch, device=DEVICE)
    log_ret = torch.log(S_hist[:, 1:]/S_hist[:, :-1])

    return log_ret.std(dim=1, unbiased=False)/(H_DAILY**0.5)


# Black-Scholes delta from the realised-vol estimate, never the true sigma
def bs_delta_feat(St, sigma_hat, tau):

    sig = torch.clamp(sigma_hat, min=0.01)
    if tau <= 0:
        return (St > K).float()
    d1 = (torch.log(St / K) + 0.5*sig**2*tau) / (sig*tau**0.5)

    return 0.5*(1.0 + torch.erf(d1/(2**0.5)))


# stack the chosen input features into one (batch, INPUT_DIM) tensor
def make_inputs(S_hist, tau):

    batch = S_hist.shape[0]
    St = S_hist[:, -1]
    tau_t = torch.full((batch,), tau, device=DEVICE)
    if FEATURE_NAMES == ["moneyness", "tau"]:
        return torch.stack([St/K, tau_t], dim=1)
    sig = realized_vol_feat(S_hist)

    return torch.stack([St/K, tau_t, sig, bs_delta_feat(St, sig, tau)], dim=1)

# all the tunable hyperparameters for one training run
@dataclasses.dataclass
class hyper_params:
    hidden: int = 64
    depth: int = 3
    lr: float = 1e-2
    batch_size: int = 512
    epochs: int = 40
    clip_norm: float = 1.0

# feed-forward delta network; sigmoid output keeps a call delta in (0,1)
class hedger_NN(nn.Module):
    # build the layer stack from the hyperparameters
    def __init__(self, hp):
        super().__init__()
        layers = [nn.Linear(INPUT_DIM, hp.hidden), nn.ReLU()]
        for _ in range(hp.depth - 1):
            layers += [nn.Linear(hp.hidden, hp.hidden), nn.ReLU()]
        layers += [nn.Linear(hp.hidden, 1), nn.Sigmoid()]
        self.net = nn.Sequential(*layers)
        self.premium = nn.Parameter(torch.tensor(0.0))

    # forward pass: inputs -> hedge ratio (delta)
    def forward(self, x):
        return self.net(x).squeeze(-1)

# step the hedge portfolio through every rebalancing date and return terminal P&L
def roll_hedge(S_batch, net):

    batch = S_batch.shape[0]
    currency = torch.zeros(batch, device=DEVICE)
    underlying = torch.zeros(batch, device=DEVICE)
    prev_delta = torch.zeros(batch, device=DEVICE)

    for daily_idx, tau in zip(REBAL_INDICES, REBAL_TAUS):
        S_hist = S_batch[:, :daily_idx + 1]
        delta = net(make_inputs(S_hist, tau))
        trade = delta - prev_delta
        currency -= trade*S_hist[:, -1]
        underlying += trade
        prev_delta = delta

    S_T = S_batch[:, DAILY_STEPS]
    payoff = torch.clamp(S_T - K, min=0)

    return net.premium + underlying*S_T + currency - payoff

# vectorised Black-Scholes call price
def bsm_call_vec(S0_vec, sigma_vec):

    d1 = (np.log(S0_vec/K) + 0.5*sigma_vec**2*T)/(sigma_vec*np.sqrt(T))
    d2 = d1 - sigma_vec*np.sqrt(T)

    return S0_vec*norm.cdf(d1) - K*norm.cdf(d2)

# vectorised Black-Scholes delta at time t
def bsm_delta_vec(S, sigma_vec, t):

    tau = T - t
    if tau <= 0:
        return np.where(S > K, 1.0, 0.0)
    d1 = (np.log(S / K) + 0.5*sigma_vec**2*tau)/(sigma_vec*np.sqrt(tau))

    return norm.cdf(d1)

# annualised realised vol from price history (numpy, for the benchmark)
def realized_vol_np(S_hist_np):

    if S_hist_np.shape[0] < 2:
        return np.zeros(S_hist_np.shape[1])
    log_ret = np.log(S_hist_np[1:]/S_hist_np[:-1])

    return log_ret.std(axis=0, ddof=0)/(H_DAILY**0.5)

# benchmark P&L using the same info the network sees (fixed sigma, or realised vol)
def practitioner_bsm_pnl(S, moneynesses, feature_set):

    n_paths = S.shape[1]
    currency = np.zeros(n_paths)
    underlying = np.zeros(n_paths)
    prev_delta = np.zeros(n_paths)

    if feature_set == "base":
        sigma_arr = np.full(n_paths, SIGMA_FIXED)
        premium = float(bsm_call_vec(moneynesses, sigma_arr).mean())
    else:
        premium = float(bsm_call_vec(moneynesses, np.full(n_paths, SIGMA_FIXED)).mean())

    for daily_idx in REBAL_INDICES:
        if feature_set == "base":
            sig_use = sigma_arr
        else:
            rv = realized_vol_np(S[:daily_idx + 1, :])
            sig_use = np.clip(rv, 0.01, None)

        St = S[daily_idx, :]
        delta = bsm_delta_vec(St, sig_use, daily_idx * H_DAILY)
        trade = delta - prev_delta
        currency -= trade*St
        underlying += trade
        prev_delta = delta

    S_T = S[DAILY_STEPS, :]

    return premium + underlying*S_T + currency - np.maximum(S_T - K, 0)

# train one hyperparameter config; return the net and its best validation P&L std
def train_one_config(hp, train_t, val_t):

    net = hedger_NN(hp).to(DEVICE)
    opt = optim.Adam(net.parameters(), lr=hp.lr, weight_decay=WEIGHT_DECAY)
    sched = optim.lr_scheduler.StepLR(opt, step_size=STEP_SIZE, gamma=GAMMA)
    loader = DataLoader(TensorDataset(train_t), batch_size=hp.batch_size, shuffle=True)
    best_val = float("inf")
    for _ in range(hp.epochs):
        net.train()
        for (S_batch,) in loader:
            S_batch = S_batch.to(DEVICE)
            opt.zero_grad()
            loss = (roll_hedge(S_batch, net)**2).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), hp.clip_norm)
            opt.step()
        sched.step()
        net.eval()
        with torch.no_grad():
            val_std = roll_hedge(val_t, net).std().item()
        best_val = min(best_val, val_std)

    return net, best_val

# retrain the winning config across seeds to check it isn't an initialisation fluke
def confirm_with_seeds(hp, train_t, val_t, n_seeds=5):

    scores = []
    for seed in range(n_seeds):
        torch.manual_seed(seed)
        _, val_std = train_one_config(hp, train_t, val_t)
        print(f"    seed {seed}: val_std={val_std:.5f}")
        scores.append(val_std)
    arr = np.array(scores)
    print(f"    mean={arr.mean():.5f}  std_across_seeds={arr.std():.5f}")

    return arr

# run the full grid search for one version and confirm the best across seeds
def grid_search(feature_set, rebalance_freq, epochs=40, n_seeds=5):

    configure_version(feature_set, rebalance_freq)
    space = MONTHLY_GRID if rebalance_freq == "monthly" else DAILY_GRID
    keys = list(space.keys())
    combos = list(itertools.product(*[space[k] for k in keys]))
    print(f"\nGenerating train ({N_TRAIN:,}) and validation ({N_VAL:,}) sets...")
    torch.manual_seed(0)
    S_train, _, _ = make_paths(N_TRAIN, seed=0)
    S_val, _, mon_val = make_paths(N_VAL, seed=VAL_SEED)
    train_t = torch.tensor(S_train.T, dtype=torch.float32)
    val_t = torch.tensor(S_val.T, dtype=torch.float32).to(DEVICE)
    prac_std = practitioner_bsm_pnl(S_val, mon_val, feature_set).std()
    print(f"Practitioner BSM std on validation set: {prac_std:.5f}")
    print(f"Grid has {len(combos)} combinations.\n")

    results = []
    for i, combo in enumerate(combos):
        hp = hyper_params(**dict(zip(keys, combo)), epochs=epochs)
        torch.manual_seed(0)
        _, val_std = train_one_config(hp, train_t, val_t)
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
    confirm_with_seeds(best_hp, train_t, val_t, n_seeds=n_seeds)

    return results

# MAIN programme run
if __name__ == "__main__":

    VERSIONS_TO_RUN = [
        ("base", "monthly"),
        ("base", "daily"),
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
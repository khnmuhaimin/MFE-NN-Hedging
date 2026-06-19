"""Confirmation sweep: retrain the configs chosen on the single case over a range of S0/K/sigma,
to check the hyperparameter and feature-set choices still hold once volatility is no longer a known
constant. Paths are normalized by their own strike (delta is scale-invariant in (S, K) at r=0), so
the existing K=1 rollout in tune.py is reused unchanged and the only real new axis is sigma."""

import numpy as np
import torch
from scipy.stats import norm

import tune
from project.stock.generators import parameter_to_np


# ---------------------------------------------------------------------------
#  EDIT HERE
# ---------------------------------------------------------------------------

# Each entry is a scalar (held fixed) or a [low, high] range drawn uniformly per path.
# S0 and K only matter through the initial moneyness S0/K after normalization.
PARAM_RANGES = {
    "S0":    [0.8, 1.2],
    "K":     1.0,
    "sigma": [0.05, 0.30],
}

# Which versions to confirm, and the candidate configs to try for each. Paste the winners from
# tune.py, and add a higher-capacity variant or two: a wider/deeper net often only starts paying
# off once sigma varies across paths, which the single-case search can't reveal.
CONFIRM_CANDIDATES = {
    "base": [
        tune.HParams(hidden=128, depth=3, lr=3e-2, batch_size=512, clip_norm=2.0),
        tune.HParams(hidden=256, depth=4, lr=1e-2, batch_size=512, clip_norm=2.0),
    ],
    "base_relvol_bsdelta": [
        tune.HParams(hidden=128, depth=3, lr=3e-2, batch_size=512, clip_norm=2.0),
        tune.HParams(hidden=256, depth=4, lr=1e-2, batch_size=512, clip_norm=2.0),
    ],
}

REBALANCE_FREQS = ["monthly"]   # add "daily" once the monthly pass looks sensible
N_TRAIN = 8_000
N_VAL = 2_000
N_REPEATS = 3                   # re-draw the range a few times to average over sampling noise
EPOCHS = 40

# ---------------------------------------------------------------------------


def simulate_range_paths(n_paths, ranges, rng):
    """Draw per-path (S0, K, sigma) from the ranges, build GBM, normalize by K so strike == 1."""
    s0 = parameter_to_np(ranges["S0"], n_paths, rng)
    strike = parameter_to_np(ranges["K"], n_paths, rng)
    vol = parameter_to_np(ranges["sigma"], n_paths, rng)

    steps = tune.N
    Z = rng.standard_normal((steps, n_paths))
    log_returns = (tune.r - 0.5 * vol ** 2) * tune.h + vol * np.sqrt(tune.h) * Z
    cum = np.exp(np.cumsum(log_returns, axis=0))
    raw = np.vstack((np.ones((1, n_paths)), cum)) * s0

    return raw / strike, vol          # shape (N+1, n_paths) to match generate_gbm


def bs_call_norm(S0_eff, vol, T):
    """BS call price with K=1, r=0, vectorized over paths."""
    sig = np.maximum(vol, 1e-8)
    d1 = (np.log(S0_eff) + 0.5 * sig ** 2 * T) / (sig * np.sqrt(T))
    d2 = d1 - sig * np.sqrt(T)
    return S0_eff * norm.cdf(d1) - norm.cdf(d2)


def bs_delta_norm(St, vol, tau):
    """BS call delta with K=1, r=0, per-path vol."""
    if tau <= 0:
        return np.where(St > 1.0, 1.0, 0.0)
    sig = np.maximum(vol, 1e-8)
    d1 = (np.log(St) + 0.5 * sig ** 2 * tau) / (sig * np.sqrt(tau))
    return norm.cdf(d1)


def bsm_floor_range(S, vol):
    """Discrete BS-delta hedge P&L on normalized paths with per-path vol: the benchmark floor."""
    n_paths = S.shape[1]
    currency = np.zeros(n_paths)
    underlying = np.zeros(n_paths)
    prev_delta = np.zeros(n_paths)
    premium = bs_call_norm(S[0, :], vol, tune.T)

    for t in range(tune.N):
        St = S[t, :]
        delta = bs_delta_norm(St, vol, tune.T - t * tune.h)
        trade = delta - prev_delta
        currency -= trade * St
        underlying += trade
        prev_delta = delta

    S_T = S[tune.N, :]
    payoff = np.maximum(S_T - 1.0, 0.0)
    return premium + underlying * S_T + currency - payoff


def confirm_version(feature_set, rebalance_freq):
    """Score every candidate config for one version across repeated range draws."""
    tune.configure_version(feature_set, rebalance_freq)
    print(f"\n{'=' * 70}\nCONFIRM: {tune.VERSION_NAME}   ranges={PARAM_RANGES}\n{'=' * 70}")

    candidates = CONFIRM_CANDIDATES[feature_set]
    summary = []

    for hp in candidates:
        hp.epochs = EPOCHS
        val_stds, gaps = [], []

        for rep in range(N_REPEATS):
            rng = np.random.default_rng(rep)
            S_train, _ = simulate_range_paths(N_TRAIN, PARAM_RANGES, rng)
            S_val, vol_val = simulate_range_paths(N_VAL, PARAM_RANGES, rng)
            bsm_std = bsm_floor_range(S_val, vol_val).std()

            torch.manual_seed(0)
            _, val_std, _ = tune.train_one_config(hp, S_train, S_val)
            val_stds.append(val_std)
            gaps.append(val_std - bsm_std)
            print(f"  {hp.hidden}x{hp.depth} lr={hp.lr:<6} rep {rep}: "
                  f"val_std={val_std:.4f}  bsm_std={bsm_std:.4f}  gap={val_std - bsm_std:+.4f}")

        val_stds = np.array(val_stds)
        print(f"  -> mean val_std={val_stds.mean():.4f}  "
              f"mean gap={np.mean(gaps):+.4f}  spread={val_stds.std():.4f}\n")
        summary.append((hp, val_stds.mean(), float(np.mean(gaps))))

    summary.sort(key=lambda row: row[1])
    print(f"Best candidate for '{tune.VERSION_NAME}':")
    hp, mean_std, mean_gap = summary[0]
    print(f"  mean val_std={mean_std:.4f}  mean gap={mean_gap:+.4f}  {hp}")
    return summary


if __name__ == "__main__":
    all_summaries = {}
    for feature_set in CONFIRM_CANDIDATES:
        for rebalance_freq in REBALANCE_FREQS:
            all_summaries[f"{feature_set}__{rebalance_freq}"] = confirm_version(feature_set, rebalance_freq)
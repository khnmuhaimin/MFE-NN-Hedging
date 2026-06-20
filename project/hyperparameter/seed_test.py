"""Seed-robustness test: train each candidate config across several initialisations and report which
ones stay near the floor on every seed. Aimed at base_relvol_bsdelta/daily, whose tuning winner
converged for 4 of 5 seeds and diverged on the 5th. Reuses tune.py so training matches the search exactly.

Each seed gets its own (paths, init) pair, shared across candidates, so every candidate faces the same
scenarios. A candidate is flagged unstable if any seed's val_std exceeds BLOWUP_FACTOR x its own median."""

import numpy as np
import torch

from project.hyperparameter import tuning as tune

# ---- edit here ----
FEATURE_SET = "base_relvol_bsdelta"
REBALANCE = "daily"

# the current winner plus the natural robustness alternatives: looser clip, shallower net, lower lr
CANDIDATES = [
    tune.HParams(hidden=128, depth=4, lr=0.03, batch_size=512, clip_norm=0.5),   # tuning winner (fragile)
    tune.HParams(hidden=128, depth=4, lr=0.03, batch_size=512, clip_norm=2.0),
    tune.HParams(hidden=128, depth=4, lr=0.01, batch_size=512, clip_norm=0.5),
    #tune.HParams(hidden=128, depth=3, lr=0.03, batch_size=512, clip_norm=2.0),
    #tune.HParams(hidden=128, depth=4, lr=0.01, batch_size=512, clip_norm=2.0),
    #tune.HParams(hidden=128, depth=3, lr=0.01, batch_size=512, clip_norm=2.0),
]

N_SEEDS = 5
EPOCHS = 40
N_TRAIN, N_VAL = 8_000, 2_000
BLOWUP_FACTOR = 3.0
# -------------------


def make_path_sets():
    """One (train, val, bsm_floor) per seed, drawn reproducibly and shared across candidates."""
    sets = []
    for seed in range(N_SEEDS):
        rng = np.random.default_rng(seed)
        S_train = tune.generate_gbm(tune.S0, tune.r, tune.sigma, tune.h, N_TRAIN, tune.N + 1, rng)
        S_val = tune.generate_gbm(tune.S0, tune.r, tune.sigma, tune.h, N_VAL, tune.N + 1, rng)
        sets.append((S_train, S_val, tune.bsm_hedge_pnl(S_val).std()))
    return sets


def run():
    tune.configure_version(FEATURE_SET, REBALANCE)
    path_sets = make_path_sets()
    print(f"\nSeed test for '{tune.VERSION_NAME}'  ({N_SEEDS} seeds, {EPOCHS} epochs each)\n")

    summary = []
    for hp in CANDIDATES:
        hp.epochs = EPOCHS
        tag = f"h{hp.hidden} d{hp.depth} lr{hp.lr} bs{hp.batch_size} clip{hp.clip_norm}"
        print(tag)

        scores = []
        for seed in range(N_SEEDS):
            S_train, S_val, bsm_std = path_sets[seed]
            torch.manual_seed(seed)
            _, val_std, _ = tune.train_one_config(hp, S_train, S_val)
            scores.append(val_std)
            print(f"   seed {seed}: val_std={val_std:.4f}  gap={val_std - bsm_std:+.4f}")

        scores = np.array(scores)
        median = np.median(scores)
        diverged = int((scores > BLOWUP_FACTOR * median).sum())
        stable = diverged == 0
        print(f"   -> mean={scores.mean():.4f}  worst={scores.max():.4f}  "
              f"spread={scores.std():.4f}  diverged={diverged}/{N_SEEDS}  "
              f"{'STABLE' if stable else 'UNSTABLE'}\n")
        summary.append((tag, scores, stable))

    stable_ones = [(tag, s) for tag, s, ok in summary if ok]
    print("=" * 60)
    if stable_ones:
        # among configs that never diverged, prefer the lowest worst-case seed
        tag, s = min(stable_ones, key=lambda r: r[1].max())
        print(f"Most robust candidate: {tag}")
        print(f"  mean={s.mean():.4f}  worst={s.max():.4f}  spread={s.std():.4f}")
    else:
        print("No candidate was stable across all seeds. Widen CANDIDATES "
              "(lower lr, smaller hidden, or more clipping) and rerun.")
    print("=" * 60)
    return summary


if __name__ == "__main__":
    run()
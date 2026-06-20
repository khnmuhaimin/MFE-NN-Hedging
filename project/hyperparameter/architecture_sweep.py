"""Depth x width sweep with the optimizer held fixed, to see how much architecture matters and
whether one hidden layer reaches the BS floor. Monthly is the cheap scout (full width range);
daily is sampled at a trimmed band. Output: gap-to-floor vs width, one line per depth, per frequency.

Reuses tune.py so training matches the search. Each seed gets its own (paths, init), shared across
architectures, so every config faces the same scenarios; the plotted gap is the mean over seeds."""

import numpy as np
import torch
import matplotlib.pyplot as plt
import os

from project.hyperparameter import tuning as tune

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ---- edit here ----
FEATURE_SET = "base"            # the plain delta network; this is the UAT/architecture question

# optimizer fixed at the known-good region so only architecture varies
FIXED = dict(lr=0.03, batch_size=512, step_size=200, gamma=0.5, clip_norm=1.0)

#DEPTHS = [1, 2, 3, 4]
DEPTHS = [1]
WIDTHS = {
    #"monthly": [4, 8, 16, 32, 64, 128],     # cheap: run the full range
    #"daily":   [8, 16, 32, 128],            # ~21x cost: small band + one converged anchor
    "monthly": [16, 32, 64, 128],
    "daily": [64, 128],
}

EPOCHS = 150
N_SEEDS = 3                   # mean over a few inits so one bad seed doesn't spike the curve
N_TRAIN, N_VAL = 8_000, 2_000
# -------------------


def path_sets(freq):
    """One (train, val, bs_floor) per seed for the active frequency, shared across architectures."""
    tune.configure_version(FEATURE_SET, freq)
    sets = []
    for seed in range(N_SEEDS):
        rng = np.random.default_rng(seed)
        S_train = tune.generate_gbm(tune.S0, tune.r, tune.sigma, tune.h, N_TRAIN, tune.N + 1, rng)
        S_val = tune.generate_gbm(tune.S0, tune.r, tune.sigma, tune.h, N_VAL, tune.N + 1, rng)
        sets.append((S_train, S_val, tune.bsm_hedge_pnl(S_val).std()))
    return sets


def sweep(freq):
    """Train every depth x width and return mean gap-to-floor per config."""
    sets = path_sets(freq)
    floor = np.mean([s[2] for s in sets])
    print(f"\n{'=' * 60}\n{freq}  (N={tune.N})   mean BS floor std = {floor:.4f}\n{'=' * 60}")
    print(f"{'depth':>6}{'width':>7}{'gap':>10}{'spread':>9}")

    results = {d: {} for d in DEPTHS}
    for depth in DEPTHS:
        for width in WIDTHS[freq]:
            gaps = []
            for seed in range(N_SEEDS):
                S_train, S_val, bs_std = sets[seed]
                hp = tune.HParams(hidden=width, depth=depth, epochs=EPOCHS, **FIXED)
                torch.manual_seed(seed)
                _, val_std, _ = tune.train_one_config(hp, S_train, S_val)
                gaps.append(val_std - bs_std)
            gaps = np.array(gaps)
            results[depth][width] = (gaps.mean(), gaps.std())
            print(f"{depth:>6}{width:>7}{gaps.mean():>+10.4f}{gaps.std():>9.4f}")
    return results, floor


def plot(all_results):
    """One subplot per frequency: gap-to-floor vs width, a line per depth."""
    freqs = list(all_results.keys())
    fig, axes = plt.subplots(1, len(freqs), figsize=(6.5 * len(freqs), 4.8))
    if len(freqs) == 1:
        axes = [axes]

    for ax, freq in zip(axes, freqs):
        results, _ = all_results[freq]
        for depth in DEPTHS:
            widths = WIDTHS[freq]
            means = [results[depth][w][0] for w in widths]
            ax.plot(widths, means, marker="o", linewidth=1.4, label=f"depth {depth}")
        ax.axhline(0.0, color="black", linestyle=":", linewidth=1.0)
        ax.text(widths[0], 0, " BS floor", va="bottom", ha="left", fontsize=8)
        ax.set_xscale("log", base=2)
        ax.set_xticks(WIDTHS[freq])
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.set_xlabel("hidden width")
        ax.set_ylabel("validation std gap to BS floor")
        ax.set_title(f"{freq}  (N={12 if freq == 'monthly' else 252})")
        ax.legend(fontsize=9)

    plt.tight_layout()
    out = os.path.join(SCRIPT_DIR, "architecture_sweep.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nPlot saved to {out}")


if __name__ == "__main__":
    all_results = {}
    for freq in ["monthly", "daily"]:
        all_results[freq] = sweep(freq)
    plot(all_results)
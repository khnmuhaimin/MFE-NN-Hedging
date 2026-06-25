import os
import json
import torch
import numpy as np
from pricer import (
    DEVICE, TAG, N_TRAIN, SCRIPT_DIR,
    CANDIDATE_SIZES,
    generate_paths, run_paths, load_model,
)
N_BOOTSTRAP = 2_000
CONFIDENCE = 0.95
REL_TOL = 0.01
BOOTSTRAP_SEED = 99999

# run the hedging strategy on numpy paths and return P&L as a numpy array
def compute_pnl_np(S_np, net):

    S_tensor = torch.tensor(S_np.T, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        pnl = run_paths(S_tensor, net)

    return pnl.cpu().numpy()

# bootstrap 95% CI for P&L standard deviation
def bootstrap_ci(pnl):

    rng = np.random.default_rng(0)
    n = len(pnl)
    boot_std = np.array([pnl[rng.integers(0, n, n)].std() for _ in range(N_BOOTSTRAP)])
    alpha = 1 - CONFIDENCE
    lo, hi = np.percentile(boot_std, [100 * alpha / 2, 100 * (1 - alpha / 2)])

    return float(lo), float(hi), float(boot_std.mean())

# grow the path set until the bootstrap CI width is within REL_TOL
def run_bootstrap(net, set_type, seed_offset):

    print(f"\n{'='*62}")
    print(f"Bootstrap CI -- {set_type} set sizing  (target rel. half-width <= {REL_TOL:.0%})")
    print(f"{'='*62}")
    S_pool, _, _ = generate_paths(max(CANDIDATE_SIZES), seed=BOOTSTRAP_SEED + seed_offset)
    results = []
    chosen = None
    for size in CANDIDATE_SIZES:
        pnl = compute_pnl_np(S_pool[:, :size], net)
        lo, hi, boot_mean = bootstrap_ci(pnl)
        ci_half = (hi - lo) / 2
        rel_half = ci_half / boot_mean if boot_mean > 0 else float("inf")
        sufficient = rel_half <= REL_TOL
        print(f"  size={size:>7,}  pnl_std={pnl.std():.5f}  "
              f"95% CI=[{lo:.5f}, {hi:.5f}]  rel={rel_half:.3%}  "
              f"{'✓' if sufficient else ''}")
        results.append({"size": size, "pnl_std": float(pnl.std()),
                        "ci_lo": lo, "ci_hi": hi, "ci_half": ci_half, "rel_half": rel_half})
        if sufficient and chosen is None:
            chosen = size
    if chosen is None:
        chosen = max(CANDIDATE_SIZES)
        print(f"\n  WARNING: target not met -- using {chosen:,}")
    else:
        print(f"\n  Sufficient {set_type} set size: {chosen:,} paths")

    return chosen, results

# main programme run
if __name__ == "__main__":
    model_path = os.path.join(SCRIPT_DIR, "models", f"{TAG}.pt")
    net = load_model(model_path)

    chosen_val, results_val = run_bootstrap(net, set_type="validation", seed_offset=0)
    chosen_test, results_test = run_bootstrap(net, set_type="test", seed_offset=1)

    print(f"\n{'='*62}")
    print(f"  DATASET SIZE SUMMARY  ({TAG})")
    print(f"{'='*62}")
    print(f"  Training set :  {N_TRAIN:,} paths  (fixed)")
    for set_type, chosen, results in [("Validation", chosen_val, results_val),
                                       ("Test", chosen_test, results_test)]:
        row = next(r for r in results if r["size"] == chosen)
        print(f"  {set_type:12s}:  {chosen:,} paths  pnl_std={row['pnl_std']:.5f}  "
              f"95% CI=[{row['ci_lo']:.5f}, {row['ci_hi']:.5f}]  rel={row['rel_half']:.3%}")

    out_path = os.path.join(SCRIPT_DIR, f"bootstrap_{TAG}.json")
    with open(out_path, "w") as f:
        json.dump({"tag": TAG, "n_train": N_TRAIN,
                   "validation": {"chosen": chosen_val, "results": results_val},
                   "test": {"chosen": chosen_test, "results": results_test}}, f, indent=2)
    print(f"\nBootstrap results saved to {out_path}")
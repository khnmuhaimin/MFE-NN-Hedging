"""Bootstrap CI procedure for determining validation and test set sizes.

Loads a saved model and evaluates it on increasingly large path sets,
bootstrapping the 95% CI for the P&L standard deviation at each size.
Stops when the CI width falls within the target relative tolerance.

Run for validation set first, then test set (using the same model).

Usage:
    python -m project.hyperparameter.bootstrap_set_size --set-type validation
    python -m project.hyperparameter.bootstrap_set_size --set-type test
    python -m project.hyperparameter.bootstrap_set_size --set-type both
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import norm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── methodology constants (must match hedge_relvol.py / hedge_base.py) ────────
K, r, T        = 1.0, 0.0, 1.0
MONEYNESS_RANGE = (0.85, 1.15)
SIGMA_RANGE     = (0.1, 0.3)
DAILY_STEPS     = 252
H_DAILY         = T / DAILY_STEPS

# ── stopping criteria ─────────────────────────────────────────────────────────
VAL_REL_TOL    = 0.01    # 1%  relative CI half-width for validation set
TEST_REL_TOL   = 0.01   # 1% relative CI half-width for test set
CONFIDENCE     = 0.95
N_BOOTSTRAP    = 2_000   # bootstrap resamples per size estimate
SEED_EVAL      = 99999   # fixed seed for all evaluation paths

# ── candidate sizes to try (in order) ─────────────────────────────────────────
CANDIDATE_SIZES = [1_000, 2_000, 5_000, 10_000, 20_000, 50_000, 100_000]


# ── path generation ────────────────────────────────────────────────────────────

def generate_paths(n_paths, seed=None):
    rng         = np.random.default_rng(seed)
    moneynesses = rng.uniform(*MONEYNESS_RANGE, n_paths)
    sigmas      = rng.uniform(*SIGMA_RANGE, n_paths)
    Z           = rng.standard_normal((DAILY_STEPS, n_paths))
    W           = np.cumsum(np.sqrt(H_DAILY) * Z, axis=0)
    t_grid      = np.arange(1, DAILY_STEPS + 1)[:, None] * H_DAILY
    log_S       = (np.log(moneynesses)[None, :]
                   - 0.5 * sigmas[None, :] ** 2 * t_grid
                   + sigmas[None, :] * W)
    S_full      = np.concatenate([moneynesses[None, :], np.exp(log_S)], axis=0)
    return S_full, sigmas, moneynesses


# ── network and feature helpers ────────────────────────────────────────────────

class HedgingNet(nn.Module):
    def __init__(self, input_dim, hidden, depth):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden), nn.ReLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.ReLU()]
        layers += [nn.Linear(hidden, 1), nn.Sigmoid()]
        self.net     = nn.Sequential(*layers)
        self.premium = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def load_model(model_path):
    ckpt      = torch.load(model_path, map_location=DEVICE, weights_only=False)
    input_dim = ckpt["input_dim"]
    hp        = ckpt["hyperparameters"]
    net       = HedgingNet(input_dim, hp["hidden"], hp["depth"]).to(DEVICE)
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    rebal_indices = ckpt["rebal_indices"]
    rebal_taus    = [1.0 - i / DAILY_STEPS for i in rebal_indices]
    feature_set   = ckpt["feature_set"]
    print(f"Loaded model: {ckpt['tag']}  "
          f"feature_set={feature_set}  "
          f"N={len(rebal_indices)}  "
          f"premium={ckpt['premium']:.4f}")
    return net, rebal_indices, rebal_taus, feature_set


def realized_vol_torch(S_hist):
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
    St    = S_hist[:, -1]
    tau_t = torch.full((batch,), tau, device=DEVICE)
    if feature_set == "base":
        return torch.stack([St / K, tau_t], dim=1)
    else:
        sig = realized_vol_torch(S_hist)
        return torch.stack([St / K, tau_t, sig, bs_delta_feat(St, sig, tau)], dim=1)


def compute_pnl(S_np, net, rebal_indices, rebal_taus, feature_set):
    S_tensor   = torch.tensor(S_np.T, dtype=torch.float32).to(DEVICE)
    batch      = S_tensor.shape[0]
    currency   = torch.zeros(batch, device=DEVICE)
    underlying = torch.zeros(batch, device=DEVICE)
    prev_delta = torch.zeros(batch, device=DEVICE)

    with torch.no_grad():
        for daily_idx, tau in zip(rebal_indices, rebal_taus):
            S_hist = S_tensor[:, :daily_idx + 1]
            delta  = net(build_state(S_hist, tau, feature_set))
            trade  = delta - prev_delta
            currency   -= trade * S_hist[:, -1]
            underlying += trade
            prev_delta  = delta

        S_T    = S_tensor[:, DAILY_STEPS]
        payoff = torch.clamp(S_T - K, min=0)
        pnl    = net.premium + underlying * S_T + currency - payoff

    return pnl.cpu().numpy()


# ── bootstrap CI ──────────────────────────────────────────────────────────────

def bootstrap_ci(pnl, n_bootstrap, confidence):
    """Bootstrap 95% CI for P&L standard deviation."""
    rng      = np.random.default_rng(0)
    n        = len(pnl)
    boot_std = np.array([
        pnl[rng.integers(0, n, n)].std()
        for _ in range(n_bootstrap)
    ])
    alpha    = 1 - confidence
    lo, hi   = np.percentile(boot_std, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi), float(boot_std.mean())


# ── main procedure ─────────────────────────────────────────────────────────────

def find_sufficient_size(net, rebal_indices, rebal_taus, feature_set,
                         rel_tol, set_type, seed_offset=0):
    """Grow the evaluation set until the bootstrap CI width is within rel_tol."""
    print(f"\n{'=' * 60}")
    print(f"Finding sufficient {set_type} set size  (rel_tol={rel_tol:.1%})")
    print(f"{'=' * 60}")

    # generate the largest pool once and slice it
    max_size  = max(CANDIDATE_SIZES)
    seed      = SEED_EVAL + seed_offset
    S_pool, sigmas_pool, mon_pool = generate_paths(max_size, seed=seed)

    results = []
    chosen  = None

    for size in CANDIDATE_SIZES:
        S      = S_pool[:, :size]
        pnl    = compute_pnl(S, net, rebal_indices, rebal_taus, feature_set)
        lo, hi, boot_mean = bootstrap_ci(pnl, N_BOOTSTRAP, CONFIDENCE)
        ci_half  = (hi - lo) / 2
        rel_half = ci_half / boot_mean if boot_mean > 0 else float("inf")

        print(f"  size={size:>7,}  pnl_std={pnl.std():.5f}  "
              f"boot_mean={boot_mean:.5f}  "
              f"95% CI=[{lo:.5f}, {hi:.5f}]  "
              f"half_width={ci_half:.5f}  rel={rel_half:.3%}  "
              f"{'✓ SUFFICIENT' if rel_half <= rel_tol else ''}")

        results.append({
            "size":      size,
            "pnl_std":   float(pnl.std()),
            "boot_mean": boot_mean,
            "ci_lo":     lo,
            "ci_hi":     hi,
            "ci_half":   ci_half,
            "rel_half":  rel_half,
        })

        if rel_half <= rel_tol and chosen is None:
            chosen = size

    if chosen is None:
        chosen = max(CANDIDATE_SIZES)
        print(f"\n  WARNING: CI never reached target within tested sizes.")
        print(f"  Using largest size ({chosen:,}) — consider extending CANDIDATE_SIZES.")
    else:
        print(f"\n  Chosen {set_type} set size: {chosen:,} paths")

    return chosen, results


def run(model_path, set_type):
    net, rebal_indices, rebal_taus, feature_set = load_model(model_path)

    summaries = {}

    if set_type in ("validation", "both"):
        chosen_val, results_val = find_sufficient_size(
            net, rebal_indices, rebal_taus, feature_set,
            rel_tol=VAL_REL_TOL, set_type="validation", seed_offset=0
        )
        summaries["validation"] = {
            "chosen_size": chosen_val,
            "rel_tol":     VAL_REL_TOL,
            "confidence":  CONFIDENCE,
            "results":     results_val,
        }

    if set_type in ("test", "both"):
        chosen_test, results_test = find_sufficient_size(
            net, rebal_indices, rebal_taus, feature_set,
            rel_tol=TEST_REL_TOL, set_type="test", seed_offset=1
        )
        summaries["test"] = {
            "chosen_size": chosen_test,
            "rel_tol":     TEST_REL_TOL,
            "confidence":  CONFIDENCE,
            "results":     results_test,
        }

    out_dir = Path("results/set_sizes")
    out_dir.mkdir(parents=True, exist_ok=True)
    tag     = Path(model_path).stem
    out_path = out_dir / f"{tag}_{set_type}.json"
    with open(out_path, "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"\nResults saved to {out_path}")

    print(f"\n{'=' * 60}")
    print(f"Summary")
    print(f"{'=' * 60}")
    if "validation" in summaries:
        print(f"  Validation set: {summaries['validation']['chosen_size']:,} paths")
    if "test" in summaries:
        print(f"  Test set:       {summaries['test']['chosen_size']:,} paths")


def parse_args():
    parser = argparse.ArgumentParser(description="Bootstrap CI for set size determination")
    parser.add_argument("--model-path", default=None,
                        help="Path to .pt model file. Defaults to relvol_monthly.pt "
                             "in the models/ folder next to this script.")
    parser.add_argument("--set-type", default="both",
                        choices=["validation", "test", "both"])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.model_path is None:
        # default: look for relvol_monthly.pt next to this script
        script_dir = Path(__file__).parent
        model_path = script_dir / "models" / "relvol_monthly.pt"
    else:
        model_path = Path(args.model_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. "
            f"Either train it first or pass --model-path explicitly."
        )

    run(str(model_path), args.set_type)
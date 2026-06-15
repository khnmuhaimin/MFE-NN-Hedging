"""
CVaR Multi-Kappa Training
Trains one CVaR model per kappa and saves each independently.
Run this before CVaREvaluate.py.
"""

import sys
import pathlib

_root = pathlib.Path(__file__).resolve()
while not (_root / "pyproject.toml").exists():
    _root = _root.parent
sys.path.insert(0, str(_root))
sys.path.insert(0, str(pathlib.Path(__file__).parent))

import numpy as np
from project.stock.generators import generate_gbm
from pricer_cvar import train, save_model, S0, r, sigma, h, N, N_PATHS_TRAIN, ALPHA, LAMBDA_MEAN

KAPPAS = [0.0, 0.001, 0.005, 0.01, 0.02]

if __name__ == "__main__":
    print(f"CVaR training — α={ALPHA}, λ_mean={LAMBDA_MEAN}")
    for kappa in KAPPAS:
        print(f"\n{'='*63}")
        print(f"  Training κ = {kappa}  (CVaR α={ALPHA}, λ={LAMBDA_MEAN})")
        print(f"{'='*63}")

        S_train = generate_gbm(S0, r, sigma, h, N_PATHS_TRAIN, N + 1)
        hedging_net, epoch_losses = train(S_train, kappa=kappa, alpha=ALPHA,
                                          lambda_mean=LAMBDA_MEAN)
        save_model(hedging_net, epoch_losses, kappa=kappa, alpha=ALPHA)

    print(f"\n{'='*63}")
    print(f"  All CVaR models trained and saved.")
    print(f"{'='*63}")

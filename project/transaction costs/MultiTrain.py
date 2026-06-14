"""
Multi-Kappa Training
Trains one model per kappa value and saves each independently.
Run this before MultiEvaluate.py.
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
from pricer import train, save_model, S0, r, sigma, h, N, N_PATHS_TRAIN

KAPPAS = [0.0, 0.001, 0.005, 0.01, 0.02]

if __name__ == "__main__":
    for kappa in KAPPAS:
        print(f"\n{'='*63}")
        print(f"  Training κ = {kappa}")
        print(f"{'='*63}")

        S_train = generate_gbm(S0, r, sigma, h, N_PATHS_TRAIN, N + 1)
        hedging_net, epoch_losses = train(S_train, kappa=kappa)
        save_model(hedging_net, epoch_losses, kappa=kappa)

    print(f"\n{'='*63}")
    print(f"  All models trained and saved.")
    print(f"{'='*63}")

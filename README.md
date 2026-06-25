# Constructing Hedge Portfolios with Neural Networks

**MFE Research Project — Group 2, June 2026**

Authors: Alande Reve (RVXALA001), Muhaimin Khan (KHNMUH063), Ralf Louw (LWXRUD001)

---

## Overview

This project implements a neural network system that simultaneously learns an option premium and a discrete-time delta hedging strategy for a European call option from simulated asset-price data. The system is evaluated against a practitioner Black-Scholes benchmark under identical information constraints.

Two input configurations are compared across two rebalancing frequencies, yielding four model versions:

| Version | Inputs | Rebalancing |
|---------|--------|-------------|
| V1 (`base_monthly`) | Moneyness, time to maturity | Monthly (N=12) |
| V2 (`base_daily`) | Moneyness, time to maturity | Daily (N=252) |
| V3 (`base_relvol_bsdelta_monthly`) | Moneyness, time to maturity, realised volatility, BS delta | Monthly (N=12) |
| V4 (`base_relvol_bsdelta_daily`) | Moneyness, time to maturity, realised volatility, BS delta | Daily (N=252) |

All four configurations outperform the Black-Scholes benchmark, with V4 (4-input daily) achieving the best overall performance — roughly a threefold reduction in P&L standard deviation relative to the BS benchmark.

## Repository Structure

```
MFE-NN-Hedging/
├── project/
│   ├── minimal/                  # Core model constants, hyperparameters, and BS benchmark
│   │   ├── constants.py          # All hyperparameters and simulation parameters
│   │   ├── model_name.py         # Select which model version to train
│   │   └── bs_model.py           # Black-Scholes analytical benchmark
│   ├── minimal2/                 # Main training and evaluation entry point
│   │   ├── new_pricer.py         # Train and evaluate the full dual-network system
│   │   └── model.py              # HedgingNet2 architecture (hedge + premium networks)
│   ├── hyperparameter/           # Hyperparameter tuning scripts
│   │   ├── hyperparam_tuning.py  # Grid search over architecture and optimisation params
│   │   ├── trainingSet_sweep.py  # Training set size sweep
│   │   └── validation_bootstrap.py # Bootstrap CI for validation/test set sizing
│   ├── stock/
│   │   └── generators.py         # GBM path generation (and Heston/Bates extensions)
│   ├── helpers/
│   │   ├── helpers.py            # Device selection and general utilities
│   │   └── path_helpers.py       # File path utilities
│   └── transaction costs/        # Experimental extension: TC-aware hedging with CVaR/entropic loss
├── results/
│   ├── figures/                  # Output plots (P&L distributions, delta curves, forest plots, etc.)
│   ├── models/                   # Saved model weights (.pt)
│   ├── logs/                     # Training logs per model version
│   ├── hyperparameter/           # Grid search results
│   ├── dataset_size/             # Training set size sweep results
│   └── set_sizes/                # Bootstrap CI results for val/test sizing
├── requirements.txt
└── README.md
```

## Installation

Python 3.12 is recommended. Install dependencies with:

```bash
pip install -r requirements.txt
```

Key dependencies: `torch==2.11.0`, `numpy==2.4.4`, `scipy==1.17.1`, `matplotlib==3.10.8`.

## Usage

### 1. Select the model version

Edit `project/minimal/model_name.py` and uncomment the desired version:

```python
MODEL_NAME = "base_monthly"
# MODEL_NAME = "base_daily"
# MODEL_NAME = "base_relvol_bsdelta_monthly"
# MODEL_NAME = "base_relvol_bsdelta_daily"
```

### 2. Train and evaluate

Run the main pricer from the repository root:

```bash
python -m project.minimal2.new_pricer
```

This trains the hedge and premium networks jointly, evaluates against the practitioner BS benchmark on a held-out test set, and writes figures to `results/figures/` and model weights to `results/models/`.

### 3. Hyperparameter tuning

To reproduce the grid search:

```bash
python -m project.hyperparameter.hyperparam_tuning
```

To reproduce the training set size sweep:

```bash
python -m project.hyperparameter.trainingSet_sweep
```

To reproduce the bootstrap CI for validation/test set sizing:

```bash
python -m project.hyperparameter.validation_bootstrap
```

## Model Architecture

The system comprises two feedforward networks trained jointly under an MSE loss on terminal P&L:

- **Hedge network** — takes 2 or 4 input features at each rebalancing date and outputs a hedge ratio δ̂ ∈ (0,1) via a sigmoid output layer. The same network is applied at every rebalancing date (weight sharing / BPTT). Hidden layers use ReLU activations.
- **Premium network** — takes initial moneyness S₀/K as its sole input and outputs a per-path premium π. A linear output layer is used; the network is called once per path with no BPTT.

The MSE-optimal premium converges to the Black-Scholes risk-neutral price (see Section 3.7 of the report).

## Simulation Parameters

| Parameter | Value |
|-----------|-------|
| Initial moneyness S₀/K | Uniform on [0.8, 1.2] |
| Effective volatility σ_eff | Uniform on [0.1, 0.3] |
| Maturity T | 1 year (fixed) |
| Risk-free rate r | 0 |
| Training paths | 10,000 |
| Validation paths | 20,000 |
| Test paths | 20,000 |

## Results Summary

| Strategy | Mean P&L | Std P&L |
|----------|----------|---------|
| 4-Input Daily (V4) | −0.16 | **1.98** |
| 4-Input Monthly (V3) | −0.15 | 2.72 |
| 2-Input Daily (V2) | −0.17 | 2.33 |
| 2-Input Monthly (V1) | −0.75 | 2.86 |
| BS Hedge Daily | +2.19 | 7.92 |
| BS Hedge Monthly | −1.92 | 7.40 |

P&L is expressed as a percentage of the strike. The neural network strategies achieve roughly a **threefold reduction** in P&L standard deviation relative to the BS benchmark across all moneyness buckets (OTM, ATM, ITM).

## Citation

If referencing this work, please cite:

> Reve, A., Khan, M., & Louw, R. (2026). *Constructing Hedge Portfolios with Neural Networks*. MFE Research Project, African Institute for Financial Markets and Risk Management (AIFMRM).

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
│   └── stock/
│       └── generators.py         # GBM path generation (and Heston/Bates extensions)
├── requirements.txt
└── README.md
```

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

This trains the hedge and premium networks jointly and evaluates against the practitioner BS benchmark on a held-out test set.

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


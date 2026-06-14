# Transaction Costs Extension

## Overview

This extension modifies the minimal deep hedging pricer to operate under **proportional transaction costs**. At each rebalancing step the hedger pays `κ * |trade| * S_t` in friction costs in addition to the cost of the shares traded. The neural network is trained knowing these costs exist, allowing it to discover a less aggressive hedging strategy that minimises mean-squared terminal P&L including transaction cost drag.

The key insight: BSM delta hedging was derived assuming a frictionless market. When costs are present, blindly following the BSM delta becomes suboptimal. The NN learns this purely from the structure of the loss function — no theoretical knowledge of the optimal strategy is required.

## Files

| File | Purpose |
|------|---------|
| `pricer.py` | Simulate paths, train network, save model and epoch losses |
| `evaluate.py` | Load trained model, run all tests, generate all plots |
| `README.md` | This file |

**Run order:**
```
python "project/transaction costs/pricer.py"    # train
python "project/transaction costs/evaluate.py"  # evaluate
```
Both must be run from the project root.

## Model Setup

| Parameter | Symbol | Value |
|-----------|--------|-------|
| Initial stock price | S0 | 1.0 |
| Strike | K | 1.0 |
| Volatility | σ | 0.1 |
| Risk-free rate | r | 0.0 |
| Maturity | T | 1.0 |
| Time steps | N | 100 |
| Transaction cost rate | κ | 0.001 |

## Neural Network Architecture

| Component | Detail |
|-----------|--------|
| Inputs | Moneyness (S_t / K), time-to-maturity (τ = 1 − t/N) |
| Hidden layers | 3 × 64 neurons, ReLU activation |
| Output | 1 neuron, **Sigmoid** → delta ∈ (0, 1) |
| Learnable parameter | Scalar premium π |

The Sigmoid output layer enforces delta ∈ (0, 1), which is theoretically correct for a European call option and prevents the network from taking unintended short stock positions.

## Training Hyperparameters

| Hyperparameter | Value |
|----------------|-------|
| Training paths | 10,000 |
| Test paths | 10,000 |
| Batch size | 2,048 |
| Epochs | 100 |
| Optimiser | Adam (lr = 1e-3) |
| LR scheduler | StepLR (step=30, γ=0.5) |
| Loss function | Mean-squared terminal P&L |

---

## Results

> Update this section after each training run.

### Run 1 — Tanh activation, κ = 0.001

#### P&L Summary (10,000 test paths)

| Metric | NN (κ=0.001) | BSM (κ=0.001) | BSM (no TC) |
|--------|:------------:|:-------------:|:-----------:|
| Mean P&L | -0.0001 | -0.0038 | -0.0001 |
| Std P&L | 0.0052 | 0.0038 | 0.0035 |
| P1 | -0.0122 | -0.0153 | -0.0096 |
| P5 | -0.0081 | -0.0105 | -0.0058 |
| P25 | -0.0033 | -0.0057 | -0.0020 |
| P75 | 0.0026 | -0.0013 | 0.0020 |
| P95 | 0.0098 | 0.0015 | 0.0054 |
| P99 | 0.0144 | 0.0042 | 0.0088 |

BSM theoretical price: 0.0399 | Learned premium: 0.0434

#### Key Findings

1. **NN dramatically outperforms BSM under TC.** The BSM hedger loses on average −0.0038 per path due to transaction cost drag from blind rebalancing. The NN reduces this to −0.0001, nearly eliminating the TC burden.

2. **The NN learns to trade less aggressively.** By penalising each rebalance with `κ * |trade| * S_t` in the loss, the network discovers that trading less frequently reduces TC drag. The tradeoff is slightly wider P&L variance (0.0052 vs 0.0035 for frictionless BSM).

3. **The NN charges a higher premium.** The learned premium (0.0434) exceeds the BSM price (0.0399), reflecting the expected TC burden the hedger will incur over the option's life.

4. **BSM with TC has negative P&L at P75.** More than 75% of paths result in a loss for the naive BSM hedger — TC drag is systematic and substantial relative to option value at σ = 0.1.

5. **Anomaly: negative delta on OTM paths (Tanh activation).** The Tanh output allows delta < 0, which is not theoretically valid for a European call. The network occasionally went short the stock near expiry on deep OTM paths, likely as a speculative attempt to recover accumulated TC losses. **Fixed in Run 2 by switching to Sigmoid.**

---

### Run 2 — Sigmoid activation, κ = 0.001

#### P&L Summary (10,000 test paths)

| Metric | NN (κ=0.001) | BSM (κ=0.001) | BSM (no TC) |
|--------|:------------:|:-------------:|:-----------:|
| Mean P&L | -0.0000 | -0.0037 | -0.0000 |
| Std P&L | 0.0051 | 0.0037 | 0.0035 |
| P1 | -0.0123 | -0.0147 | -0.0092 |
| P5 | -0.0083 | -0.0105 | -0.0057 |
| P25 | -0.0032 | -0.0057 | -0.0020 |
| P75 | 0.0028 | -0.0013 | 0.0020 |
| P95 | 0.0092 | 0.0016 | 0.0057 |
| P99 | 0.0134 | 0.0044 | 0.0089 |

BSM theoretical price: 0.0399 | Learned premium: 0.0436

Total TC paid — NN: 0.003674 | BSM: 0.003692 | Reduction: **0.5%**

#### Key Findings

1. **Sigmoid fix confirmed.** Delta stays within (0, 1) on all paths including deep OTM near expiry. The anomalous short stock positions seen in Run 1 are eliminated.

2. **At κ=0.001, the NN does not meaningfully change its trading strategy.** Total TC paid is virtually identical to BSM (0.5% reduction). The trade size plot confirms this — NN and BSM rebalance at nearly the same frequency and magnitude throughout the path.

3. **The mean P&L improvement over BSM is driven by premium adjustment, not strategy change.** The NN learns to charge a higher premium (0.0436 vs BSM 0.0399) that approximately covers the expected TC burden (~0.0037). The BSM hedger charges the frictionless price and so absorbs the TC as a loss. At this κ level, adjusting the price is cheaper than adjusting the hedge.

4. **At κ=0.001, TC are too small to alter optimal hedging behaviour.** The cost of accepting wider P&L variance (from hedging less precisely) outweighs the TC savings. This implies there is a threshold κ above which the NN will meaningfully depart from BSM delta hedging.

5. **This motivates kappa sensitivity analysis** — the natural next experiment is to run at increasing κ values and observe at what point the network begins to genuinely trade less aggressively. This is the core research contribution of this extension.

---

---

## Multi-Kappa Sensitivity Analysis

Trained and evaluated separate models at κ ∈ {0.0, 0.001, 0.005, 0.01, 0.02} using `MultiTrain.py` and `MultiEvaluate.py`. All models use the same architecture and hyperparameters; the same 10,000 test paths are used for fair comparison across κ values.

### Summary Table

| κ | NN Mean P&L | NN Std P&L | BSM Mean P&L | BSM Std P&L | TC Reduction | NN Premium |
|---|:-----------:|:----------:|:------------:|:-----------:|:------------:|:----------:|
| 0.0   |  0.0000 | 0.0061 |  0.0000 | 0.0035 |   0.0% | 0.0399 |
| 0.001 |  0.0001 | 0.0063 | -0.0037 | 0.0038 |  -2.3% | 0.0437 |
| 0.005 |  0.0000 | 0.0055 | -0.0185 | 0.0071 |  11.8% | 0.0561 |
| 0.01  |  0.0002 | 0.0068 | -0.0369 | 0.0125 |  20.0% | 0.0695 |
| 0.02  | -0.0008 | 0.0081 | -0.0738 | 0.0240 |  35.4% | 0.0866 |

### Key Findings

1. **Two distinct regimes separated by a threshold near κ ≈ 0.005.**

   - **Low TC (κ ≤ 0.001):** The NN mirrors BSM delta hedging almost exactly and adapts purely through premium adjustment — charging a higher premium to cover expected TC while maintaining the same trading frequency. TC reduction is essentially zero.
   - **High TC (κ ≥ 0.005):** The NN genuinely alters its trading strategy. Delta paths become smoother and slower-moving, reflecting deliberate avoidance of rapid rebalancing. TC reduction rises from 12% to 35% as κ increases.

2. **The delta path comparison is the most compelling visual evidence.** At κ=0.02, the NN delta path is dramatically smoother than BSM — the network learns to hold positions longer rather than tracking the theoretical delta at every step. At κ=0.001 the two paths are nearly indistinguishable.

3. **BSM mean P&L deteriorates linearly with κ** (−0.0037, −0.0185, −0.0369, −0.0738), exactly doubling when κ doubles. This is expected: BSM pays TC proportional to κ without adjusting its strategy. The NN maintains near-zero mean P&L across all κ.

4. **At high κ, the NN dominates BSM on both mean and variance.** At κ=0.02 the NN achieves std 0.0081 vs BSM's 0.0240 — a 3× improvement in P&L stability. The NN does not merely accept a variance penalty to save costs; it genuinely reduces risk as well.

5. **The NN autonomously prices in TC through the premium.** At κ=0.0 the learned premium equals the BSM price (0.0399). As κ rises, the premium increases to cover the expected TC burden (0.0866 at κ=0.02), with no explicit instruction to do so.

6. **The −2.3% TC reduction at κ=0.001 is a subtle artefact.** At very low TC the NN pays marginally more in costs than BSM. This is likely because the Sigmoid activation initialises at 0.5, forcing a larger initial trade than the BSM delta at t=0. The effect is small and disappears at higher κ.

7. **κ=0.02 may benefit from more training.** The learning curve for κ=0.02 was still declining at epoch 100, suggesting the model had not fully converged. The slight negative mean P&L (−0.0008) at this level may improve with additional epochs.

---

## Planned Extensions / TODO

- [x] Kappa sensitivity: train and evaluate at κ ∈ {0.0, 0.001, 0.005, 0.01, 0.02}
- [x] Quantify TC reduction: total TC paid per path comparison (NN vs BSM)
- [x] Trade size plots: |Δδ| per step to visualise reduced rebalancing
- [ ] Retrain κ=0.02 with more epochs (e.g. 200) to confirm convergence
- [ ] Heston model: swap GBM paths for Heston paths using existing generator in project/stock/generators.py
- [ ] CVaR loss: replace MSE with conditional value-at-risk to optimise the left tail directly
- [ ] Increase N_PATHS_TRAIN for better generalisation at high κ

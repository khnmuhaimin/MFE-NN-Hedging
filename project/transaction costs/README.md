# Transaction Costs Extension

## Overview

This extension modifies the minimal deep hedging pricer to operate under **proportional transaction costs**. At each rebalancing step the hedger pays `κ * |trade| * S_t` in friction costs in addition to the cost of the shares traded. The neural network is trained knowing these costs exist, allowing it to discover a less aggressive hedging strategy that minimises mean-squared terminal P&L including transaction cost drag.

The key insight: BSM delta hedging was derived assuming a frictionless market. When costs are present, blindly following the BSM delta becomes suboptimal. The NN learns this purely from the structure of the loss function — no theoretical knowledge of the optimal strategy is required.

## Files

| File | Purpose |
|------|---------|
| `pricer.py` | Train MSE model (single κ), save to `results/transaction costs/models/` |
| `evaluate.py` | Load MSE model, run all tests and plots (single κ) |
| `MultiTrain.py` | Train MSE models across κ ∈ {0.0, 0.001, 0.005, 0.01, 0.02} |
| `MultiEvaluate.py` | Load all MSE models, produce multi-κ comparison plots |
| `pricer_cvar.py` | Train CVaR model (single κ), save with `_cvar` suffix |
| `CVaRTrain.py` | Train CVaR models across all κ values |
| `CVaREvaluate.py` | Load MSE + CVaR models, produce side-by-side tail risk comparison |
| `pricer_entropic.py` | Train Entropic Risk Measure model (single κ), save with `_entropic` suffix |
| `EntropicTrain.py` | Train entropic models across all κ values |
| `CompareEvaluate.py` | 3-way comparison: MSE vs CVaR vs Entropic across all κ |
| `README.md` | This file |

**Run order (MSE):**
```
python "project/transaction costs/MultiTrain.py"
python "project/transaction costs/MultiEvaluate.py"
```

**Run order (3-way loss function comparison):**
```
python "project/transaction costs/CVaRTrain.py"
python "project/transaction costs/EntropicTrain.py"
python "project/transaction costs/CompareEvaluate.py"
```
All scripts must be run from the project root.

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

---

## CVaR Loss Extension

### Motivation

MSE loss minimises `E[pnl²] = Var[pnl] + E[pnl]²`, treating all paths equally. An alternative is **Conditional Value-at-Risk (CVaR / Expected Shortfall)**, which targets only the worst (1−α) fraction of outcomes. A CVaR-trained hedger should theoretically tolerate wider variance on average in order to reduce catastrophic tail losses.

### Implementation Notes

CVaR loss uses the Rockafellar-Uryasev (2000) formula:

```
CVaR_α(losses) = z + 1/(1−α) * E[max(losses − z, 0)]
```

where `losses = −pnl` and `z` is the α-quantile of losses (the VaR estimate).

**Key implementation challenge — premium anchoring:**
Unlike MSE (where `E[pnl²]` is minimised at `E[pnl]=0` automatically), the CVaR gradient with respect to the premium is always −1: raising the premium improves every path's P&L, which always reduces CVaR. Without a constraint, gradient descent drives the premium to +∞.

The fix uses gradient routing: the CVaR gradient is *blocked* from reaching the premium by detaching it within the CVaR computation. A separate zero-mean penalty (`λ * E[pnl]²`) anchors the premium independently. This cleanly separates the two objectives:
- **Delta parameters**: updated purely by CVaR (tail optimisation)
- **Premium**: updated purely by `E[pnl]²` (fair pricing)

### CVaR vs MSE Results (α = 0.95, κ sensitivity)

All models share the same architecture and training setup. Test paths are held fixed for fair comparison. VaR and Expected Shortfall (ES) are computed as quantiles of losses (i.e. −P&L), so lower is better.

| κ | MSE Mean | CVaR Mean | MSE ES | CVaR ES | ES Improvement |
|---|:--------:|:---------:|:------:|:-------:|:--------------:|
| 0.0   | 0.0000 | 0.0001 | 0.0120 | 0.0109 | +9% |
| 0.001 | 0.0001 | 0.0000 | 0.0123 | 0.0109 | +11% |
| 0.005 | 0.0000 | 0.0001 | 0.0114 | 0.0106 | +7% |
| 0.01  | 0.0001 | −0.0001 | 0.0134 | 0.0127 | +5% |
| 0.02  | −0.0009 | −0.0001 | 0.0167 | 0.0183 | **−10%** |

### Key Findings

1. **CVaR improves tail risk at low-to-medium TC (κ ≤ 0.01).** Expected Shortfall is reduced by 5–11% relative to MSE. Both hedgers charge fair premiums (mean P&L ≈ 0), so the improvement is purely from better tail-shaping of the hedging strategy.

2. **CVaR reverses at high TC (κ = 0.02), where MSE produces better tail outcomes.** The CVaR model aggressively under-hedges at κ=0.02 (lower delta throughout the path) to avoid paying transaction costs in bad scenarios. But this leaves unhedged exposure that creates larger tail losses when the stock moves sharply. The TC saving is outweighed by unhedged risk.

3. **There is a TC threshold beyond which CVaR tail optimisation is counterproductive.** At low κ the dominant source of tail losses is hedging error; CVaR correctly reduces this. At high κ, TC drag becomes the dominant risk and CVaR avoids it by under-hedging — but unhedged exposure in a directional move is worse.

4. **CVaR models have wider P&L variance than MSE across all κ.** MSE directly minimises variance (since `E[pnl²] = Var + Mean²`); CVaR does not penalise variance on paths outside the tail. The wider variance is the cost paid for better tail outcomes at moderate TC levels.

5. **Both loss functions price fairly.** After applying gradient routing, both MSE and CVaR premiums settle at values giving `E[pnl] ≈ 0`. The comparison is therefore purely about hedging strategy, not premium differences.

---

## Entropic Risk Measure Extension and 3-Way Comparison

### Motivation

Having established that CVaR α=0.95 improves tail risk at moderate TC but deteriorates at κ=0.02, the natural question is whether a different tail-focused loss function avoids this breakdown. The **Entropic Risk Measure (ERM)** — also known as the certainty equivalent under CARA exponential utility — is theoretically smoother than CVaR because it weights every path continuously by how bad it is, rather than applying a binary weight to the worst α fraction:

```
ER_γ(pnl) = (1/γ) * log( E[ exp(−γ * pnl) ] )
```

The intuition: ERM assigns gradient weight `exp(−γ * pnl)` to each path, so bad paths receive exponentially more influence than good ones. γ is the risk aversion parameter (γ→0: risk-neutral; γ→∞: worst-case).

The same gradient routing fix applies: d(ER)/dπ = −1 always, so the premium is detached from the ERM gradient and anchored via a zero-mean penalty independently.

### 3-Way Comparison Results (γ = 100, α = 0.95)

| κ | MSE ES | CVaR ES | Entropic ES | Winner |
|---|:------:|:-------:|:-----------:|:------:|
| 0.0   | 0.0120 | **0.0110** | 0.0137 | CVaR |
| 0.001 | 0.0124 | **0.0111** | 0.0153 | CVaR |
| 0.005 | 0.0116 | **0.0108** | 0.0156 | CVaR |
| 0.01  | 0.0135 | **0.0128** | 0.0224 | CVaR |
| 0.02  | **0.0170** | 0.0184 | 0.0367 | MSE  |

ES = Expected Shortfall at α = 0.95 (lower is better).

### Key Findings

1. **CVaR α=0.95 is the best-performing loss function across κ ≤ 0.01.** It reduces Expected Shortfall by 5–11% relative to MSE. Entropic γ=100 is the worst performer at every single κ value — including κ=0 where there are no transaction costs at all.

2. **Entropic γ=100 is more extreme than CVaR α=0.95, not less.** The intuition that ERM would be smoother than CVaR proved incorrect at this γ. CVaR at α=0.95 gives the worst 5% of paths a weight of 1/(1−0.95) = 20×. The ERM at γ=100 gives a path with pnl=−0.10 a gradient weight of exp(100 × 0.10) = exp(10) ≈ 22,000× relative to a zero-pnl path. On rare extreme paths, the ERM gradient is several orders of magnitude larger than CVaR, causing far more aggressive under-hedging.

3. **The result reveals a non-monotonic relationship between tail-focus intensity and actual tail performance.** Ranking the three loss functions by tail-focus intensity: MSE (none) < CVaR α=0.95 (moderate) < Entropic γ=100 (extreme). But ranking by actual ES at κ=0.005: CVaR (0.0108) < MSE (0.0116) < Entropic (0.0156). The best tail outcome comes from moderate tail focus — too much is as harmful as too little.

4. **Entropic under-hedges more aggressively than CVaR at high κ.** The delta path comparison shows Entropic (green) consistently below CVaR (orange) at κ ≥ 0.005, and far below BSM. The exponential gradient weighting causes the network to abandon hedging even more drastically than CVaR to avoid the worst TC scenarios — creating severe unhedged exposure.

5. **P&L variance ordering: MSE < CVaR < Entropic across all κ.** At κ=0.02 the standard deviations are 0.0081, 0.0097, 0.0173 respectively — Entropic is more than twice as volatile as MSE. This confirms that more tail-focused objectives sacrifice variance control, and at γ=100 the sacrifice is too large.

6. **All three loss functions achieve fair premiums (mean P&L ≈ 0).** The gradient routing fix ensures this for CVaR and Entropic. The comparison is purely about hedging strategy, not pricing.

7. **CVaR α=0.95 remains the recommended loss function for moderate TC.** The 3-way comparison confirms it occupies the best operating point: strong enough tail focus to improve on MSE at realistic κ values, but not so extreme that it abandons hedging quality altogether.

---

## Planned Extensions / TODO

- [x] Kappa sensitivity: train and evaluate at κ ∈ {0.0, 0.001, 0.005, 0.01, 0.02}
- [x] Quantify TC reduction: total TC paid per path comparison (NN vs BSM)
- [x] Trade size plots: |Δδ| per step to visualise reduced rebalancing
- [x] CVaR loss: gradient-routed implementation with multi-κ comparison
- [x] Entropic risk measure: 3-way loss function comparison (MSE / CVaR / Entropic)
- [ ] Tune Entropic γ: find the value where ERM first matches or beats CVaR
- [ ] Retrain κ=0.02 with more epochs (e.g. 200) to confirm convergence
- [ ] Heston model: swap GBM paths for Heston paths using existing generator in project/stock/generators.py
- [ ] Increase N_PATHS_TRAIN for better generalisation at high κ

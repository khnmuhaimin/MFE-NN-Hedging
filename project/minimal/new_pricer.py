"""
Deep Hedging - Option Pricing & Replication via Neural Networks
Based on Buehler et al. (2018) "Deep Hedging"

Uses generate_gbm_augmented, which returns a single (paths, timesteps, 4)
float32 tensor ready for PyTorch.  Feature layout:
    [:, :, 0]  S_t / K          (normalised price)
    [:, :, 1]  time to maturity
    [:, :, 2]  realised vol
    [:, :, 3]  BS delta
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import norm
import matplotlib.pyplot as plt
import os

from project.stock.generators import generate_gbm_augmented
from project.helpers.path_helpers import project_path
from project.helpers.helpers import get_torch_device
from project.minimal.constants import S0, K_LO, K_HI, SIGMA_LO, SIGMA_HI, T, get_model_params
from project.minimal.model import HedgingNet
from project.minimal.model_name import MODEL_NAME

os.makedirs(project_path("results/figures"), exist_ok=True)
os.makedirs(project_path("results/models"), exist_ok=True)
os.makedirs(project_path("results/logs"),   exist_ok=True)

# ── Training hyper-parameters ─────────────────────────────────────────────────
N_PATHS_TRAIN = 10_000
N_PATHS_TEST  = 10_000

SEED = 42

HIDDEN_NEURONS = 0
HIDDEN_LAYERS = 0
LEARNING_RATE = 0
BATCH_SIZE = 0
N_EPOCHS = 0
ACTIVATION_PARAM = 0
GRAD_CLIP_THRESHOLD = 0
N_FEATURES = 0
N = 0
H = 0
locals().update(get_model_params(MODEL_NAME))
# HIDDEN_NEURONS = 265
# HIDDEN_LAYERS = 8


# ── Device selection ──────────────────────────────────────────────────────────
DEVICE = get_torch_device()
print(f"Using device: {DEVICE}")
    

# ── Weight Initializaation ──────────────────────────────────────────────────────────────────────
# def init_weights_he(m):
#     # Check if the module is a linear layer
#     if isinstance(m, nn.Linear):
#         # Apply Kaiming normal to weights
#         nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        
#         # Initialize biases to 0 to prevent initial shifts
#         if m.bias is not None:
#             nn.init.zeros_(m.bias)


# ── Loss ──────────────────────────────────────────────────────────────────────

def loss_function(pnl: torch.Tensor) -> torch.Tensor:
    """Mean-squared P&L (variance-minimisation objective)."""
    return (pnl ** 2).mean()


# ── Path rollout ──────────────────────────────────────────────────────────────

def run_paths(
    features_batch: torch.Tensor,
    hedging_net: HedgingNet,
) -> torch.Tensor:
    """
    Roll out the hedging strategy for one mini-batch.

    Parameters
    ----------
    features_batch : (batch, N+1, N_FEATURES)
        Output of generate_gbm_augmented, already on DEVICE.
        features_batch[:, t, 0] is S_t/K -- used directly as the price
        for computing cash flows and the terminal payoff.

    Returns
    -------
    pnl : (batch,)
    """
    batch   = features_batch.shape[0]
    n_steps = features_batch.shape[1] - 1   # = N

    currency   = torch.zeros(batch, device=DEVICE)
    underlying = torch.zeros(batch, device=DEVICE)
    prev_delta = torch.zeros(batch, device=DEVICE)

    for t in range(n_steps):
        state     = features_batch[:, t, :]    # (batch, N_FEATURES)
        delta     = hedging_net(state)          # (batch,)
        St_scaled = features_batch[:, t, 0]    # S_t / K

        trade      = delta - prev_delta
        currency  -= trade * St_scaled
        underlying += trade
        prev_delta  = delta

    # Terminal payoff in S/K units  (strike normalised to 1)
    S_T_scaled = features_batch[:, -1, 0]
    payoff     = torch.clamp(S_T_scaled - 1.0, min=0.0)

    pnl = hedging_net.premium + underlying * S_T_scaled + currency - payoff
    return pnl


# ── Training ──────────────────────────────────────────────────────────────────

def train(features_train: torch.Tensor):
    """
    Parameters
    ----------
    features_train : (paths, N+1, N_FEATURES)  -- direct output of generator

    Returns
    -------
    hedging_net  : trained HedgingNet
    epoch_losses : list[float]
    """
    dataset = TensorDataset(features_train)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    hedging_net = HedgingNet(N_FEATURES, HIDDEN_NEURONS, HIDDEN_LAYERS, ACTIVATION_PARAM).to(DEVICE)
    # hedging_net.apply(init_weights_he)

    optimizer = optim.Adam(
        hedging_net.parameters(), 
        lr=LEARNING_RATE, 
        betas=(0.9, 0.999),  # beta1 and beta2
        eps=1e-8             # epsilon to prevent division by zero
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    print(f"\n{'─'*70}")
    print(f"{'Epoch':>6}  {'Loss':>12}  {'Mean P&L':>12}  {'Std P&L':>10}  {'Premium':>10}")
    print(f"{'─'*70}")

    epoch_losses = []

    for epoch in range(1, N_EPOCHS + 1):
        hedging_net.train()
        batch_losses = []

        for (F_batch,) in loader:
            F_batch = F_batch.to(DEVICE)

            optimizer.zero_grad()
            pnl  = run_paths(F_batch, hedging_net)
            loss = loss_function(pnl)
            loss.backward()
            nn.utils.clip_grad_norm_(hedging_net.parameters(), max_norm=GRAD_CLIP_THRESHOLD, norm_type=2)
            optimizer.step()
            batch_losses.append(loss.item())

        scheduler.step()
        epoch_losses.append(float(np.mean(batch_losses)))

        if epoch % 10 == 0 or epoch == 1:
            hedging_net.eval()
            with torch.no_grad():
                n_eval  = min(2_000, features_train.shape[0])
                pnl_all = run_paths(features_train[:n_eval].to(DEVICE), hedging_net)
            print(
                f"{epoch:>6}  {epoch_losses[-1]:>12.6f}  "
                f"{pnl_all.mean().item():>12.6f}  {pnl_all.std().item():>10.6f}  "
                f"{hedging_net.premium.item():>10.6f}"
            )

    print(f"{'─'*70}\n")
    return hedging_net, epoch_losses


# ── BSM helpers ───────────────────────────────────────────────────────────────

def bsm_call(S0, K, r, sigma, T):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bsm_delta(S, K, r, sigma, t, T):
    tau = T - t
    if np.isscalar(tau) and tau <= 0:
        return np.where(S > K, 1.0, 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * tau) \
         / (sigma * np.sqrt(np.maximum(tau, 1e-8)))
    return norm.cdf(d1)


def bsm_hedge_pnl(features: torch.Tensor, sigmas: np.ndarray, bsm_deltas: np.ndarray) -> np.ndarray:
    """
    Realistic BSM delta hedge benchmark.
    Uses the historical rolling realized volatilities/deltas available at each step 
    rather than perfect lookahead knowledge of the true path volatility.
    """
    # S/K values: (paths, N+1) -> transpose to (N+1, paths) for sequential looping
    S_scaled = features[:, :, 0].numpy().T   # Shape: (N+1, paths)
    n_paths  = S_scaled.shape[1]

    # bsm_deltas shape is (paths, N) -> transpose to (N, paths) to align with time loops
    deltas_matrix = bsm_deltas.T             # Shape: (N, paths)

    currency   = np.zeros(n_paths)
    underlying = np.zeros(n_paths)
    prev_delta = np.zeros(n_paths)
    
    # 1. Premium calculation
    # For a realistic baseline, we price the initial option using the true sigma 
    # (implied vol at t=0), or you can use your training midpoint here.
    premium = bsm_call(1.0, 1.0, 0, sigmas, T) 

    # 2. Dynamic Rebalancing Loop
    for t in range(N):
        St = S_scaled[t, :]
        
        # Pull the realistic delta directly from your pre-computed array
        delta = deltas_matrix[t, :]
        
        trade       = delta - prev_delta
        currency   -= trade * St
        underlying += trade
        currency   *= np.exp(0 * H)
        prev_delta  = delta

    # 3. Final Settlement at Expiry
    S_T    = S_scaled[N, :]
    payoff = np.maximum(S_T - 1.0, 0.0)
    pnl    = premium + underlying * S_T + currency - payoff
    
    return pnl


# ── Evaluation / plotting ─────────────────────────────────────────────────────

def get_nn_deltas(features_path: torch.Tensor, hedging_net: HedgingNet) -> np.ndarray:
    """
    NN delta at each of the N rebalancing steps for a single path.

    Parameters
    ----------
    features_path : (N+1, N_FEATURES)

    Returns
    -------
    deltas : (N,)
    """
    hedging_net.eval()
    deltas = np.zeros(N)
    with torch.no_grad():
        for t in range(N):
            state     = features_path[t].unsqueeze(0).to(DEVICE)   # (1, N_FEATURES)
            deltas[t] = hedging_net(state).item()
    return deltas


def plot_learning_curve(epoch_losses: list) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, len(epoch_losses) + 1), epoch_losses, linewidth=1.2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss (Learning Curve)")
    plt.tight_layout()
    plt.savefig(project_path(f"results/figures/{MODEL_NAME}_learning_curve.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Learning curve saved.")


def plot_delta_paths(features_test: torch.Tensor, Sigma_test: np.ndarray, bsm_deltas, hedging_net: HedgingNet) -> None:
    """
    NN vs BSM delta over time for one ITM and one OTM path.

    features_test : (paths, N+1, N_FEATURES)
    """
    S_T = features_test[:, -1, 0].numpy()   # terminal S/K for all paths
    itm_candidates = np.where(S_T > 1.05)[0]
    otm_candidates = np.where(S_T < 0.95)[0]

    if len(itm_candidates) == 0 or len(otm_candidates) == 0:
        print("Could not find suitable ITM/OTM paths -- skipping delta path plot.")
        return

    itm_idx = itm_candidates[np.argmax(S_T[itm_candidates])]
    otm_idx = otm_candidates[np.argmin(S_T[otm_candidates])]
    times   = np.linspace(0, T, N)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Delta Hedge Path: NN vs BSM", fontsize=13)

    for ax, idx, label in zip(axes, [itm_idx, otm_idx], ["In-the-Money", "Out-of-the-Money"]):
        fp         = features_test[idx]           # (N+1, N_FEATURES)
        nn_deltas  = get_nn_deltas(fp, hedging_net)
        # bsm_path_deltas = np.zeros(N)
        # for t in range(N):
        #     St = fp[t, 0].item()           # Current S_t / K value
        #     bsm_path_deltas[t] = bsm_delta(
        #         S=St, K=1.0, r=R, sigma=Sigma_test[idx], t=t * H, T=T
        #     )
        bsm_path_deltas = bsm_deltas[idx, :]

        ax.plot(times, nn_deltas,  label="NN hedge",  linewidth=1.2)
        ax.plot(times, bsm_path_deltas, label="BSM hedge", linewidth=1.2, linestyle="--")
        ax.set_title(f"{label}  (S_T/K = {S_T[idx]:.3f})")
        ax.set_xlabel("Time")
        ax.set_ylabel("Delta")
        ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(project_path(f"results/figures/{MODEL_NAME}_delta_paths.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Delta path plot saved.")


def test(features_test: torch.Tensor, Sigma_test: np.ndarray, bsm_deltas, hedging_net: HedgingNet) -> dict:
    """
    Evaluate the trained network on held-out paths.

    features_test : (paths, N+1, N_FEATURES)

    Returns a dict of test statistics for logging.
    """
    hedging_net.eval()

    with torch.no_grad():
        pnl    = run_paths(features_test.to(DEVICE), hedging_net)
        pnl_np = pnl.cpu().numpy()

    S_T          = features_test[:, -1, 0].numpy()   # terminal S/K
    payoffs      = np.maximum(S_T - 1.0, 0.0)
    bsm_pnl      = bsm_hedge_pnl(features_test, Sigma_test, bsm_deltas)
    unhedged_pnl = hedging_net.premium.item() - payoffs

    percentiles = [1, 5, 25, 75, 95, 99]
    nn_pcts     = np.percentile(pnl_np, percentiles)
    bsm_pcts    = np.percentile(bsm_pnl, percentiles)

    print("=" * 70)
    print("  TEST RESULTS")
    print("=" * 70)
    print(f"\n  {'':35s}  {'NN Hedge':>10}  {'BSM Hedge':>10}")
    print(f"  {'─'*57}")
    print(f"  {'Mean P&L':35s}  {pnl_np.mean():>10.6f}  {bsm_pnl.mean():>10.6f}")
    print(f"  {'Std P&L':35s}  {pnl_np.std():>10.6f}  {bsm_pnl.std():>10.6f}")
    for p, nn_v, bsm_v in zip(percentiles, nn_pcts, bsm_pcts):
        print(f"  {f'P{p}':35s}  {nn_v:>10.6f}  {bsm_v:>10.6f}")
    print(f"\n  {'Learned premium (S/K units)':35s}  {hedging_net.premium.item():>10.6f}")
    print(f"  {'BSM ATM price (sigma=0.20)':35s}  {bsm_call(1.0, 1.0, 0, 0.20, T):>10.6f}")
    print("=" * 70)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    fig.suptitle("Terminal P&L Distribution (S/K units)", fontsize=13)

    for ax, data, title in zip(
        axes,
        [pnl_np, bsm_pnl],
        ["Deep Hedging (learned)", "BSM Delta Hedge"],
    ):
        ax.hist(data, bins=50, edgecolor="none", alpha=0.8)
        ax.axvline(data.mean(), color="red",   linestyle="--", linewidth=1.2,
                   label=f"Mean {data.mean():.4f}")
        ax.axvline(0,           color="black", linestyle=":",  linewidth=1.0, label="Zero")
        ax.set_title(title)
        ax.set_xlabel("P&L")
        ax.set_ylabel("Count")
        ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(project_path(f"results/figures/{MODEL_NAME}_pnl_distribution.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("P&L distribution plot saved.")

    return {
        "nn_mean_pnl":     float(pnl_np.mean()),
        "nn_std_pnl":      float(pnl_np.std()),
        "bsm_mean_pnl":    float(bsm_pnl.mean()),
        "bsm_std_pnl":     float(bsm_pnl.std()),
        "nn_percentiles":  dict(zip([f"P{p}" for p in percentiles], nn_pcts.tolist())),
        "bsm_percentiles": dict(zip([f"P{p}" for p in percentiles], bsm_pcts.tolist())),
        "learned_premium": float(hedging_net.premium.item()),
        "bsm_atm_price":   float(bsm_call(1.0, 1.0, 0, 0.20, T)),
    }


# ── Results logger ────────────────────────────────────────────────────────────

def save_results(epoch_losses: list, test_stats: dict) -> None:
    path = project_path(f"results/logs/{MODEL_NAME}_training_results.txt")
    with open(path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("  DEEP HEDGING - TRAINING RUN RESULTS\n")
        f.write("=" * 70 + "\n\n")

        f.write("── Run configuration ──────────────────────────────────────────\n")
        f.write(f"  Seed              : {SEED}\n")
        f.write(f"  N_PATHS_TRAIN     : {N_PATHS_TRAIN}\n")
        f.write(f"  N_PATHS_TEST      : {N_PATHS_TEST}\n")
        f.write(f"  N (steps)         : {N}\n")
        f.write(f"  T (years)         : {T}\n")
        f.write(f"  S0                : {S0}\n")
        f.write(f"  K range           : [{K_LO}, {K_HI}]\n")
        f.write(f"  sigma range       : [{SIGMA_LO}, {SIGMA_HI}]\n")
        f.write(f"  Batch size        : {BATCH_SIZE}\n")
        f.write(f"  Epochs            : {N_EPOCHS}\n")
        f.write(f"  Learning rate     : {LEARNING_RATE}\n")
        f.write(f"  Optimizer         : Adam (torch.optim.Adam)\n")
        f.write(f"  Device            : {DEVICE}\n\n")

        f.write("── Loss per epoch ─────────────────────────────────────────────\n")
        f.write(f"  {'Epoch':>6}  {'Loss':>14}\n")
        f.write(f"  {'─'*22}\n")
        for i, loss in enumerate(epoch_losses, start=1):
            f.write(f"  {i:>6}  {loss:>14.8f}\n")

        f.write("\n── Test results ───────────────────────────────────────────────\n")
        f.write(f"  {'Metric':40s}  {'NN Hedge':>12}  {'BSM Hedge':>12}\n")
        f.write(f"  {'─'*66}\n")
        f.write(f"  {'Mean P&L':40s}  {test_stats['nn_mean_pnl']:>12.6f}  "
                f"{test_stats['bsm_mean_pnl']:>12.6f}\n")
        f.write(f"  {'Std P&L':40s}  {test_stats['nn_std_pnl']:>12.6f}  "
                f"{test_stats['bsm_std_pnl']:>12.6f}\n")
        for key in test_stats["nn_percentiles"]:
            nn_v  = test_stats["nn_percentiles"][key]
            bsm_v = test_stats["bsm_percentiles"][key]
            f.write(f"  {key:40s}  {nn_v:>12.6f}  {bsm_v:>12.6f}\n")
        f.write(f"\n  {'Learned premium (S/K units)':40s}  "
                f"{test_stats['learned_premium']:>12.6f}\n")
        f.write(f"  {'BSM ATM price (sigma=0.20)':40s}  "
                f"{test_stats['bsm_atm_price']:>12.6f}\n")
        f.write("=" * 70 + "\n")

    print(f"Results saved to '{path}'")


# ── Model persistence ─────────────────────────────────────────────────────────

def save_model(hedging_net: HedgingNet) -> None:
    path = project_path(f"results/models/{MODEL_NAME}_hedging_model.pt")
    torch.save({
        "hedging_net_state": hedging_net.state_dict(),
        "params": {
            "S0": S0, "K_LO": K_LO, "K_HI": K_HI,
            "SIGMA_LO": SIGMA_LO, "SIGMA_HI": SIGMA_HI,
            "R": 0, "T": T, "N": N, "H": H,
            "N_FEATURES": N_FEATURES,
        },
        "seed": SEED,
    }, path)
    print(f"Model saved to '{path}'")


def load_model() -> HedgingNet:
    path        = project_path(f"results/models/{MODEL_NAME}_hedging_model.pt")
    checkpoint  = torch.load(path, map_location=DEVICE)
    
    hedging_net = HedgingNet(N_FEATURES, HIDDEN_NEURONS, HIDDEN_LAYERS, ACTIVATION_PARAM).to(DEVICE)
    
    hedging_net.load_state_dict(checkpoint["hedging_net_state"])
    return hedging_net


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)

    print("── Simulating training paths ──")
    features_train, K_train, Sigma_train = generate_gbm_augmented(
        S0       = S0,
        K_lo     = K_LO,
        K_hi     = K_HI,
        sigma_lo = SIGMA_LO,
        sigma_hi = SIGMA_HI,
        N        = N,
        T        = T,
        paths    = N_PATHS_TRAIN,
        rng      = rng,
    )   # (N_PATHS_TRAIN, N+1, 4)
    features_train = features_train[:, :, :N_FEATURES]

    print("\n── Training ──")
    hedging_net, epoch_losses = train(features_train)

    print("\n── Plotting learning curve ──")
    plot_learning_curve(epoch_losses)

    print("\n── Simulating test paths ──")
    features_test, K_test, Sigma_test = generate_gbm_augmented(
        S0       = S0,
        K_lo     = K_LO,
        K_hi     = K_HI,
        sigma_lo = SIGMA_LO,
        sigma_hi = SIGMA_HI,
        N        = N,
        T        = T,
        paths    = N_PATHS_TEST,
        rng      = rng,
    )
    bsm_deltas = features_test[:, :N, 3].numpy()
    features_test = features_test[:, :, :N_FEATURES]

    print("── Testing ──")
    test_stats = test(features_test, Sigma_test, bsm_deltas, hedging_net)

    print("\n── Plotting delta paths ──")
    plot_delta_paths(features_test, Sigma_test, bsm_deltas, hedging_net)

    print("\n── Saving results and model ──")
    save_results(epoch_losses, test_stats)
    save_model(hedging_net)
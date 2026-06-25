import numpy as np
import torch
import os
import matplotlib.pyplot as plt

from project.helpers.path_helpers import project_path
from project.helpers.helpers import get_torch_device
from project.minimal.constants import S0, K_LO, K_HI, SIGMA_LO, SIGMA_HI, T, get_model_params
from project.stock.generators import generate_gbm_augmented
from project.minimal2.new_pricer import run_paths
from project.minimal.bs_model import bsm_call, bsm_delta, bsm_hedge_pnl
from project.minimal.model_name import MODEL_NAMES

# Upgraded model import to reference your dual network class
from project.minimal2.model import HedgingNet2

SEED = 23
CONFIDENCE_THRESHOLD = 0.001  # max acceptable CI half-width (std of bootstrap estimates)
N_PATHS_TEST = 1000
BSM_MODELS = ["base_daily", "base_monthly"]


def bootstrap(data, stat_fn, n_samples=1000):
    """
    Bootstrap a statistic from 1D data.

    Parameters
    ----------
    data      : array-like, 1D
    stat_fn   : callable, e.g. np.mean, np.median, np.std
    n_samples : number of bootstrap resamples (default 1000)

    Returns
    -------
    (mean, std) of the bootstrapped statistic
    """
    data = np.asarray(data)
    n = len(data)

    stats = np.array([
        stat_fn(np.random.choice(data, size=n, replace=True))
        for _ in range(n_samples)
    ])

    return stats.mean(), stats.std()


def load_model(model_name) -> HedgingNet2:
    DEVICE = get_torch_device()
    path       = project_path(f"results/models/{model_name}2_hedging_model.pt")
    checkpoint = torch.load(path, map_location=DEVICE)

    params             = get_model_params(model_name)
    hidden_neurons     = params["HIDDEN_NEURONS"]
    hidden_layers      = params["HIDDEN_LAYERS"]
    activation_param   = params["ACTIVATION_PARAM"]
    n_features         = params["N_FEATURES"]
    
    # Static parameters matching assignment architecture specifications
    pricing_hidden_neurons = 32
    pricing_hidden_layers = 2

    # Initialize the custom dual-network instance
    hedging_net = HedgingNet2(
        hedging_features=n_features, 
        hedging_hidden_neurons=hidden_neurons, 
        hedging_depth=hidden_layers, 
        gamma=activation_param,
        pricing_hidden_neurons=pricing_hidden_neurons,
        pricing_depth=pricing_hidden_layers
    ).to(DEVICE)
    
    hedging_net.load_state_dict(checkpoint["hedging_net_state"])
    return hedging_net


def test_model(model_name: str):
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)

    params             = get_model_params(model_name)
    N_FEATURES         = params["N_FEATURES"]
    N                  = params["N"]
    hedging_net = load_model(model_name)
    hedging_net.eval()
    
    pnl_samples = []
    premium_samples = []

    while True:
        features, K, Sigma = generate_gbm_augmented(
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
        features   = features[:, :, :N_FEATURES]
        features   = features.to(get_torch_device())

        # 1. Run paths to pull the tracking portfolio's actual outcome
        batch_pnl = run_paths(features, hedging_net).cpu().detach().numpy()  # 1D array
        pnl_samples.extend(batch_pnl)

        # 2. Extract dynamic batch premiums via the t=0 traffic router flag
        with torch.no_grad():
            initial_moneyness = features[:, 0, [0]]
            batch_premiums = hedging_net(initial_moneyness, is_initial=True).cpu().numpy()
            premium_samples.extend(batch_premiums)

        data = np.array(pnl_samples)
        _, mean_se = bootstrap(data, np.mean)
        _, std_se  = bootstrap(data, np.std)

        print(f"n={len(data):>6}  mean_SE={mean_se:.5f}  std_SE={std_se:.5f}")

        if mean_se < CONFIDENCE_THRESHOLD and std_se < CONFIDENCE_THRESHOLD:
            break

    final_mean, mean_se = bootstrap(data, np.mean)
    final_std, std_se = bootstrap(data, np.std)
    
    return {
        "mean": final_mean, 
        "std": final_std, 
        "n": len(data), 
        "mean_se": mean_se, 
        "std_se": std_se,
        "mean_learned_premium": float(np.mean(premium_samples))
    }


def test_bsm_model(model_name: str):
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)

    params = get_model_params(model_name)
    N      = params["N"]
    
    bsm_pnl_samples = []

    print(f"Running standalone BSM evaluation for {model_name}...")
    print(f"{'Paths':>7} | {'BSM Mean SE':>11} {'BSM Std SE':>10}")
    print("─" * 35)

    while True:
        features, K, Sigma = generate_gbm_augmented(
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
        
        bsm_deltas = features[:, :N, 3].numpy()

        batch_pnl = bsm_hedge_pnl(features, Sigma, bsm_deltas)
        bsm_pnl_samples.extend(batch_pnl)

        data = np.array(bsm_pnl_samples)
        final_mean, mean_se = bootstrap(data, np.mean)
        final_std, std_se  = bootstrap(data, np.std)

        print(f"n={len(data):>6}  mean_SE={mean_se:.5f}  std_SE={std_se:.5f}")

        if mean_se < CONFIDENCE_THRESHOLD and std_se < CONFIDENCE_THRESHOLD:
            break
    
    return {
        "mean": final_mean, 
        "std": final_std, 
        "n": len(data), 
        "mean_se": mean_se, 
        "std_se": std_se
    }



def summary_report():
    results_report = {}

    # 1. Run Loop over all Neural Network variations
    print("Evaluating Neural Networks across config list...")
    for model_name in MODEL_NAMES:
        print(f" -> Testing NN: {model_name}")
        results_report[f"NN_{model_name}"] = test_model(model_name)

    # 2. Run Loop over specified Black-Scholes Baselines
    print("\nEvaluating Standalone BSM Baseline Frameworks...")
    for bsm_name in BSM_MODELS:
        print(f" -> Testing BSM: {bsm_name}")
        results_report[f"BSM_{bsm_name}"] = test_bsm_model(bsm_name)

    # 3. Print Clean, Easy-to-Read Summary Report
    print(f"\n{'═'*50}\n  FINAL PERFORMANCE REPORT SUMMARY\n{'═'*50}")
    
    for strategy_key, metrics in results_report.items():
        print(f"\n● Strategy: {strategy_key}")
        print(f"  {'─'*40}")
        for item, value in metrics.items():
            if item != "n":
                print(f"  {item:22s}: {(value*100):.4f}")
            else:
                print(f"  {item:22s}: {value}")
                
    print(f"\n{'═'*50}\n")

    return results_report


def sort_report_keys(report_keys):
    """
    Forces a strict structural ordering:
    BSM Daily -> BSM Monthly -> Base Daily -> Base Monthly -> Extended Daily -> Extended Monthly
    """
    order_mapping = {
        "BSM_base_daily": 0,
        "BSM_base_monthly": 1,
        "NN_base_daily": 2,
        "NN_base_monthly": 3,
        "NN_base_relvol_bsdelta_daily": 4,
        "NN_base_relvol_bsdelta_monthly": 5
    }
    # Sort based on the map, fallback to end of list if a key is unrecognized
    return sorted(report_keys, key=lambda k: order_mapping.get(k, 99))

def forest_plot():
    results_report = summary_report()
    
    # Enforce strict sequential ordering
    strategies = sort_report_keys(list(results_report.keys()))
    
    # Map raw internal dictionary keys to clean presentation names
    name_map = {
        "BSM_base_daily": "BS Hedge Daily",
        "BSM_base_monthly": "BS Hedge Monthly",
        "NN_base_daily": "2-Input Model Daily",
        "NN_base_monthly": "2-Input Model Monthly",
        "NN_base_relvol_bsdelta_daily": "4-Input Model Daily",
        "NN_base_relvol_bsdelta_monthly": "4-Input Model Monthly"
    }
    labels = [name_map.get(s, s.replace("_", " ")) for s in strategies]

    means     = [float(results_report[s]["mean"]) * 100     for s in strategies]
    stds      = [float(results_report[s]["std"]) * 100      for s in strategies]
    mean_ses  = [float(results_report[s]["mean_se"]) * 100  for s in strategies]
    std_ses   = [float(results_report[s]["std_se"]) * 100   for s in strategies]

    # Vibrant Strategy-Level Palette assignment
    colors = []
    for s in strategies:
        if s.startswith("BSM"):
            colors.append("#FF3B30")      # Vibrant Coral/Red
        elif "relvol" in s:
            colors.append("#34C759")      # Sharp Emerald Green
        else:
            colors.append("#007AFF")      # Bright Electric Blue
            
    y_pos  = np.arange(len(strategies))

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)

    # ── Left: Mean P&L ────────────────────────────────────────────────────────────
    ax = axes[0]
    for i, (m, se, color) in enumerate(zip(means, mean_ses, colors)):
        ax.plot([m - 1.96 * se, m + 1.96 * se], [i, i], color=color, linewidth=2.5)
        ax.plot(m, i, "o", color=color, markersize=7)

    ax.axvline(0, color="black", linestyle="--", linewidth=1.2)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Mean P&L (%)")
    ax.set_title("Mean P&L with 95% CI")
    
    # Solid black axes borders with functional grey grid lines
    ax.grid(True, linestyle="-", alpha=0.25, color="grey")
    for spine in ["top", "right", "left", "bottom"]:
        ax.spines[spine].set_color("black")
        ax.spines[spine].set_linewidth(1.0)

    # ── Right: Std P&L ────────────────────────────────────────────────────────────
    ax = axes[1]
    for i, (s, se, color) in enumerate(zip(stds, std_ses, colors)):
        ax.plot([s - 1.96 * se, s + 1.96 * se], [i, i], color=color, linewidth=2.5)
        ax.plot(s, i, "o", color=color, markersize=7)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([])
    ax.set_xlabel("Std P&L (%)")
    ax.set_title("Std P&L with 95% CI")
    
    # Solid black axes borders with functional grey grid lines
    ax.grid(True, linestyle="-", alpha=0.25, color="grey")
    for spine in ["top", "right", "left", "bottom"]:
        ax.spines[spine].set_color("black")
        ax.spines[spine].set_linewidth(1.0)

    # ── Shared legend ─────────────────────────────────────────────────────────────
    handles = [
        plt.Line2D([0], [0], color="#FF3B30", marker="o", linewidth=2.5, label="Black-Scholes Baseline"),
        plt.Line2D([0], [0], color="#007AFF", marker="o", linewidth=2.5, label="2-Input Neural Net"),
        plt.Line2D([0], [0], color="#34C759", marker="o", linewidth=2.5, label="4-Input Neural Net"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9, bbox_to_anchor=(0.5, -0.04))

    # plt.tight_layout()
    plt.savefig(project_path("results/figures/forest_plot2.png"), dpi=150, bbox_inches="tight")
    plt.show()


def forest_plot_nn_only():
    results_report = summary_report()

    # Filter out everything except Neural Network targets and sort them
    nn_strategies = [s for s in results_report.keys() if s.startswith("NN_")]
    nn_strategies = sort_report_keys(nn_strategies)
    
    name_map = {
        "NN_base_daily": "2-Input Model Daily",
        "NN_base_monthly": "2-Input Model Monthly",
        "NN_base_relvol_bsdelta_daily": "4-Input Model Daily",
        "NN_base_relvol_bsdelta_monthly": "4-Input Model Monthly"
    }
    labels = [name_map.get(s, s.replace("_", " ")) for s in nn_strategies]

    means     = [float(results_report[s]["mean"]) * 100     for s in nn_strategies]
    stds      = [float(results_report[s]["std"]) * 100      for s in nn_strategies]
    mean_ses  = [float(results_report[s]["mean_se"]) * 100  for s in nn_strategies]
    std_ses   = [float(results_report[s]["std_se"]) * 100   for s in nn_strategies]

    # Split colors exclusively across network structural input lines
    colors = ["#34C759" if "relvol" in s else "#007AFF" for s in nn_strategies]
    y_pos  = np.arange(len(nn_strategies))

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)

    # ── Left: Mean P&L ────────────────────────────────────────────────────────────
    ax = axes[0]
    for i, (m, se, color) in enumerate(zip(means, mean_ses, colors)):
        ax.plot([m - 1.96 * se, m + 1.96 * se], [i, i], color=color, linewidth=2.5)
        ax.plot(m, i, "o", color=color, markersize=7)

    ax.axvline(0, color="black", linestyle="--", linewidth=1.2)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Mean P&L (%)")
    ax.set_title("NN Mean P&L with 95% CI")
    
    ax.grid(True, linestyle="-", alpha=0.25, color="grey")
    for spine in ["top", "right", "left", "bottom"]:
        ax.spines[spine].set_color("black")
        ax.spines[spine].set_linewidth(1.0)

    # ── Right: Std P&L ────────────────────────────────────────────────────────────
    ax = axes[1]
    for i, (s, se, color) in enumerate(zip(stds, std_ses, colors)):
        ax.plot([s - 1.96 * se, s + 1.96 * se], [i, i], color=color, linewidth=2.5)
        ax.plot(s, i, "o", color=color, markersize=7)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([])
    ax.set_xlabel("Std P&L (%)")
    ax.set_title("NN Std P&L with 95% CI")
    
    ax.grid(True, linestyle="-", alpha=0.25, color="grey")
    for spine in ["top", "right", "left", "bottom"]:
        ax.spines[spine].set_color("black")
        ax.spines[spine].set_linewidth(1.0)

    # ── Shared legend ─────────────────────────────────────────────────────────────
    handles = [
        plt.Line2D([0], [0], color="#007AFF", marker="o", linewidth=2.5, label="2-Input Neural Net"),
        plt.Line2D([0], [0], color="#34C759", marker="o", linewidth=2.5, label="4-Input Neural Net"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=9, bbox_to_anchor=(0.5, -0.05))

    # plt.tight_layout()
    plt.savefig(project_path("results/figures/forest_plot_nn_only2.png"), dpi=150, bbox_inches="tight")
    plt.show()


def plot_premium_curves_vs_S0():
    """
    Plots the initial Option Premium (C_0) vs Initial Stock Price (S0) from 80 to 120
    across all 4 trained Neural Networks and the analytical Black-Scholes benchmark.
    All outputs are scaled by 100 for absolute percentage visibility.
    """
    # 1. Parameter and input coordinate sweep setup
    K_fixed = 100.0
    sigma_ref = (SIGMA_LO + SIGMA_HI)/2
    r_ref = 0.0
    
    S0_sweep = np.linspace(80, 120, 100)
    S0_scaled_sweep = S0_sweep / K_fixed  # Normalized input scale (S_0 / K)
    
    # 2. Compute analytical Black-Scholes benchmark prices
    bsm_premiums = [bsm_call(S0/K_fixed, 1, r_ref, sigma_ref, T) for S0 in S0_sweep]
    
    # 3. Pull learned outputs sequentially across model architectures
    device = get_torch_device()
    model_list = ["base_daily", "base_monthly", "base_relvol_bsdelta_daily", "base_relvol_bsdelta_monthly"]
    nn_premiums = {m: [] for m in model_list}
    
    for m_name in model_list:
        net = load_model(m_name).to(device)
        net.eval()
        
        with torch.no_grad():
            for S_norm in S0_scaled_sweep:
                # Construct clean 2D column tensor of shape (1, 1) to pass to pricer net
                feat_tensor = torch.tensor([[S_norm]], dtype=torch.float32).to(device)
                
                # Extract dynamic batch premium directly from model instance
                p_raw = net(feat_tensor, is_initial=True).cpu().item()
                
                # Map from normalized state space back to absolute cash terms
                nn_premiums[m_name].append(p_raw)

    # 4. Plotting Phase
    plt.figure(figsize=(10, 6))
    
    # Strictly preserve the established color mapping archetype
    # Black-Scholes = Crimson, 2-Input Models = Electric Blue, 4-Input Models = Emerald Green
    plt.plot(S0_sweep, np.array(bsm_premiums) * 100.0, color="#FF3B30", linewidth=2.5, label="BSM Benchmark (Analytical)")
    
    plt.plot(S0_sweep, np.array(nn_premiums["base_daily"]) * 100.0, color="#007AFF", linewidth=2.0, linestyle="-", label="2-Input Model Daily")
    plt.plot(S0_sweep, np.array(nn_premiums["base_monthly"]) * 100.0, color="#007AFF", linewidth=2.0, linestyle="--", label="2-Input Model Monthly")
    
    plt.plot(S0_sweep, np.array(nn_premiums["base_relvol_bsdelta_daily"]) * 100.0, color="#34C759", linewidth=2.0, linestyle="-", label="4-Input Model Daily")
    plt.plot(S0_sweep, np.array(nn_premiums["base_relvol_bsdelta_monthly"]) * 100.0, color="#34C759", linewidth=2.0, linestyle="--", label="4-Input Model Monthly")
    
    # Structural Markers and Solid Boundaries
    plt.axvline(K_fixed, color="black", linestyle=":", linewidth=1.2, label=f"At-The-Money Reference (K={int(K_fixed)})")
    
    plt.title("Initial Option Premium ($\pi$) vs. Initial Stock Price ($S_0$)", fontsize=12, fontweight="bold")
    plt.xlabel("Initial Stock Price ($S_0$)", fontsize=10)
    plt.ylabel("Option Premium ($\pi$)", fontsize=10)
    plt.xlim([80, 120])
    
    # Force exact solid framework styling lines
    plt.grid(True, linestyle="-", alpha=0.25, color="grey")
    for spine in ["top", "right", "left", "bottom"]:
        plt.gca().spines[spine].set_color("black")
        plt.gca().spines[spine].set_linewidth(1.0)
        
    plt.legend(loc="upper left", frameon=True, fontsize=9)
    plt.tight_layout()
    
    # Save directly to the designated figures directory
    plot_save_path = project_path("results/figures/option_pricing_curves_vs_S02.png")
    plt.savefig(plot_save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Premium surface grid profile successfully saved to: {plot_save_path}")


def plot_delta_slices_vibrant_by_vol():
    """
    Generates a 3x1 grid of equal-sized subplots sliced by Time to Maturity (t = 0, 0.5, 0.9).
    Curves are grouped cleanly by volatility tier using distinct colors, while line styles
    distinguish between the Analytical BSM baseline (dashed) and the 4-Input NN (solid).
    """
    device = get_torch_device()
    K_fixed = 100.0
    S_sweep = np.linspace(80, 120, 200)
    S_scaled = S_sweep / K_fixed

    # Unified dimensional slices
    time_slices = {
        "Initial Setup ($t = 0$)": 0.0,
        "Mid-Horizon ($t = 0.5T$)": 0.5 * T,
        "Near Expiration ($t = 0.9T$)": 0.9 * T
    }
    vol_slices = [0.1, 0.2, 0.3]
    
    # Track matching vibrant colors explicitly per volatility regime
    vol_colors = {
        0.1: "#007AFF",  # Electric Blue
        0.2: "#FF9500",  # Vibrant Orange
        0.3: "#AF52DE"   # Sharp Purple
    }

    # Load 4-Input Extended Model
    net_4in = load_model("base_relvol_bsdelta_daily").to(device)
    net_4in.eval()

    fig, axes = plt.subplots(3, 1, figsize=(10, 12), constrained_layout=True)
    
    for row_idx, (time_label, t_val) in enumerate(time_slices.items()):
        ax = axes[row_idx]
        
        for sigma_val in vol_slices:
            bsm_line = []
            four_in_line = []
            v_color = vol_colors[sigma_val]
            
            for S_norm in S_scaled:
                # 1. Analytical BSM Reference
                d_bsm = bsm_delta(S_norm, 1.0, 0.0, sigma_val, t_val, T)
                bsm_line.append(d_bsm)
                
                # 2. 4-Input Extended NN Policy
                with torch.no_grad():
                    feat_4in = torch.tensor([[S_norm, t_val, sigma_val, d_bsm]], dtype=torch.float32).to(device)
                    four_in_line.append(net_4in(feat_4in).cpu().item())
            
            # Line Style Rule: BSM = Gapped/Dashed (--), 4-Input NN = Solid (-)
            ax.plot(S_sweep, bsm_line, color=v_color, linestyle="--", linewidth=1.8, alpha=0.85,
                    label=f"BSM ($\sigma={sigma_val}$)" if row_idx == 0 else "")
            ax.plot(S_sweep, four_in_line, color=v_color, linestyle="-", linewidth=2.2,
                    label=f"4-Input NN ($\sigma={sigma_val}$)" if row_idx == 0 else "")
        
        # Grid boundaries and solid formatting
        ax.axvline(K_fixed, color="black", linestyle=":", alpha=0.4)
        ax.set_title(time_label, fontsize=12, fontweight="bold")
        ax.set_ylabel("Hedging Delta ($\Delta$)")
        ax.set_ylim([-0.05, 1.05])
        ax.grid(True, linestyle="-", alpha=0.25, color="grey")
        
        for spine in ["top", "right", "left", "bottom"]:
            ax.spines[spine].set_color("black")
            ax.spines[spine].set_linewidth(1.0)
            
        if row_idx == 2:
            ax.set_xlabel("Stock Price ($S_t$)")

    # Interactive legend tracking both color tiering and structural styles
    fig.legend(loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.04), frameon=True, fontsize=9)
    
    plot_save_path = project_path("results/figures/delta_4in_vs_bsm_by_vol_slices.png")
    plt.savefig(plot_save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_delta_4input_matrix_3x2():
    """
    Generates a 3x2 grid comparing the 4-Input Daily vs 4-Input Monthly models.
    Rows: Time Slices (t = 0.0, 0.5, 0.9)
    Cols: 4-Input Daily, 4-Input Monthly
    """
    device = get_torch_device()
    K_fixed = 100.0
    S_sweep = np.linspace(80, 120, 150)
    S_scaled = S_sweep / K_fixed

    time_slices = {"$t = 0.0$": 0.0, "$t = 0.5T$": 0.5 * T, "$t = 0.9T$": 0.9 * T}
    vol_slices = [0.1, 0.2, 0.3]
    vol_colors = {0.1: "#007AFF", 0.2: "#FF9500", 0.3: "#AF52DE"}

    net_4in_daily   = load_model("base_relvol_bsdelta_daily").to(device)
    net_4in_monthly = load_model("base_relvol_bsdelta_monthly").to(device)
    net_4in_daily.eval()
    net_4in_monthly.eval()

    fig, axes = plt.subplots(3, 2, figsize=(11, 12), constrained_layout=True)

    model_cols = [
        {"name": "4-Input Model Daily", "net": net_4in_daily},
        {"name": "4-Input Model Monthly", "net": net_4in_monthly}
    ]

    for row_idx, (time_lbl, t_val) in enumerate(time_slices.items()):
        for col_idx, col_meta in enumerate(model_cols):
            ax = axes[row_idx, col_idx]
            net = col_meta["net"]
            
            for sigma_val in vol_slices:
                nn_line_data = []
                bsm_line_data = []
                v_color = vol_colors[sigma_val]
                
                d_bsm = bsm_delta(S_scaled, 1.0, 0.0, sigma_val, t_val, T)
                
                for idx, S_norm in enumerate(S_scaled):
                    bsm_line_data.append(d_bsm[idx])
                    feat = torch.tensor([[S_norm, t_val, sigma_val, d_bsm[idx]]], dtype=torch.float32).to(device)
                    with torch.no_grad():
                        nn_line_data.append(net(feat).cpu().item())
                
                ax.plot(S_sweep, bsm_line_data, color=v_color, linestyle="--", linewidth=1.5, alpha=0.7,
                        label=f"BSM ($\sigma={sigma_val}$)" if (row_idx == 0 and col_idx == 0) else "")
                ax.plot(S_sweep, nn_line_data, color=v_color, linestyle="-", linewidth=2.0,
                        label=f"NN ($\sigma={sigma_val}$)" if (row_idx == 0 and col_idx == 0) else "")
            
            ax.axvline(K_fixed, color="black", linestyle=":", alpha=0.3)
            ax.set_ylim([-0.05, 1.05])
            ax.grid(True, linestyle="-", alpha=0.25, color="grey")
            
            for spine in ["top", "right", "left", "bottom"]:
                ax.spines[spine].set_color("black")
                ax.spines[spine].set_linewidth(1.0)

            if row_idx == 0:
                ax.set_title(col_meta["name"], fontsize=12, fontweight="bold")
            if col_idx == 0:
                ax.set_ylabel(f"{time_lbl}\n\nHedging Delta ($\Delta$)")
            if row_idx == 2:
                ax.set_xlabel("Stock Price ($S_t$)")

    fig.legend(loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.03), frameon=True, fontsize=10)
    plt.savefig(project_path("results/figures/delta_3x2_4input_matrix.png"), dpi=150, bbox_inches="tight")
    plt.show()


def plot_delta_2input_matrix_3x2():
    """
    Generates a 3x2 grid comparing the 2-Input Daily vs 2-Input Monthly models.
    Rows: Time Slices (t = 0.0, 0.5, 0.9)
    Cols: 2-Input Daily, 2-Input Monthly
    """
    device = get_torch_device()
    K_fixed = 100.0
    S_sweep = np.linspace(80, 120, 150)
    S_scaled = S_sweep / K_fixed

    time_slices = {"$t = 0.0$": 0.0, "$t = 0.5T$": 0.5 * T, "$t = 0.9T$": 0.9 * T}
    vol_slices = [0.1, 0.2, 0.3]
    vol_colors = {0.1: "#007AFF", 0.2: "#FF9500", 0.3: "#AF52DE"}

    net_2in_daily   = load_model("base_daily").to(device)
    net_2in_monthly = load_model("base_monthly").to(device)
    net_2in_daily.eval()
    net_2in_monthly.eval()

    fig, axes = plt.subplots(3, 2, figsize=(11, 12), constrained_layout=True)

    model_cols = [
        {"name": "2-Input Model Daily", "net": net_2in_daily},
        {"name": "2-Input Model Monthly", "net": net_2in_monthly}
    ]

    for row_idx, (time_lbl, t_val) in enumerate(time_slices.items()):
        for col_idx, col_meta in enumerate(model_cols):
            ax = axes[row_idx, col_idx]
            net = col_meta["net"]
            
            for sigma_val in vol_slices:
                nn_line_data = []
                bsm_line_data = []
                v_color = vol_colors[sigma_val]
                
                d_bsm = bsm_delta(S_scaled, 1.0, 0.0, sigma_val, t_val, T)
                
                for idx, S_norm in enumerate(S_scaled):
                    bsm_line_data.append(d_bsm[idx])
                    feat = torch.tensor([[S_norm, t_val]], dtype=torch.float32).to(device)
                    with torch.no_grad():
                        nn_line_data.append(net(feat).cpu().item())
                
                ax.plot(S_sweep, bsm_line_data, color=v_color, linestyle="--", linewidth=1.5, alpha=0.7,
                        label=f"BSM ($\sigma={sigma_val}$)" if (row_idx == 0 and col_idx == 0) else "")
                # Note: These NN solid lines will stack cleanly on top of one another!
                ax.plot(S_sweep, nn_line_data, color=v_color, linestyle="-", linewidth=2.0,
                        label=f"NN ($\sigma={sigma_val}$)" if (row_idx == 0 and col_idx == 0) else "")
            
            ax.axvline(K_fixed, color="black", linestyle=":", alpha=0.3)
            ax.set_ylim([-0.05, 1.05])
            ax.grid(True, linestyle="-", alpha=0.25, color="grey")
            
            for spine in ["top", "right", "left", "bottom"]:
                ax.spines[spine].set_color("black")
                ax.spines[spine].set_linewidth(1.0)

            if row_idx == 0:
                ax.set_title(col_meta["name"], fontsize=12, fontweight="bold")
            if col_idx == 0:
                ax.set_ylabel(f"{time_lbl}\n\nHedging Delta ($\Delta$)")
            if row_idx == 2:
                ax.set_xlabel("Stock Price ($S_t$)")

    fig.legend(loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.03), frameon=True, fontsize=10)
    plt.savefig(project_path("results/figures/delta_3x2_2input_matrix.png"), dpi=150, bbox_inches="tight")
    plt.show()



plt.rcParams.update({
    "font.size":        11,
    "axes.titlesize":   12,
    "axes.labelsize":   11,
    "xtick.labelsize":  11,
    "ytick.labelsize":  11,
    "legend.fontsize":  11,
})

def plot_pnl_by_moneyness():
    device = get_torch_device()
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)

    N_PATHS = 5000
    K_BINS = [
        ("OTM", 106.7, 120.0),
        ("ATM",  93.3, 106.7),
        ("ITM",  80.0,  93.3),
    ]
    BUCKET_COLORS = {
        "OTM": "#FF9500",
        "ATM": "#007AFF",
        "ITM": "#34C759",
    }

    daily_models = [
        {"key": "BSM", "label": "BS Hedge",      "type": "bsm", "name": "base_daily"},
        {"key": "2in", "label": "2-Input Model",  "type": "nn",  "name": "base_daily"},
        {"key": "4in", "label": "4-Input Model",  "type": "nn",  "name": "base_relvol_bsdelta_daily"},
    ]
    monthly_models = [
        {"key": "BSM", "label": "BS Hedge",      "type": "bsm", "name": "base_monthly"},
        {"key": "2in", "label": "2-Input Model",  "type": "nn",  "name": "base_monthly"},
        {"key": "4in", "label": "4-Input Model",  "type": "nn",  "name": "base_relvol_bsdelta_monthly"},
    ]

    def get_pnl_for_bucket(model_meta, k_lo, k_hi):
        params     = get_model_params(model_meta["name"])
        N          = params["N"]
        N_FEATURES = params["N_FEATURES"]

        features, K, Sigma = generate_gbm_augmented(
            S0       = S0,
            K_lo     = k_lo,
            K_hi     = k_hi,
            sigma_lo = SIGMA_LO,
            sigma_hi = SIGMA_HI,
            N        = N,
            T        = T,
            paths    = N_PATHS,
            rng      = rng,
        )

        if model_meta["type"] == "nn":
            net = load_model(model_meta["name"]).to(device)
            net.eval()
            feat = features[:, :, :N_FEATURES].to(device)
            with torch.no_grad():
                pnl = run_paths(feat, net).cpu().numpy()
        else:
            bsm_deltas = features[:, :N, 3].numpy()
            pnl        = bsm_hedge_pnl(features, Sigma, bsm_deltas)

        return float(np.mean(pnl)), float(np.std(pnl))

    def compute_results(model_list):
        results = {m["key"]: {} for m in model_list}
        for m in model_list:
            for bucket_label, k_lo, k_hi in K_BINS:
                print(f"  {m['key']} | {bucket_label} (K in [{k_lo}, {k_hi}])")
                mean, std = get_pnl_for_bucket(m, k_lo, k_hi)
                results[m["key"]][bucket_label] = (mean, std)
        return results

    print("Computing daily model results...")
    daily_results   = compute_results(daily_models)
    print("Computing monthly model results...")
    monthly_results = compute_results(monthly_models)

    def draw_subplot(ax, model_list, results, title, show_xlabel):
        bucket_labels = [b[0] for b in K_BINS]
        n_buckets     = len(bucket_labels)
        group_width   = 0.7
        bar_width     = group_width / n_buckets
        offsets       = np.linspace(
            -group_width / 2 + bar_width / 2,
             group_width / 2 - bar_width / 2,
            n_buckets,
        )

        for m_idx, m in enumerate(model_list):
            for b_idx, bucket_label in enumerate(bucket_labels):
                mean, std = results[m["key"]][bucket_label]
                x     = m_idx + offsets[b_idx]
                color = BUCKET_COLORS[bucket_label]
                ax.bar(x, mean, width=bar_width * 0.92, color=color, zorder=3)
                ax.errorbar(
                    x, mean, yerr=std,
                    fmt="none", color="black",
                    capsize=3, linewidth=1.2, zorder=4,
                )

        ax.axhline(0, color="black", linewidth=0.9, linestyle="--")
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel("P&L")
        ax.set_xticks(range(len(model_list)))
        ax.set_xticklabels([m["label"] for m in model_list])
        if show_xlabel:
            ax.set_xlabel("Model")
        ax.grid(True, axis="y", linestyle="-", alpha=0.25, color="grey")
        ax.set_axisbelow(True)
        for spine in ["top", "right", "left", "bottom"]:
            ax.spines[spine].set_color("black")
            ax.spines[spine].set_linewidth(1.0)

    fig, axes = plt.subplots(2, 1, figsize=(6.3, 8), constrained_layout=True)
    draw_subplot(axes[0], daily_models,   daily_results,   "Daily Rebalancing",   show_xlabel=False)
    draw_subplot(axes[1], monthly_models, monthly_results, "Monthly Rebalancing", show_xlabel=True)

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=BUCKET_COLORS[b], label=b)
        for b in ["OTM", "ATM", "ITM"]
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, -0.04),
        frameon=True,
    )

    plt.savefig(project_path("results/figures/pnl_by_moneyness.png"), dpi=150, bbox_inches="tight")
    plt.show()
    print("Moneyness P&L plot saved.")

# forest_plot()
# forest_plot_nn_only()
# plot_premium_curves_vs_S0()
# plot_delta_slices_vibrant_by_vol()
# plot_delta_4input_matrix_3x2()
# plot_delta_2input_matrix_3x2()
plot_pnl_by_moneyness()
import numpy as np
import torch
import matplotlib.pyplot as plt

from project.helpers.path_helpers import project_path
from project.helpers.helpers import get_torch_device
from project.minimal.constants import S0, K_LO, K_HI, SIGMA_LO, SIGMA_HI, T, DEVICE, get_model_params
from project.minimal.model import HedgingNet
from project.stock.generators import generate_gbm_augmented
from project.minimal.new_pricer import run_paths
from project.minimal.bs_model import bsm_call, bsm_delta, bsm_hedge_pnl
from project.minimal.model_name import MODEL_NAME

SEED = 23
CONFIDENCE_THRESHOLD = 0.001  # max acceptable CI half-width (std of bootstrap estimates)
N_PATHS_TEST = 1000


# these are just to satisfy intellisense
# HIDDEN_NEURONS = 0
# HIDDEN_LAYERS = 0
# LEARNING_RATE = 0
# BATCH_SIZE = 0
# N_EPOCHS = 0
# ACTIVATION_PARAM = 0
# GRAD_CLIP_THRESHOLD = 0
# N_FEATURES = 0
# N = 0
# H = 0

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



def load_model(model_name) -> HedgingNet:
    DEVICE = get_torch_device()
    path       = project_path(f"results/models/{model_name}_hedging_model.pt")
    checkpoint = torch.load(path, map_location=DEVICE)

    params             = get_model_params(model_name)
    hidden_neurons     = params["HIDDEN_NEURONS"]
    hidden_layers      = params["HIDDEN_LAYERS"]
    activation_param   = params["ACTIVATION_PARAM"]
    n_features         = params["N_FEATURES"]

    hedging_net = HedgingNet(n_features, hidden_neurons, hidden_layers, activation_param).to(DEVICE)
    hedging_net.load_state_dict(checkpoint["hedging_net_state"])
    hedging_net = hedging_net.to(DEVICE)
    return hedging_net



def test_model(model_name: str):

    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)

    params             = get_model_params(model_name)
    N_FEATURES         = params["N_FEATURES"]
    N                  = params["N"]
    hedging_net = load_model(model_name)
    
    pnl_samples = []

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
        # bsm_deltas = features[:, :N, 3].numpy()
        features   = features[:, :, :N_FEATURES]
        features = features.to(get_torch_device())

        batch_pnl = run_paths(features, hedging_net).cpu().detach().numpy()  # 1D array
        pnl_samples.extend(batch_pnl)

        data = np.array(pnl_samples)
        _, mean_se = bootstrap(data, np.mean)
        _, std_se  = bootstrap(data, np.std)

        print(f"n={len(data):>6}  mean_SE={mean_se:.5f}  std_SE={std_se:.5f}")

        if mean_se < CONFIDENCE_THRESHOLD and std_se < CONFIDENCE_THRESHOLD:
            break

    final_mean, mean_se = bootstrap(data, np.mean)
    final_std, std_se = bootstrap(data, np.std)
    return {"mean": final_mean, "std": final_std, "n": len(data), "mean_se": mean_se, "std_se": std_se}


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
        # 1. Generate paths (Always pull full 4 features to get the BSM delta array)
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
        
        # 2. Extract pre-computed rolling BSM deltas (Shape: paths, N)
        bsm_deltas = features[:, :N, 3].numpy()

        # 3. Roll out the realistic BSM hedging simulation
        batch_pnl = bsm_hedge_pnl(features, Sigma, bsm_deltas)
        bsm_pnl_samples.extend(batch_pnl)

        # 4. Monitor bootstrap convergence
        data = np.array(bsm_pnl_samples)
        _, mean_se = bootstrap(data, np.mean)
        _, std_se  = bootstrap(data, np.std)

        print(f"n={len(data):>6}  mean_SE={mean_se:.5f}  std_SE={std_se:.5f}")

        # Break when the standard errors drop below your threshold
        if mean_se < CONFIDENCE_THRESHOLD and std_se < CONFIDENCE_THRESHOLD:
            break

    final_mean, mean_se = bootstrap(data, np.mean)
    final_std, std_se   = bootstrap(data, np.std)
    
    return {
        "mean": final_mean, 
        "std": final_std, 
        "n": len(data), 
        "mean_se": mean_se, 
        "std_se": std_se
    }


def generate_and_plot_distributions():
    """
    Evaluates all 6 strategies over a massive, identical out-of-sample path block
    and exports a 2x3 normalized PDF histogram grid layout.
    """
    # 1. Use a fixed, massive test size to capture clean distributions
    n_plot_paths = 50000
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)
    
    # 2. Define our structural grid matrix layout
    frequencies = ["daily", "monthly"]
    models = ["base", "4 input", "BS"]
    
    # Pre-allocate dictionary arrays to store raw values for identical graphing
    pnl_data = {f"{freq}_{model}": [] for freq in frequencies for model in models}
    
    print(f"Generating out-of-sample PnL profiles across {n_plot_paths} paths...")
    
    # Loop over frequencies to run proper underlying generators (Daily=252 vs Monthly=12)
    for freq in frequencies:
        N_steps = 252 if freq == "daily" else 12
        
        # Generate raw data path block for this frequency matrix
        features_raw, K, Sigma = generate_gbm_augmented(
            S0=S0, K_lo=K_LO, K_hi=K_HI, sigma_lo=SIGMA_LO, sigma_hi=SIGMA_HI,
            N=N_steps, T=T, paths=n_plot_paths, rng=rng,
        )
        
        # Extract the pre-computed BSM deltas (Shape: paths, N)
        bsm_deltas = features_raw[:, :N_steps, 3].numpy()
        
    
        
        # --- B. Evaluate the 2-Input Base NN Model ---
        model_name_base = f"base_{freq}"
        net_base = load_model(model_name_base)
        net_base.eval()
        features_base = features_raw[:, :, :2].to(get_torch_device())
        with torch.no_grad():
            # Multiply by 100 to scale up the raw values
            pnl_data[f"{freq}_base"] = run_paths(features_base, net_base).cpu().numpy() * 100.0
            
        # --- C. Evaluate the 4-Input NN Model ---
        model_name_4in = f"base_relvol_bsdelta_{freq}"
        net_4in = load_model(model_name_4in)
        net_4in.eval()
        features_4in = features_raw[:, :, :4].to(get_torch_device())
        with torch.no_grad():
            # Multiply by 100 to scale up the raw values
            pnl_data[f"{freq}_4 input"] = run_paths(features_4in, net_4in).cpu().numpy() * 100.0

    # 3. Plotting Phase
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), sharex=True, sharey=True)
    plt.style.use('seaborn-v0_8-whitegrid') # Professional, clean grid style
    
    # Standardize colors to distinguish neural nets from classical benchmarks
    colors = {"base": "#1f77b4", "4 input": "#aec7e8", "BS": "#ff7f0e"}
    
    for row_idx, freq in enumerate(frequencies):
        for col_idx, model in enumerate(models):
            ax = axes[row_idx, col_idx]
            key = f"{freq}_{model}"
            data = pnl_data[key]
            
            # Calculate metrics to drop in the chart annotation labels
            mu = np.mean(data)
            std = np.std(data)
            
            # Draw normalized PDF histogram
            ax.hist(data, bins=100, density=True, color=colors[model], alpha=0.75, 
                    edgecolor='black', linewidth=0.3)
            
            # Draw an explicit vertical line marking expected zero-tracking reference
            ax.axvline(0.0, color='red', linestyle='--', linewidth=1.0, alpha=0.7, label='Zero PnL')
            
            # Titles and text boxes
            ax.set_title(f"{freq.capitalize()} Framework - {model.upper()}", fontsize=12, fontweight='bold')
            ax.text(0.05, 0.85, f"$\mu$: {mu:.4f}\n$\sigma$: {std:.4f}", 
                    transform=ax.transAxes, fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Layout formatting
            if row_idx == 1:
                ax.set_xlabel("Terminal PnL ($S/K$ Units)", fontsize=11)
            if col_idx == 0:
                ax.set_ylabel("Probability Density (PDF)", fontsize=11)
                
            ax.set_xlim([-20, 20]) # Center bounding symmetrically to track risk tails cleanly
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_save_path = project_path("results/figures/hedging_error_distributions.png")
    plt.savefig(plot_save_path, dpi=300)
    plt.show()
    print(f"Distribution plot successfully saved to: {plot_save_path}")


def generate_and_plot_pnl_cloud():
    """
    Generates a scatter plot comparing Terminal Stock Price vs Terminal Hedging Error (PnL)
    across all daily strategies, scaled to absolute cash terms (x100).
    """
    n_plot_paths = 50000
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)
    
    # We focus on the Daily resolution to see the granular tracking footprints
    freq = "daily"
    N_steps = 252 
    
    print(f"Generating PnL cloud profiles across {n_plot_paths} paths...")
    
    # 1. Generate raw data path block
    features_raw, K, Sigma = generate_gbm_augmented(
        S0=S0, K_lo=K_LO, K_hi=K_HI, sigma_lo=SIGMA_LO, sigma_hi=SIGMA_HI,
        N=N_steps, T=T, paths=n_plot_paths, rng=rng,
    )
    
    # Extract terminal stock prices and scale by 100
    # S_scaled is index 0 of features, terminal step is at index N_steps
    S_T_scaled = features_raw[:, N_steps, 0].numpy() * 100.0
    
    # Extract the pre-computed BSM deltas (Shape: paths, N)
    bsm_deltas = features_raw[:, :N_steps, 3].numpy()
    
    # --- A. Evaluate the BSM Model (Updated for compatibility) ---
    N = 252 if freq == "daily" else 12
    H = T / N
    pnl_bs = bsm_hedge_pnl(features_raw, Sigma, bsm_deltas, N, H) * 100.0
    
    # --- B. Evaluate the 2-Input Base NN ---
    net_base = load_model(f"base_{freq}")
    net_base.eval()
    features_base = features_raw[:, :, :2].to(get_torch_device())
    with torch.no_grad():
        pnl_base = run_paths(features_base, net_base).cpu().numpy() * 100.0
        
    # --- C. Evaluate the 4-Input NN ---
    net_4in = load_model(f"base_relvol_bsdelta_{freq}")
    net_4in.eval()
    features_4in = features_raw[:, :, :4].to(get_torch_device())
    with torch.no_grad():
        pnl_4in = run_paths(features_4in, net_4in).cpu().numpy() * 100.0

    # 3. Plotting Phase
    plt.figure(figsize=(11, 7))
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Use small alpha (transparency) and small markers since we have 50,000 points
    plt.scatter(S_T_scaled, pnl_base, color="#1f77b4", alpha=0.15, s=1.5, label="Base NN (2-Input)")
    plt.scatter(S_T_scaled, pnl_4in, color="#aec7e8", alpha=0.15, s=1.5, label="Extended NN (4-Input)")
    plt.scatter(S_T_scaled, pnl_bs, color="#ff7f0e", alpha=0.15, s=1.5, label="BSM Baseline")
    
    # Add key baseline guide lines
    plt.axhline(0.0, color='black', linestyle='-', linewidth=1.0, alpha=0.6)
    plt.axvline(100.0, color='red', linestyle='--', linewidth=1.2, alpha=0.7, label='Strike (K=100)')
    
    # Titles and formatting
    plt.title("Terminal Hedging Error (PnL) Cloud vs. Terminal Asset Price", fontsize=13, fontweight='bold')
    plt.xlabel("Terminal Stock Price ($S_T$ scaled to $R100$ reference)", fontsize=11)
    plt.ylabel("Terminal Hedging Error (PnL in Cash Units)", fontsize=11)
    
    # Set reasonable axis bounds to keep outliers from blowing out the scale
    plt.xlim([50, 160])
    plt.ylim([-25, 25])
    
    # Refined legend layout to handle transparency visibility in labels
    lgnd = plt.legend(loc="upper right", frameon=True, fontsize=10)
    # for handle in lgnd.legend_handles:
        # handle.set_sizes([20.0]) # Make legend dots larger and solid
        # handle.set_alpha(1.0)
        
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_save_path = project_path("results/figures/pnl_cloud_scatter.png")
    plt.savefig(plot_save_path, dpi=300)
    plt.show()
    print(f"PnL cloud scatter plot successfully saved to: {plot_save_path}")


def generate_and_plot_pnl_cloud_grid():
    """
    Generates a 2x3 grid of scatter plots comparing Terminal Stock Price vs Terminal PnL,
    scaled to absolute cash terms (x100), across all 6 model variations.
    """
    n_plot_paths = 50000
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)
    
    frequencies = ["daily", "monthly"]
    models = ["base", "4 input", "BS"]
    
    # Dictionaries to hold data vectors matching grid coordinates
    pnl_data = {}
    st_data = {}
    
    print(f"Generating out-of-sample PnL cloud coordinate arrays across {n_plot_paths} paths...")
    
    for freq in frequencies:
        N_steps = 252 if freq == "daily" else 12
        
        # 1. Generate fresh paths matching the specific frequency scope
        features_raw, K, Sigma = generate_gbm_augmented(
            S0=S0, K_lo=K_LO, K_hi=K_HI, sigma_lo=SIGMA_LO, sigma_hi=SIGMA_HI,
            N=N_steps, T=T, paths=n_plot_paths, rng=rng,
        )
        
        # Extract terminal asset prices and multiply by 100
        S_T_scaled = features_raw[:, N_steps, 0].numpy() * 100.0
        st_data[f"{freq}_base"] = S_T_scaled
        st_data[f"{freq}_4 input"] = S_T_scaled
        st_data[f"{freq}_BS"] = S_T_scaled
        
        # Extract pre-computed BSM deltas
        bsm_deltas = features_raw[:, :N_steps, 3].numpy()
        
        # --- A. Evaluate the BSM Model ---
        N = 252 if freq == "daily" else 12
        H = T / N
        pnl_data[f"{freq}_BS"] = bsm_hedge_pnl(features_raw, Sigma, bsm_deltas, N, H) * 100.0
        
        # --- B. Evaluate the 2-Input Base NN Model ---
        net_base = load_model(f"base_{freq}")
        net_base.eval()
        features_base = features_raw[:, :, :2].to(get_torch_device())
        with torch.no_grad():
            pnl_data[f"{freq}_base"] = run_paths(features_base, net_base).cpu().numpy() * 100.0
            
        # --- C. Evaluate the 4-Input NN Model ---
        net_4in = load_model(f"base_relvol_bsdelta_{freq}")
        net_4in.eval()
        features_4in = features_raw[:, :, :4].to(get_torch_device())
        with torch.no_grad():
            pnl_data[f"{freq}_4 input"] = run_paths(features_4in, net_4in).cpu().numpy() * 100.0

    # 2. Grid Plotting Matrix Setup
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), sharex=True, sharey=True)
    plt.style.use('seaborn-v0_8-whitegrid')
    
    colors = {"base": "#1f77b4", "4 input": "#aec7e8", "BS": "#ff7f0e"}
    
    for row_idx, freq in enumerate(frequencies):
        for col_idx, model in enumerate(models):
            ax = axes[row_idx, col_idx]
            key = f"{freq}_{model}"
            
            x = st_data[key]
            y = pnl_data[key]
            
            # Draw the scattering points (Tiny size 's' and soft transparency 'alpha' are critical)
            ax.scatter(x, y, color=colors[model], alpha=0.3, s=1, edgecolors='none')
            
            # Ground-truth guide markers
            ax.axhline(0.0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
            ax.axvline(100.0, color='red', linestyle='--', linewidth=1.0, alpha=0.6, label='Strike (K=100)')
            
            # Descriptive text metrics box
            mu = np.mean(y)
            std = np.std(y)
            ax.text(0.05, 0.05, f"$\mu$: {mu:.2f}\n$\sigma$: {std:.2f}", 
                    transform=ax.transAxes, fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Axis Title and labels
            ax.set_title(f"{freq.capitalize()} - {model.upper()}", fontsize=12, fontweight='bold')
            
            if row_idx == 1:
                ax.set_xlabel("Terminal Stock Price ($S_T$)", fontsize=11)
            if col_idx == 0:
                ax.set_ylabel("Terminal Hedging Error (PnL)", fontsize=11)
                
            # Keep structural limits tightly bound for comparison
            ax.set_xlim([50, 160])
            ax.set_ylim([-25, 25])
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_save_path = project_path("results/figures/pnl_cloud_grid.png")
    plt.savefig(plot_save_path, dpi=300)
    plt.show()
    print(f"6-plot PnL cloud matrix successfully saved to: {plot_save_path}")


# print(test_model(MODEL_NAME))
# print(test_bsm_model("base_daily"))
# generate_and_plot_distributions()
generate_and_plot_pnl_cloud_grid()

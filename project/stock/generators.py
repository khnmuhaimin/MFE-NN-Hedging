import numpy as np
import torch
from scipy.stats import norm

def generate_gbm(S0, mu, sigma, dt, paths, timesteps, rng=None):
    """
    Simulates multiple paths of a geometric Brownian motion (GBM).
    Each path is stored in a column.
    """
    if rng is None:
        rng = np.random.default_rng()
    Z = rng.standard_normal((timesteps - 1, paths))
    log_returns = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
    cum_returns = np.exp(np.cumsum(log_returns, axis=0))
    S = S0 * np.vstack((np.ones((1, paths)), cum_returns))
    return S


def generate_gbm_augmented(
    S0,
    K_lo,
    K_hi,
    sigma_lo,
    sigma_hi,
    N,
    T,
    paths,
    rng=None,
):
    if rng is None:
        rng = np.random.default_rng()
 
    # Force base simulation at the daily resolution (N=252)
    N_sim = 252
    timesteps_sim = N_sim + 1
    dt_sim = T / N_sim
 
    # ── Sample per-path parameters ────────────────────────────────────────
    # Kept as (1, paths) for internal broadcasting; flattened to 1D at the end
    K_vals     = rng.uniform(K_lo,    K_hi,    size=(1, paths))   
    sigma_vals = rng.uniform(sigma_lo, sigma_hi, size=(1, paths)) 
 
    # ── Simulate log-normal paths (Always at N=252) ───────────────────────
    Z          = rng.standard_normal((N_sim, paths))                  
    drift      = (0.0 - 0.5 * sigma_vals ** 2) * dt_sim              
    diffusion  = sigma_vals * np.sqrt(dt_sim) * Z
    log_returns = drift + diffusion                               
    cum_returns = np.exp(np.cumsum(log_returns, axis=0))          
    S = S0 * np.vstack((np.ones((1, paths)), cum_returns))        # (timesteps_sim, paths)
 
    # ── Feature 1: normalised price S_t / K ──────────────────────────────
    S_scaled = S / K_vals                                         
 
    # ── Feature 2: time to maturity  T - t ───────────────────────────────
    t_indices = np.arange(timesteps_sim).reshape(-1, 1)               
    time_left = np.broadcast_to(
        np.maximum(T - t_indices * dt_sim, 0.0), (timesteps_sim, paths)
    ).copy()                                                       
 
    # ── Feature 3: realised volatility (annualised) ───────────────────────
    realised_vol = np.zeros((timesteps_sim, paths))
    for t in range(2, timesteps_sim):
        past = log_returns[:t, :]                                 
        realised_vol[t] = np.std(past, axis=0, ddof=1) / np.sqrt(dt_sim)
 
    # ── Feature 4: Black-Scholes delta ────────────────────────────────────
    time_left_safe = np.maximum(time_left, 1e-8)
    vol_safe       = np.where(realised_vol > 0, realised_vol, 1e-8)
    d1             = (np.log(S_scaled) + 0.5 * vol_safe ** 2 * time_left_safe) \
                     / (vol_safe * np.sqrt(time_left_safe))
    bs_delta       = norm.cdf(d1)     
    bs_delta[0:3, :] = 0.5                            
 
    # ── Downsampling Logic (If N=12) ──────────────────────────────────────
    if N == 12:
        # 252 steps / 12 intervals = 21 steps per interval
        step_size = N_sim // N  # Exactly 21
        indices = np.arange(0, timesteps_sim, step_size) # [0, 21, 42, ..., 252]
        
        # Downsample along the time axis (axis 0)
        S_scaled     = S_scaled[indices, :]
        time_left    = time_left[indices, :]
        realised_vol = realised_vol[indices, :]
        bs_delta     = bs_delta[indices, :]
 
    # ── Pack into a single float32 tensor: (paths, timesteps, 4) ─────────
    features_np = np.stack(
        [S_scaled, time_left, realised_vol, bs_delta], axis=-1
    )                                                             
    features = torch.from_numpy(
        features_np.transpose(1, 0, 2)                           
    ).float()
 
    # Flatten parameter arrays to clean 1D profiles
    K_arr = K_vals.squeeze(0)
    sigma_arr = sigma_vals.squeeze(0)
 
    return features, K_arr, sigma_arr
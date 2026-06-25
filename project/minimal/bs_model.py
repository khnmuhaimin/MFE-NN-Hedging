"""
Utility functions for the Black-Scholes model
"""


import numpy as np
import torch
from scipy.stats import norm
from project.minimal.constants import T, get_model_params
from project.minimal.model_name import MODEL_NAME


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


def bsm_hedge_pnl(features: torch.Tensor, sigmas: np.ndarray, bsm_deltas: np.ndarray, N=None, H=None) -> np.ndarray:
    """
    Realistic BSM delta hedge benchmark.
    Uses the historical rolling realized volatilities/deltas available at each step 
    rather than perfect lookahead knowledge of the true path volatility.
    """
    if N is None:
        N = get_model_params(MODEL_NAME)["N"]
    if H is None:
        H = get_model_params(MODEL_NAME)["H"]
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
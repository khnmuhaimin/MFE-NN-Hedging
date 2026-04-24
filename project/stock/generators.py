import numpy as np

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


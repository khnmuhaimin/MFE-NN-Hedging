import numpy as np
import matplotlib.pyplot as plt

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


def parameter_to_np(parameter, size, rng=None):
    
    if np.isscalar(parameter):
        return np.full(size, parameter)
    
    if rng is None:
        rng = np.random.default_rng()

    if isinstance(parameter, (list, tuple, np.ndarray)) and len(parameter) == 2:
        a, b = parameter
        return rng.uniform(a, b, size=size)

    raise ValueError(
        "Parameter must be either a scalar or a length-2 range [a, b]."
    )
    

def generate_heston_vol_parameters(var_reversion_speed, mean_var, vol_of_vol, num_triplets, max_iterations=1000, rng=None):

    if rng is None:
        rng = np.random.default_rng()

    # Validate that the feasible region is non-empty
    kappa_max = var_reversion_speed if np.isscalar(var_reversion_speed) else var_reversion_speed[1]
    theta_max = mean_var if np.isscalar(mean_var) else mean_var[1]
    xi_min = vol_of_vol if np.isscalar(vol_of_vol) else vol_of_vol[0]

    if 2 * kappa_max * theta_max <= xi_min**2:
        raise ValueError(
            f"Feller condition cannot be satisfied: "
            f"2 * kappa_max * theta_max = {2 * kappa_max * theta_max:.4f} "
            f"<= xi_min^2 = {xi_min**2:.4f}"
        )

    parameters = np.empty((3, num_triplets))
    generated = 0
    iterations = 0

    while generated < num_triplets and iterations < max_iterations:
        to_generate = num_triplets - generated

        kappa = parameter_to_np(var_reversion_speed, to_generate, rng)
        theta = parameter_to_np(mean_var, to_generate, rng)
        xi = parameter_to_np(vol_of_vol, to_generate, rng)

        valid = 2 * kappa * theta > xi**2
        n_valid = valid.sum()

        if n_valid > 0:
            parameters[0, generated:generated + n_valid] = kappa[valid]
            parameters[1, generated:generated + n_valid] = theta[valid]
            parameters[2, generated:generated + n_valid] = xi[valid]
            generated += n_valid

        iterations += 1

    if generated < num_triplets:
        raise RuntimeError(
            f"Only generated {generated}/{num_triplets} valid triplets after "
            f"{max_iterations} iterations. Consider widening your parameter ranges "
            f"or increasing max_iterations."
        )

    return parameters[0], parameters[1], parameters[2]
        


def generate_heston(initial_price,
                    drift,
                    initial_var,
                    mean_var,
                    var_reversion_speed,
                    vol_of_vol,
                    var_stock_correlation,
                    dt,
                    num_paths,
                    num_timesteps,
                    rng=None):
    if rng is None:
        rng = np.random.default_rng()

    t = np.arange(num_timesteps) * dt
    mu = parameter_to_np(drift, num_paths, rng)
    v0 = parameter_to_np(initial_var, num_paths, rng)
    rho = parameter_to_np(var_stock_correlation, num_paths, rng)
    kappa, theta, xi = generate_heston_vol_parameters(var_reversion_speed,
                                                      mean_var,
                                                      vol_of_vol,
                                                      num_triplets=num_paths,
                                                      rng=rng)
    


    Z1 = rng.standard_normal((num_timesteps - 1, num_paths))
    Z2 = rng.standard_normal((num_timesteps - 1, num_paths))
    dW_s = np.sqrt(dt) * Z1
    dW_v = np.sqrt(dt) * (rho * Z1 + np.sqrt(1 - rho**2) * Z2)

    # v_{t+Δt}​=vt+κ(θ−vt​)Δt  +  ξ*sqrt(vt)*sqrt(​Δt)*​Z2​  +  1/4*​ξ^2[(sqrt(Δt)*​Z2​)^2−Δt]
    # we only need num_timesteps-1 var values because the first stock price entry is fixed to 1.
    variance = np.empty((num_timesteps-1, num_paths))
    variance[0,:] = v0
    for i in range(1, num_timesteps-1):
        variance[i,:] = (variance[i-1,:]
                         + kappa*(theta - variance[i-1,:])*dt
                         + xi*np.sqrt(variance[i-1,:])*dW_v[i-1,:]
                         + 0.25*xi**2*(dW_v[i-1,:]**2 - dt))
        variance[i,:] = np.maximum(variance[i,:], 0)

    stock_prices = np.empty((num_timesteps, num_paths))
    stock_prices[0, :] = 1.0
    log_returns = (mu - 0.5 * variance) * dt + np.sqrt(variance) * dW_s
    stock_prices[1:, :] = np.exp(np.cumsum(log_returns, axis=0))
    stock_prices *= initial_price

    # plt.plot(stock_prices[:,0:5])
    # plt.show()

    return stock_prices

def generate_bates(initial_price,
                   drift,
                   initial_var,
                   mean_var,
                   var_reversion_speed,
                   vol_of_vol,
                   var_stock_correlation,
                   jump_intensity,
                   jump_mean,
                   jump_std,
                   dt,
                   num_paths,
                   num_timesteps,
                   rng=None):
    if rng is None:
        rng = np.random.default_rng()

    mu = parameter_to_np(drift, num_paths, rng)
    v0 = parameter_to_np(initial_var, num_paths, rng)
    rho = parameter_to_np(var_stock_correlation, num_paths, rng)
    lam = parameter_to_np(jump_intensity, num_paths, rng)
    mu_j = parameter_to_np(jump_mean, num_paths, rng)
    sigma_j = parameter_to_np(jump_std, num_paths, rng)

    kappa, theta, xi = generate_heston_vol_parameters(var_reversion_speed,
                                                       mean_var,
                                                       vol_of_vol,
                                                       num_triplets=num_paths,
                                                       rng=rng)

    Z1 = rng.standard_normal((num_timesteps - 1, num_paths))
    Z2 = rng.standard_normal((num_timesteps - 1, num_paths))
    dW_s = np.sqrt(dt) * Z1
    dW_v = np.sqrt(dt) * (rho * Z1 + np.sqrt(1 - rho**2) * Z2)

    variance = np.empty((num_timesteps - 1, num_paths))
    variance[0, :] = v0
    for i in range(1, num_timesteps - 1):
        variance[i, :] = (variance[i-1, :]
                          + kappa * (theta - variance[i-1, :]) * dt
                          + xi * np.sqrt(variance[i-1, :]) * dW_v[i-1, :]
                          + 0.25 * xi**2 * (dW_v[i-1, :]**2 - dt))
        variance[i, :] = np.maximum(variance[i, :], 0)

    jump_counts = rng.poisson(lam * dt, size=(num_timesteps - 1, num_paths))
    jump_sizes = rng.normal(mu_j, sigma_j, size=(num_timesteps - 1, num_paths))
    jumps = jump_counts * jump_sizes

    stock_prices = np.empty((num_timesteps, num_paths))
    stock_prices[0, :] = 1.0
    log_returns = (mu - 0.5 * variance) * dt + np.sqrt(variance) * dW_s + jumps
    stock_prices[1:, :] = np.exp(np.cumsum(log_returns, axis=0))
    stock_prices *= initial_price

    # plt.plot(stock_prices[:,0:5])
    # plt.show()

    return stock_prices


# generate_heston(
#     100,
#     drift=(0.02, 0.05),
#     initial_var=(0.01, 0.09),
#     mean_var=(0.01, 0.09),
#     var_reversion_speed=(0.5, 5.0),
#     vol_of_vol=(0.1, 0.8),
#     var_stock_correlation=(-0.9, -0.3),
#     dt=1/252,
#     num_paths=10,
#     num_timesteps=252
# )

# prices = generate_bates(
#     100,
#     drift=(0.0, 0.1),
#     initial_var=(0.01, 0.09),
#     mean_var=(0.01, 0.09),
#     var_reversion_speed=(0.5, 5.0),
#     vol_of_vol=(0.1, 0.8),
#     var_stock_correlation=(-0.9, -0.3),
#     jump_intensity=(0.5, 2.0),
#     jump_mean=(-0.1, 0.0),
#     jump_std=(0.05, 0.2),
#     dt=1/252,
#     num_paths=1,
#     num_timesteps=10000
# )


# returns = np.diff(np.log(prices.ravel()))
# plt.hist(returns, bins=50)
# plt.show()
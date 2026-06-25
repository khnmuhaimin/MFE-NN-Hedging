"""
Dual-network model architecture for deep hedging with learned option pricing.

This module defines a single nn.Module that encapsulates two separate networks:
a hedging network that outputs a delta at each rebalancing timestep, and a
pricing network that outputs an initial option premium at t=0. A single
is_initial flag in the forward pass routes the input to the appropriate network,
allowing both networks to be trained jointly under a single optimizer.
"""

import torch
import torch.nn as nn


class HedgingNet2(nn.Module):
    """
    Dual-network architecture combining a hedging network and a pricing network.

    The hedging network maps market state features at each timestep to a
    hedging delta. The pricing network maps initial moneyness at t=0 to an
    option premium. Both networks are optimised jointly via the same loss
    function and optimizer.

    Parameters
    ----------
    hedging_features : int
        Number of input features fed to the hedging network at each timestep.
        2 for the base model (S_t/K, tau), 4 for the extended model
        (S_t/K, tau, realised_vol, BS_delta).
    hedging_hidden_neurons : int
        Number of neurons in each hidden layer of the hedging network.
    hedging_depth : int
        Total number of hidden layers in the hedging network. Must be at least 1.
    gamma : float
        Reserved activation parameter, kept for compatibility with the
        configuration loading interface. Currently unused.
    pricing_hidden_neurons : int
        Number of neurons in each hidden layer of the pricing network.
    pricing_depth : int
        Total number of hidden layers in the pricing network. Must be at least 1.
    """

    def __init__(
        self,
        hedging_features: int,
        hedging_hidden_neurons: int,
        hedging_depth: int,
        gamma: float,  # not used
        pricing_hidden_neurons: int,
        pricing_depth: int,
    ):
        super().__init__()

        # ── Hedging network ───────────────────────────────────────────────────
        # Maps market state features at each timestep to a scalar delta.
        # Architecture mirrors the single-network HedgingNet for consistency.
        hedge_layers = []

        # Input layer: projects from feature space into hidden representation.
        hedge_layers.append(nn.Linear(hedging_features, hedging_hidden_neurons))
        hedge_layers.append(nn.ReLU())

        # Additional hidden layers; runs hedging_depth-1 times so that
        # hedging_depth=1 produces a single hidden layer with no additions here.
        for _ in range(hedging_depth - 1):
            hedge_layers.append(nn.Linear(hedging_hidden_neurons, hedging_hidden_neurons))
            hedge_layers.append(nn.ReLU())

        # Output layer: projects to a scalar delta bounded to (0, 1) by Sigmoid,
        # which is the valid range for a call option delta.
        hedge_layers.append(nn.Linear(hedging_hidden_neurons, 1))
        hedge_layers.append(nn.Sigmoid())

        self.hedging_net = nn.Sequential(*hedge_layers)

        # ── Pricing network ───────────────────────────────────────────────────
        # Maps initial moneyness S_0/K (a single scalar) to an option premium.
        # Takes exactly 1 input since at t=0 neither network has access to a
        # realised volatility estimate.
        pricing_layers = []

        # Input layer: projects from the single moneyness input into the hidden
        # representation.
        pricing_layers.append(nn.Linear(1, pricing_hidden_neurons))
        pricing_layers.append(nn.ReLU())

        # Additional hidden layers; same depth logic as the hedging network.
        for _ in range(pricing_depth - 1):
            pricing_layers.append(nn.Linear(pricing_hidden_neurons, pricing_hidden_neurons))
            pricing_layers.append(nn.ReLU())

        # Output layer: Softplus ensures the premium is strictly positive,
        # which is a necessary condition for any call option price.
        pricing_layers.append(nn.Linear(pricing_hidden_neurons, 1))
        pricing_layers.append(nn.Softplus())

        self.pricing_net = nn.Sequential(*pricing_layers)

        # ── Premium bias initialisation ───────────────────────────────────────
        # The final linear layer's bias is set so that Softplus(bias) ≈ 0.08,
        # which approximates the BSM ATM price in normalised S/K units at the
        # midpoint volatility. This gives the optimizer a sensible starting
        # point and prevents the premium from collapsing to near zero early
        # in training. Solving ln(1 + exp(b)) = 0.08 gives b ≈ -2.4.
        final_layer = self.pricing_net[-2]
        if isinstance(final_layer, nn.Linear):
            nn.init.constant_(final_layer.bias, -2.4)

    def forward(self, x: torch.Tensor, is_initial: bool = False) -> torch.Tensor:
        """
        Route the input to the appropriate subnetwork based on is_initial.

        At t=0, is_initial=True routes to the pricing network, which expects
        initial moneyness as its only input. At all other timesteps,
        is_initial=False routes to the hedging network, which expects the
        full feature vector.

        Parameters
        ----------
        x : torch.Tensor
            If is_initial=True,  shape (..., 1)        -- initial moneyness S_0/K.
            If is_initial=False, shape (..., n_features) -- hedging state features.
        is_initial : bool, optional
            Routing flag. Default is False (hedging network).

        Returns
        -------
        torch.Tensor
            If is_initial=True:  predicted premium (...), strictly positive.
            If is_initial=False: hedging delta (...), bounded to (0, 1).
            The trailing singleton dimension is squeezed in both cases.
        """
        if is_initial:
            # Route to pricing network for initial premium prediction.
            return self.pricing_net(x).squeeze(-1)
        else:
            # Route to hedging network for delta prediction at each timestep.
            return self.hedging_net(x).squeeze(-1)
from torch import nn
import torch


class HedgingNet(nn.Module):
    """
    At each timestep the network receives N_FEATURES inputs:
        [S_t/K,  time-to-maturity,  realised_vol,  BS_delta]
    and outputs a scalar delta (position in the underlying).

    A learnable scalar `premium` is the initial option price charged by
    the hedger; it is optimised jointly with the hedge weights.
    """
    def __init__(self, n_features: int, hidden: int, depth: int, gamma: float):
        super().__init__()
        layers = []
        
        layers.append(nn.Linear(n_features, hidden))
        layers.append(nn.ReLU())
        
        # Remaining Hidden Layers (Loop runs depth - 1 times)
        for _ in range(depth - 1):
            layers.append(nn.Linear(hidden, hidden))
            layers.append(nn.ReLU())
            
        layers.append(nn.Linear(hidden, 1))
        # layers.append(nn.Hardtanh(min_val=0.0, max_val=1.0))
        layers.append(nn.Sigmoid()) # Keeps delta bounded between 0 and 1
        
        # Unpack the list into nn.Sequential
        self.net = nn.Sequential(*layers)
        self.premium = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (..., n_features)  ->  delta : (...)"""
        return self.net(x).squeeze(-1)
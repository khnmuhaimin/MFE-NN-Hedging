import torch
import torch.nn as nn

class HedgingNet2(nn.Module):
    def __init__(self, 
                 hedging_features: int, 
                 hedging_hidden_neurons: int, 
                 hedging_depth: int, 
                 gamma: float, 
                 pricing_hidden_neurons: int, 
                 pricing_depth: int):
        super().__init__()
        
        # 1. Build the Hedging Network (Unchanged)
        hedge_layers = []
        hedge_layers.append(nn.Linear(hedging_features, hedging_hidden_neurons))
        hedge_layers.append(nn.ReLU())
        for _ in range(hedging_depth - 1):
            hedge_layers.append(nn.Linear(hedging_hidden_neurons, hedging_hidden_neurons))
            hedge_layers.append(nn.ReLU())
        hedge_layers.append(nn.Linear(hedging_hidden_neurons, 1))
        hedge_layers.append(nn.Sigmoid()) 
        self.hedging_net = nn.Sequential(*hedge_layers)
        
        # 2. Build the Pricing Network dynamically using the function signature parameters
        # Input dimension is exactly 1 (Initial Moneyness S0/K)
        pricing_layers = []
        pricing_layers.append(nn.Linear(1, pricing_hidden_neurons))
        pricing_layers.append(nn.ReLU())
        for _ in range(pricing_depth - 1):
            pricing_layers.append(nn.Linear(pricing_hidden_neurons, pricing_hidden_neurons))
            pricing_layers.append(nn.ReLU())
        pricing_layers.append(nn.Linear(pricing_hidden_neurons, 1))
        pricing_layers.append(nn.Softplus()) # Replaces the old 0.0 scalar parameter
        self.pricing_net = nn.Sequential(*pricing_layers)

        final_layer = self.pricing_net[-2] 
        if isinstance(final_layer, nn.Linear):
            # Force the initial guess to start positive (e.g., 0.20)
            nn.init.constant_(final_layer.bias, -2.4)

    def forward(self, x: torch.Tensor, is_initial: bool = False) -> torch.Tensor:
        """
        Traffic controller routing logic:
        
        If is_initial=True:
            x : (..., 1) -> Initial Moneyness (S0/K)
            Returns: predicted_premium (...)
            
        If is_initial=False:
            x : (..., n_features) -> Hedging features (St/K, tau, etc.)
            Returns: delta (...)
        """
        if is_initial:
            # Route to the pricing network and squeeze the trailing dimension
            return self.pricing_net(x).squeeze(-1)
        else:
            # Route to the standard hedging network and squeeze the trailing dimension
            return self.hedging_net(x).squeeze(-1)
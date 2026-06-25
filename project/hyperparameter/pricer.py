import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import norm
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# edit REBALANCE to switch between monthly and daily
REBALANCE = "monthly"
HP = {
    "monthly": dict(N=12,  hidden=64,  depth=4, lr=0.01, batch_size=1024, clip_norm=1.0),
    "daily":   dict(N=252, hidden=128, depth=3, lr=0.01, batch_size=1024, clip_norm=0.5),
}[REBALANCE]
EPOCHS = 100
STEP_SIZE, GAMMA = 20, 0.7
N_TRAIN = 10_000
PREMIUM_LR_SCALE = 0.5
K, r, T = 1.0, 0.0, 1.0
MONEYNESS_RANGE = (0.85, 1.15)
SIGMA_RANGE = (0.1, 0.3)
DAILY_STEPS = 252
H_DAILY = T / DAILY_STEPS
N = HP["N"]
REBAL_INDICES = list(range(0, DAILY_STEPS, DAILY_STEPS // N))
REBAL_TAUS = [1.0 - i / DAILY_STEPS for i in REBAL_INDICES]
assert len(REBAL_INDICES) == N
TAG = f"relvol_{REBALANCE}"
print(f"Device: {DEVICE}   tag: {TAG}   N={N}")
print(f"Hyperparameters: {HP}\n")

# delta network with sigmoid output + premium network mapping S0/K to a per-path premium
class HedgingNet(nn.Module):

    def __init__(self, input_dim=4, hidden=128, depth=4):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden), nn.ReLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.ReLU()]
        layers += [nn.Linear(hidden, 1), nn.Sigmoid()]
        self.net = nn.Sequential(*layers)
        self.premium_net = nn.Sequential(
            nn.Linear(1, 16), nn.ReLU(),
            nn.Linear(16, 1),
        )
        nn.init.zeros_(self.premium_net[-1].weight)
        nn.init.zeros_(self.premium_net[-1].bias)

    def forward(self, x):
        return self.net(x).squeeze(-1)

    def premium_for(self, S0):
        return self.premium_net((S0 / K).unsqueeze(-1)).squeeze(-1)

# annualised realised vol from price history (torch, for network features)
def realized_vol(S_hist):

    batch, n_obs = S_hist.shape
    if n_obs < 2:
        return torch.zeros(batch, device=DEVICE)
    log_returns = torch.log(S_hist[:, 1:] / S_hist[:, :-1])

    return log_returns.std(dim=1, unbiased=False) / (H_DAILY ** 0.5)

# annualised realised vol from price history (numpy, for the benchmark)
def realized_vol_np(S_hist_np):

    if S_hist_np.shape[0] < 2:
        return np.zeros(S_hist_np.shape[1])
    log_returns = np.log(S_hist_np[1:] / S_hist_np[:-1])

    return log_returns.std(axis=0, ddof=0) / (H_DAILY ** 0.5)

# BSM delta computed from the realised vol estimate
def bs_delta_feat(St, sigma_hat, tau):

    sig = torch.clamp(sigma_hat, min=0.01)
    if tau <= 0:
        return (St > K).float()
    d1 = (torch.log(St / K) + 0.5 * sig ** 2 * tau) / (sig * tau ** 0.5)

    return 0.5 * (1.0 + torch.erf(d1 / (2 ** 0.5)))

# stack all four network inputs for the current rebalancing date
def build_state(S_hist, tau):

    batch = S_hist.shape[0]
    St = S_hist[:, -1]
    sig = realized_vol(S_hist)

    return torch.stack([
        St / K,
        torch.full((batch,), tau, device=DEVICE),
        sig,
        bs_delta_feat(St, sig, tau),
    ], dim=1)

# simulate GBM paths with random moneyness and sigma
def generate_paths(n_paths, seed=None):

    rng = np.random.default_rng(seed)
    moneynesses = rng.uniform(*MONEYNESS_RANGE, n_paths)
    sigmas = rng.uniform(*SIGMA_RANGE, n_paths)
    Z = rng.standard_normal((DAILY_STEPS, n_paths))
    W = np.cumsum(np.sqrt(H_DAILY) * Z, axis=0)
    t_grid = np.arange(1, DAILY_STEPS + 1)[:, None] * H_DAILY
    log_S = (np.log(moneynesses)[None, :]
             - 0.5 * sigmas[None, :] ** 2 * t_grid
             + sigmas[None, :] * W)
    
    return np.concatenate([moneynesses[None, :], np.exp(log_S)], axis=0), sigmas, moneynesses

# roll the hedge forward and return terminal P&L using the premium network
def run_paths(S_batch, net):

    batch = S_batch.shape[0]
    currency = torch.zeros(batch, device=DEVICE)
    underlying = torch.zeros(batch, device=DEVICE)
    prev_delta = torch.zeros(batch, device=DEVICE)
    for i, daily_idx in enumerate(REBAL_INDICES):
        S_hist = S_batch[:, :daily_idx + 1]
        delta = net(build_state(S_hist, REBAL_TAUS[i]))
        trade = delta - prev_delta
        currency -= trade * S_hist[:, -1]
        underlying += trade
        prev_delta = delta
    premium = net.premium_for(S_batch[:, 0])
    S_T = S_batch[:, DAILY_STEPS]
    payoff = torch.clamp(S_T - K, min=0)

    return premium + underlying * S_T + currency - payoff

# BSM call price
def bsm_call_vec(S0_vec, sigma_vec):

    d1 = (np.log(S0_vec / K) + 0.5 * sigma_vec ** 2 * T) / (sigma_vec * np.sqrt(T))
    d2 = d1 - sigma_vec * np.sqrt(T)

    return S0_vec * norm.cdf(d1) - K * norm.cdf(d2)

# BSM delta
def bsm_delta_vec(S, sigma_vec, t):
    tau = T - t
    if tau <= 0:
        return np.where(S > K, 1.0, 0.0)
    d1 = (np.log(S / K) + 0.5 * sigma_vec ** 2 * tau) / (sigma_vec * np.sqrt(tau))

    return norm.cdf(d1)

# oracle BSM benchmark using true per-path sigma and per-path premium
def oracle_bsm_hedge_pnl(S, sigmas, moneynesses):

    n_paths = S.shape[1]
    currency = np.zeros(n_paths)
    underlying = np.zeros(n_paths)
    prev_delta = np.zeros(n_paths)
    premiums = bsm_call_vec(moneynesses, sigmas)
    for daily_idx in REBAL_INDICES:
        St = S[daily_idx, :]
        delta = bsm_delta_vec(St, sigmas, daily_idx * H_DAILY)
        trade = delta - prev_delta
        currency -= trade * St
        underlying += trade
        prev_delta = delta
    S_T = S[DAILY_STEPS, :]

    return premiums + underlying * S_T + currency - np.maximum(S_T - K, 0)

# practitioner BSM benchmark using realised vol and per-path premium at sigma=0.2
def practitioner_bsm_hedge_pnl(S, moneynesses):

    n_paths = S.shape[1]
    currency = np.zeros(n_paths)
    underlying = np.zeros(n_paths)
    prev_delta = np.zeros(n_paths)
    sigma_arr = np.full(n_paths, 0.2)
    premium = bsm_call_vec(moneynesses, sigma_arr)
    for daily_idx in REBAL_INDICES:
        rv = np.clip(realized_vol_np(S[:daily_idx + 1, :]), 0.01, None)
        St = S[daily_idx, :]
        delta = bsm_delta_vec(St, rv, daily_idx * H_DAILY)
        trade = delta - prev_delta
        currency -= trade * St
        underlying += trade
        prev_delta = delta
    S_T = S[DAILY_STEPS, :]

    return premium + underlying * S_T + currency - np.maximum(S_T - K, 0)

# train the network and return it alongside the epoch losses
def train(S_train):

    net = HedgingNet(4, HP["hidden"], HP["depth"]).to(DEVICE)
    opt = optim.Adam([
        {"params": net.net.parameters()},
        {"params": net.premium_net.parameters(), "lr": HP["lr"] * PREMIUM_LR_SCALE},
    ], lr=HP["lr"])
    sched = optim.lr_scheduler.StepLR(opt, step_size=STEP_SIZE, gamma=GAMMA)
    S_tensor = torch.tensor(S_train.T, dtype=torch.float32)
    loader = DataLoader(TensorDataset(S_tensor), batch_size=HP["batch_size"], shuffle=True)
    print(f"\n{'epoch':>6}  {'loss':>12}  {'mean pnl':>10}  {'std pnl':>10}  {'prem@1':>9}")
    losses = []
    for epoch in range(1, EPOCHS + 1):
        net.train()
        batch_losses = []
        for (S_batch,) in loader:
            S_batch = S_batch.to(DEVICE)
            opt.zero_grad()
            loss = (run_paths(S_batch, net) ** 2).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), HP["clip_norm"])
            opt.step()
            batch_losses.append(loss.item())
        sched.step()
        losses.append(np.mean(batch_losses))
        if epoch == 1 or epoch % 10 == 0:
            net.eval()
            with torch.no_grad():
                pnl = run_paths(S_tensor[:2000].to(DEVICE), net)
                prem_at1 = net.premium_for(torch.tensor([1.0], device=DEVICE)).item()
            print(f"{epoch:>6}  {losses[-1]:>12.4f}  {pnl.mean().item():>10.4f}  "
                  f"{pnl.std().item():>10.4f}  {prem_at1:>9.4f}")
            
    return net, losses

# save the trained model and metadata to disk
def save_model(net, losses):

    path = os.path.join(MODELS_DIR, f"{TAG}.pt")
    with torch.no_grad():
        prem_at1 = net.premium_for(torch.tensor([1.0], device=DEVICE)).item()
    torch.save({
        "tag": TAG,
        "feature_set": "base_relvol_bsdelta",
        "rebalance": REBALANCE,
        "input_dim": 4,
        "N": N,
        "daily_steps": DAILY_STEPS,
        "h_daily": H_DAILY,
        "rebal_indices": REBAL_INDICES,
        "K": K, "r": r, "T": T,
        "moneyness_range": MONEYNESS_RANGE,
        "sigma_range": SIGMA_RANGE,
        "hyperparameters": HP,
        "epochs": EPOCHS,
        "premium_at_atm": prem_at1,
        "final_train_loss": float(losses[-1]),
        "state_dict": net.state_dict(),
    }, path)
    print(f"\nModel saved to {path}")

    return path

# load a saved checkpoint and return the network in eval mode
def load_model(path):

    ckpt = torch.load(path, map_location=DEVICE, weights_only=False)
    net = HedgingNet(ckpt["input_dim"],
                     ckpt["hyperparameters"]["hidden"],
                     ckpt["hyperparameters"]["depth"]).to(DEVICE)
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    print(f"Loaded {ckpt['tag']}  premium@ATM={ckpt['premium_at_atm']:.4f}")

    return net

# print evaluation table comparing NN, practitioner BSM and oracle BSM
def evaluate(S_test, sigmas, moneynesses, net):
    net.eval()
    with torch.no_grad():
        nn_pnl = run_paths(torch.tensor(S_test.T, dtype=torch.float32).to(DEVICE), net).cpu().numpy()
        prem_at1 = net.premium_for(torch.tensor([1.0], device=DEVICE)).item()
        prem_test = net.premium_for(torch.tensor(moneynesses, dtype=torch.float32, device=DEVICE)).cpu().numpy()
    oracle_pnl = oracle_bsm_hedge_pnl(S_test, sigmas, moneynesses)
    practitioner_pnl = practitioner_bsm_hedge_pnl(S_test, moneynesses)
    mean_bsm_price = bsm_call_vec(moneynesses, sigmas).mean()
    pcts = [1, 5, 25, 75, 95, 99]
    print("\n" + "="*70)
    print(f"  {'':18s}{'NN':>12}{'BSM (prac.)':>14}{'BSM (oracle)':>14}")
    print("  " + "-"*58)
    for label, fn in [("mean pnl", np.mean), ("std pnl", np.std)]:
        print(f"  {label:18s}{fn(nn_pnl):>12.4f}{fn(practitioner_pnl):>14.4f}{fn(oracle_pnl):>14.4f}")
    for p in pcts:
        print(f"  {'P'+str(p):18s}{np.percentile(nn_pnl, p):>12.4f}"
              f"{np.percentile(practitioner_pnl, p):>14.4f}{np.percentile(oracle_pnl, p):>14.4f}")
    print(f"  {'mean bsm price':18s}{mean_bsm_price:>12.4f}")
    print(f"  {'learned prem@1':18s}{prem_at1:>12.4f}")
    print(f"  {'mean learned prem':18s}{prem_test.mean():>12.4f}")
    print("  " + "-"*58)
    gap_prac = nn_pnl.std() - practitioner_pnl.std()
    gap_oracle = nn_pnl.std() - oracle_pnl.std()
    print(f"  std gap vs practitioner BSM: {gap_prac:+.4f}  ({'NN wins' if gap_prac < 0 else 'BSM wins'})")
    print(f"  std gap vs oracle BSM:       {gap_oracle:+.4f}")
    print("="*70)

# main programme run
if __name__ == "__main__":
    print("── Generating training paths ──")
    S_train, _, _ = generate_paths(N_TRAIN, seed=0)
    print("── Training ──")
    net, losses = train(S_train)
    model_path = save_model(net, losses)
    print("\n── Quick evaluation ──")
    S_eval, sigmas_eval, mon_eval = generate_paths(5_000, seed=42)
    evaluate(S_eval, sigmas_eval, mon_eval, net)
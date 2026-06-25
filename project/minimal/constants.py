"""
Constants used by the project to train and test the neural networks
"""


from project.helpers.helpers import get_torch_device

S0 = 100.0
K_LO = 80.0
K_HI = 120.0
SIGMA_LO = 0.10
SIGMA_HI = 0.30
T = 1.0

DEVICE = get_torch_device()

HYPERPARAMETER_MAP = {
    "base_monthly": {
        "h": 64,
        "d": 4,
        "eta": 3e-2,
        "B": 512,
        "s": 30,
        "gamma": 0.5,
        "c": 1.0
    },
    "base_daily": {
        "h": 128,
        "d": 4,
        "eta": 3e-2,
        "B": 512,
        "s": 30,
        "gamma": 0.5,
        "c": 0.5
    },
    "base_relvol_bsdelta_monthly": {
        "h": 128,
        "d": 4,
        "eta": 3e-2,
        "B": 1024,
        "s": 30,
        "gamma": 0.5,
        "c": 1.0
    },
    "base_relvol_bsdelta_daily": {
        "h": 128,
        "d": 4,
        "eta": 3e-2,
        "B": 512,
        "s": 30,
        "gamma": 0.5,
        "c": 0.5
    }
}
MODEL = "base_relvol_bsdelta_monthly"

HIDDEN_NEURONS = HYPERPARAMETER_MAP[MODEL]["h"]
HIDDEN_LAYERS = HYPERPARAMETER_MAP[MODEL]["d"]
LEARNING_RATE = HYPERPARAMETER_MAP[MODEL]["eta"]
BATCH_SIZE = HYPERPARAMETER_MAP[MODEL]["B"]
N_EPOCHS = HYPERPARAMETER_MAP[MODEL]["s"] + 20
ACTIVATION_PARAM = HYPERPARAMETER_MAP[MODEL]["gamma"]
GRAD_CLIP_THRESHOLD = HYPERPARAMETER_MAP[MODEL]["c"]

if MODEL == "base_monthly" or MODEL == "base_daily":
    N_FEATURES = 2
else:
    N_FEATURES = 4

if MODEL.endswith("monthly"):
    N = 12
else:
    N = 252
H = T / N


def get_model_params(model_name: str) -> dict:
    hp = HYPERPARAMETER_MAP[model_name]
    n_features = 2 if model_name in ("base_monthly", "base_daily") else 4
    n = 12 if model_name.endswith("monthly") else 252

    return {
        "HIDDEN_NEURONS": hp["h"],
        "HIDDEN_LAYERS": hp["d"],
        "LEARNING_RATE": hp["eta"],
        "BATCH_SIZE": hp["B"],
        "N_EPOCHS": hp["s"] + 20,
        "ACTIVATION_PARAM": hp["gamma"],
        "GRAD_CLIP_THRESHOLD": hp["c"],
        "N_FEATURES": n_features,
        "N": n,
        "H": T / n,
    }
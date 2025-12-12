#%%
import matplotlib.pyplot as plt
import os, pickle
import test_constants_carpol
import numpy as np
import test_constants_carpol

actor_nn = test_constants_carpol.ELE_ACTOR_VERSION_nn
actor_st = test_constants_carpol.ELE_ACTOR_VERSION_st
WINDOW = test_constants_carpol.WINDOW  


# beta_soft  = test_constants_carpol.beta_soft
# depth_soft = test_constants_carpol.depth_soft
init_params_path = os.path.join('./assets', f"{actor_st}", "actor_soft_init_params.npz")
with np.load(init_params_path) as npz:
    depth_soft = int(npz["depth"].item())
    beta_soft = float(npz["beta"].item())

actors = [actor_nn, actor_st]

for actor_version in actors:
    # 1) Load logs
    print(f"Loading logs for {actor_version}...")
    pkl_path = os.path.join('./assets', f"{actor_version}", "learning_logs.pkl")
    with open(pkl_path, "rb") as f:
        logs_loaded = pickle.load(f)


    #  Plot 1: mean episode return collected during training batches
    train_returns = logs_loaded.get("reward", [])
    x_train = list(range(len(train_returns)))
    plt.figure(figsize=(10, 6))
    plt.plot(x_train, train_returns, label="Train Avg Episode Return", alpha=0.8)
    plt.xlabel("Iteration"); plt.ylabel("Avg Episode Return (training)")
    plt.title(f"Learning Curve — Training Return for {actor_version}")
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
    plt.show()



    #  Plot 2: cumulative evaluation return (per rollout)
    eval_returns = logs_loaded.get("eval return (sum)")
    x_eval = list(range(len(eval_returns)))
    plt.figure(figsize=(10, 6))
    plt.plot(x_eval, eval_returns, label="Eval Episode Return", alpha=0.8)
    plt.xlabel("Iteration"); plt.ylabel("Episode Return (evaluation)")
    plt.title(f"Learning Curve — Eval Return for {actor_version}")
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
    plt.show()



    #  Plot 3: Learning rate schedule
    lr_vals = logs_loaded.get("lr", [])
    x_lr = list(range(len(lr_vals)))
    plt.figure(figsize=(10, 6))
    plt.plot(x_lr, lr_vals, label="LR")
    plt.xlabel("Iteration"); plt.ylabel("Learning rate")
    plt.title(f"Learning Rate Schedule for {actor_version}")
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
    plt.show()



#  Plot 4: Compare rolling average reward per episode (training) between NN and ST
# To remove initial bias, hide first window-1 points as NaN
def rolling_mean_strict(x, window):
    x = np.asarray(x, dtype=float)
    one_over_window = np.ones(window) / window
    roll_mean = np.convolve(x, one_over_window, mode='valid')    # length = n - window + 1 / valid cause Returns only elements where the arrays fully overlap.
    hide = np.full(window - 1, np.nan)                     # hide first window-1 points (before a full window exists)
    return np.concatenate([hide, roll_mean])

plt.figure(figsize=(10, 6))

for actor_version in actors:
    pkl_path = os.path.join('./assets', actor_version, "learning_logs.pkl")
    with open(pkl_path, "rb") as f:
        logs = pickle.load(f)


    series = logs.get("train return (mean)") or logs.get("reward")
    if not series:
        raise KeyError(f"Training return series missing or empty in {actor_version}")
    
    # series = np.asarray(logs["reward"], dtype=float)
    series = np.asarray(series, dtype=float)    
    ra = rolling_mean_strict(series, WINDOW)

    x = np.arange(len(ra))
    plt.plot(x, ra, label=f"{actor_version} (window={WINDOW})", alpha=0.9)

plt.xlabel("Episode (index)")
plt.ylabel("Rolling Avg Episode Return (training)")
plt.title(f"Rolling Average Training Return per Episode (NN vs ST, depth={depth_soft}, beta={beta_soft})")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()



#%%

actor_nn = test_constants_carpol.ELE_ACTOR_VERSION_nn_vs_st
actor_st_versions = test_constants_carpol.actor_st_versions


def rolling_mean_strict(x, window):
    x = np.asarray(x, dtype=float)
    one_over_window = np.ones(window) / window
    roll_mean = np.convolve(x, one_over_window, mode='valid')
    hide = np.full(window - 1, np.nan)
    return np.concatenate([hide, roll_mean])


def load_soft_params(actor_version):
    """
    Load depth, beta, batchnorm, reg_type, reg_lambda
    from assets/<actor_version>/actor_soft_init_params.npz
    """
    init_params_path = os.path.join('./assets', actor_version, "actor_soft_init_params.npz")
    with np.load(init_params_path) as npz:
        depth_soft     = int(npz["depth"].item())
        beta_soft      = float(npz["beta"].item())
    return depth_soft, beta_soft


# ============================================================
# Plot: One figure comparing all actors (NN + all STs)
# ============================================================
plt.figure(figsize=(10, 6))

# 1) NN baseline
pkl_path_nn = os.path.join('./assets', actor_nn, "learning_logs.pkl")
with open(pkl_path_nn, "rb") as f:
    logs_nn = pickle.load(f)

if "reward" not in logs_nn or len(logs_nn["reward"]) == 0:
    raise KeyError(f"'reward' series missing or empty in {actor_nn}")

series_nn = np.asarray(logs_nn["reward"], dtype=float)
ra_nn = rolling_mean_strict(series_nn, WINDOW)
x_nn = np.arange(len(ra_nn))
plt.plot(x_nn, ra_nn, label=f"{actor_nn} (NN, window={WINDOW})", alpha=0.9, linewidth=2)

# 2) All soft-tree actors
for actor_version in actor_st_versions:
    pkl_path = os.path.join('./assets', actor_version, "learning_logs.pkl")
    with open(pkl_path, "rb") as f:
        logs = pickle.load(f)

    if "reward" not in logs or len(logs["reward"]) == 0:
        raise KeyError(f"'reward' series missing or empty in {actor_version}")

    series = np.asarray(logs["reward"], dtype=float)
    ra = rolling_mean_strict(series, WINDOW)
    x = np.arange(len(ra))

    # Load parameters for legend
    depth_soft, beta_soft = load_soft_params(actor_version)

    label = (
        f"{actor_version} "
        f"(ST, d={depth_soft}, β={beta_soft})"
    )

    plt.plot(x, ra, label=label, alpha=0.9)

plt.xlabel("Episode (index)")
plt.ylabel("Rolling Avg Reward (training)")
plt.title("Rolling Average Training Reward per Episode (NN vs Soft Trees)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
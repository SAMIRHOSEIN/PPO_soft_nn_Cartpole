#%%
import matplotlib.pyplot as plt
import os, pickle
import test_constants_carpol
import numpy as np
import test_constants_carpol

actor_nn = test_constants_carpol.ELE_ACTOR_VERSION_nn
actor_st = test_constants_carpol.ELE_ACTOR_VERSION_st
WINDOW = test_constants_carpol.WINDOW  
beta_soft  = test_constants_carpol.beta_soft
depth_soft = test_constants_carpol.depth_soft


actors = {actor_nn, actor_st}


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
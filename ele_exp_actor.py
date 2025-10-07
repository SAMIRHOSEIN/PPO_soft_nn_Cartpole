# %%
import os
import numpy as np
import torch
from torch import nn

# from torchrl_bridge import create_element_env
from torchrl_bridge import create_cartpole_env

from torchrl_bridge import ElementActorNet, ElementActorSoftTree

from torchrl.modules import ProbabilisticActor
from torch.distributions import Categorical as CategoricalDist
from tensordict.nn import TensorDictModule

from tqdm import tqdm
from collections import defaultdict
from torchrl.envs.utils import ExplorationType, set_exploration_type
import pickle

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch

if __name__ == "__main__":
    import importlib
    import test_constants
    importlib.reload(test_constants)
    
    # load constants
    horizon = test_constants.ELE_ACTOR_HORIZON
    n_episodes = test_constants.ELE_ACTOR_N_EPISODES 
    max_cost = test_constants.ELE_ACTOR_MAX_COST
    reset_prob = test_constants.ELE_ACTOR_RESET_PROB
    dirichlet_alpha = test_constants.ELE_ACTOR_DIRICHLET_ALPHA
    random_state = test_constants.ELE_ACTOR_RANDOM_STATE
    explore_type = test_constants.ELE_ACTOR_EXPLORE_TYPE
    actor_model = test_constants.actor_model


    # soft tree parameters
    depth_soft = test_constants.depth_soft
    beta_soft = test_constants.beta_soft
    batchnorm_soft = test_constants.batchnorm_soft

    actor_version = test_constants.ELE_ACTOR_VERSION
    state_dict_path = os.path.join('./assets', f"{actor_version}", "actor_net_state_dict.pt")
    init_params_path = os.path.join('./assets', f"{actor_version}", "actor_net_init_params.npz")

    init_params = np.load(init_params_path)
    input_dim = init_params['input_dim'].item()
    output_dim = init_params['output_dim'].item()
    include_step_count = init_params['include_step_count'].item()
    if actor_model == 'nn':
        actor_cells = init_params['actor_cells'].item()
        actor_layers = init_params['actor_layers'].item()

    assert horizon == init_params['horizon'].item(), \
        f"Actor was not trained for horizon={horizon} (training horizon={init_params['horizon'].item()})"

    # recreate env
    # env = create_element_env(
    #     horizon,
    #     max_cost=max_cost,
    #     include_step_count=include_step_count,
    #     reset_prob=reset_prob,
    #     dirichlet_alpha=dirichlet_alpha,
    #     random_state=random_state
    # )
    
    horizon = test_constants.ELE_PPO_HORIZON
    env = create_cartpole_env(horizon)


    # restore actor
    if actor_model == 'nn':
        actor_net = ElementActorNet(input_dim, output_dim, actor_cells, actor_layers)

    elif actor_model == 'st':
        actor_net = ElementActorSoftTree(
            input_dim, output_dim,
            depth=depth_soft, beta=beta_soft, apply_batchNorm=batchnorm_soft
    )
        
    print(f"actor_model: {actor_model}")

    actor_net.load_state_dict(
        torch.load(state_dict_path, weights_only=True)
    )
    actor_net.eval()    # set the model to evaluation mode

    # create policy
    actor_module = TensorDictModule(
        actor_net, in_keys=["observation"], out_keys=["logits"]
    )
    actor = ProbabilisticActor(
        module=actor_module,
        spec=env.action_spec,
        distribution_class=CategoricalDist,
        in_keys=["logits"],  # Key in the input tensor containing the observation
        out_keys=["action"],  # Key where the sampled action will be written
        return_log_prob=True,
    )

    # gather experience
    logs = defaultdict(list)
    eval_str = ""

    with tqdm(total=horizon * n_episodes) as pbar:
        with set_exploration_type(explore_type), torch.no_grad():
            for _ in range(n_episodes):
                # execute a rollout with the trained policy
                eval_rollout = env.rollout(horizon, actor)

                logs["observation"].append(eval_rollout["observation"].cpu().numpy())
                logs["action"].append(eval_rollout["action"].cpu().numpy())
                logs["reward"].append(eval_rollout["next", "reward"].cpu().numpy())
                logs["ep reward"].append(eval_rollout["next", "reward"].sum().item())

                eval_str = (
                    f"ep reward: {logs['ep reward'][-1]: 4.4f} "
                    f"(init: {logs['ep reward'][0]: 4.4f}), "
                )
                pbar.update(eval_rollout.numel())
                pbar.set_description(eval_str)

                del eval_rollout

    print(f"Average reward: {np.mean(logs['ep reward'])}")
    print(f"Initial state: {logs['observation'][0][0]}")
    print(f"Final state: {logs['observation'][0][-1]}")


    
    # save experience
    experience_path = os.path.join('./assets', f"{actor_version}", "experience.pkl")
    with open(experience_path, "wb") as f:
        pickle.dump(logs, f)


    # load experience
    with open(experience_path, "rb") as f:
        logs = pickle.load(f)

    # Define the list of arrays
    all_actions = logs["action"]
    all_actions = np.concatenate(all_actions)

    # check distributions of actions
    # unique, counts = np.unique(all_actions, return_counts=True)
    # print(dict(zip(unique, counts)))

    id2name = {
        0: "Do nothing",
        1: "Maintenance",
        2: "Repair",
        3: "Rehabilitation",
        4: "Replacement"
    }
    unique, counts = np.unique(all_actions, return_counts=True)
    action_distribution = {id2name.get(a, a): c for a, c in zip(unique, counts)}
    print(action_distribution)



# I added the following lines to find out the model structure of David's assets or any other model that we may use later
# --- 1) NPZ: model init params ---
npz_path = f"assets/{actor_version}/actor_net_init_params.npz"
npz = np.load(npz_path, allow_pickle=True)  
print("keys in actor_net_init_params.npz:", list(npz.keys()))

include_step_count = bool(npz["include_step_count"].item())
input_dim = int(npz["input_dim"].item())
output_dim = int(npz["output_dim"].item())
horizon = int(npz["horizon"].item())
reset_prob = npz["reset_prob"]
dirichlet_alpha = npz["dirichlet_alpha"]
if actor_model == 'nn':
    actor_cells = int(npz["actor_cells"].item())
    actor_layers = int(npz["actor_layers"].item())
    print(f"input_dim= {input_dim}, output_dim= {output_dim}, actor_cells= {actor_cells}, actor_layers= {actor_layers}, horizon= {horizon}, reset_prob= {reset_prob}, dirichlet_alpha= {dirichlet_alpha}, include_step_count= {include_step_count}\n")

elif actor_model == 'st':
    depth_soft = int(npz["depth"].item())
    beta_soft = float(npz["beta"].item())
    print(f"input_dim= {input_dim}, output_dim= {output_dim}, horizon= {horizon}, reset_prob= {reset_prob}, dirichlet_alpha= {dirichlet_alpha}, depth= {depth_soft}, beta= {beta_soft}, batchnorm= {batchnorm_soft}\n")


# --- 2) PT: actor weights (state_dict) ---
# pt_path = f"assets/{actor_version}/actor_net_state_dict.pt"
# obj = torch.load(pt_path, map_location="cpu")   
# if isinstance(obj, dict):
#     print("The odd indices (1, 3, 5) are activations (ELU) and have no parameters, only linear layers have weight and bias.")
#     print("First 10 keys:", list(obj.keys())[:10])

# %%
#%%
# action distribution summary
all_actions = np.concatenate(logs["action"])
id2name = {0:"Do nothing",1:"Maintenance",2:"Repair",3:"Rehabilitation",4:"Replacement"}

# ---- Action sequence for the single evaluation episode (t = 0..T-1 for steps actually taken) ----
ep0_actions = logs["action"][0].astype(int).flatten()          # shape: (horizon,)
ep0_obs     = logs["observation"][0]                           # shape: (horizon, obs_len)

ep0_action_names = [id2name.get(int(a), str(int(a))) for a in ep0_actions]
print("\nEvaluation action sequence (time-ordered):")
print(", ".join(ep0_action_names))

# %%
# Generated by AI
# 1) action sequence for the single evaluation episode
ep0_actions = logs["action"][0].astype(int).flatten()
T = len(ep0_actions)

# 2) color palette (crisp, colorblind-friendly-ish)
action_colors = {
    0: "#9E9E9E",  # gray
    1: "#4E79A7",  # blue
    2: "#59A14F",  # green
    3: "#F28E2B",  # orange
    4: "#E15759",  # red
}
colors = [action_colors[int(a)] for a in ep0_actions]

# 3) plot a single horizontal bar split into T segments
fig, ax = plt.subplots(figsize=(max(10, T*0.35), 1.6))
lefts = np.arange(T)
ax.barh(
    y=0, width=np.ones(T), left=lefts, height=0.85,
    color=colors, edgecolor="white", linewidth=1.4
)

# 4) label each segment with the step number (1..T) centered, with auto-contrasting text
for t, c in enumerate(colors):
    r, g, b = mcolors.to_rgb(c)
    luminance = 0.2126*r + 0.7152*g + 0.0722*b
    txt_color = "white" if luminance < 0.5 else "black"
    ax.text(t + 0.5, 0, str(t+1), ha="center", va="center", fontsize=9, color=txt_color, fontweight="bold")

# 5) cosmetics: GA label on the left, no title, clean axes
ax.set_ylim(-0.8, 0.8)
ax.set_xlim(0, T)
ax.set_yticks([0])
ax.set_yticklabels([f"PPO({actor_model})"], fontsize=11)   # put "PPO" on the left
ax.set_xticks([])                         # numbers are inside each segment already
for spine in ["top", "right", "left", "bottom"]:
    ax.spines[spine].set_visible(False)

# optional legend (comment out if you don't want it)
handles = [Patch(facecolor=action_colors[k], label=id2name[k]) for k in sorted(action_colors)]
leg = ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.35), ncol=3, frameon=False)

plt.tight_layout()
plt.show()
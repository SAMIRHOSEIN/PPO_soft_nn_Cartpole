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
    import test_constants_carpol
    importlib.reload(test_constants_carpol)
    
    # load constants
    horizon = test_constants_carpol.ELE_ACTOR_HORIZON
    n_episodes = test_constants_carpol.ELE_ACTOR_N_EPISODES 
    explore_type = test_constants_carpol.ELE_ACTOR_EXPLORE_TYPE
    actor_model = test_constants_carpol.actor_model

    eval_seed = test_constants_carpol.ELE_TRAINING_ACTOR_RESET_SEED


    # soft tree parameters
    depth_soft = test_constants_carpol.depth_soft
    beta_soft = test_constants_carpol.beta_soft
    batchnorm_soft = test_constants_carpol.batchnorm_soft

    actor_version = test_constants_carpol.ELE_ACTOR_VERSION
    state_dict_path = os.path.join('./assets', f"{actor_version}", "actor_net_state_dict.pt")
    init_params_path = os.path.join('./assets', f"{actor_version}", "actor_net_init_params.npz")

    init_params = np.load(init_params_path)
    input_dim = init_params['input_dim'].item()
    output_dim = init_params['output_dim'].item()
    if actor_model == 'nn':
        actor_cells = init_params['actor_cells'].item()
        actor_layers = init_params['actor_layers'].item()

    assert horizon == init_params['horizon'].item(), \
        f"Actor was not trained for horizon={horizon} (training horizon={init_params['horizon'].item()})"

    





    # ------------------------------------------------------------------------------
    horizon = test_constants_carpol.ELE_PPO_HORIZON
    env = create_cartpole_env()


    # deterministic initial state for this one episode / Seed once before the collector (for reproducible first reset)
    random_seed_before_collector = test_constants_carpol.ELE_TRAINING_ACTOR_RANDOM_STATE_CARTPOLE
    env.reset(seed=random_seed_before_collector)
    # ------------------------------------------------------------------------------







    



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



# I added the following lines to find out the model structure of David's assets or any other model that we may use later
# --- 1) NPZ: model init params ---
npz_path = f"assets/{actor_version}/actor_net_init_params.npz"
npz = np.load(npz_path, allow_pickle=True)  
print("keys in actor_net_init_params.npz:", list(npz.keys()))

input_dim = int(npz["input_dim"].item())
output_dim = int(npz["output_dim"].item())
horizon = int(npz["horizon"].item())
if actor_model == 'nn':
    actor_cells = int(npz["actor_cells"].item())
    actor_layers = int(npz["actor_layers"].item())
    print(f"input_dim= {input_dim}, output_dim= {output_dim}, actor_cells= {actor_cells}, actor_layers= {actor_layers}, horizon= {horizon}\n")

elif actor_model == 'st':
    depth_soft = int(npz["depth"].item())
    beta_soft = float(npz["beta"].item())
    print(f"input_dim= {input_dim}, output_dim= {output_dim}, horizon= {horizon}, depth= {depth_soft}, beta= {beta_soft}, batchnorm= {batchnorm_soft}\n")

# %%
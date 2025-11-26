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



if __name__ == "__main__":
    import importlib
    import test_constants_carpol
    importlib.reload(test_constants_carpol)
    
    # load constants
    horizon = test_constants_carpol.ELE_PPO_HORIZON_eval
    n_episodes = test_constants_carpol.ELE_ACTOR_N_EPISODES 
    explore_type = test_constants_carpol.ELE_ACTOR_EXPLORE_TYPE
    actor_model = test_constants_carpol.actor_model


    print(f"n_episodes: {n_episodes}")

    # soft tree parameters
    depth_soft = test_constants_carpol.depth_soft
    beta_soft = test_constants_carpol.beta_soft
    batchnorm_soft = test_constants_carpol.batchnorm_soft

    actor_version = test_constants_carpol.ELE_ACTOR_VERSION
    if actor_model == 'nn':
        state_dict_path = os.path.join('./assets', f"{actor_version}", "actor_net_state_dict.pt")
        init_params_path = os.path.join('./assets', f"{actor_version}", "actor_net_init_params.npz")
        
    elif actor_model == 'st' or actor_model == 'ft':
        state_dict_path = os.path.join('./assets', f"{actor_version}", "actor_soft_state_dict.pt")
        init_params_path = os.path.join('./assets', f"{actor_version}", "actor_soft_init_params.npz")

    init_params = np.load(init_params_path)
    input_dim = init_params['input_dim'].item()
    output_dim = init_params['output_dim'].item()
    if actor_model == 'nn':
        actor_cells = init_params['actor_cells'].item()
        actor_layers = init_params['actor_layers'].item()

    assert horizon == init_params['horizon'].item(), \
        f"Actor was not trained for horizon={horizon} (training horizon={init_params['horizon'].item()})"

    

    # ------------------------------------------------------------------------------
    # recreate environment
    limit_for_cartpole_env_eval = test_constants_carpol.limit_for_cartpole_env_eval

    if limit_for_cartpole_env_eval:
        # env and evaluation limited to train_horizon
        env = create_cartpole_env(max_episode_steps=horizon)    
        eval_horizon = horizon

    elif not limit_for_cartpole_env_eval:
        # default CartPole (500 steps max)
        env = create_cartpole_env()
        eval_horizon = 500                  # evaluate over the full CartPole horizon        


    # # deterministic initial state for this one episode / Seed once before the collector (for reproducible first reset)
    # random_seed_before_collector = test_constants_carpol.ELE_TRAINING_ACTOR_RANDOM_STATE_CARTPOLE
    # env.reset(seed=random_seed_before_collector)
    # ------------------------------------------------------------------------------


    # restore actor
    # ============================================================
    # Path A: Restore NN actor for evaluation
    # ============================================================
    if actor_model == 'nn':
        print(f"actor_model: {actor_model}")

        # restore actor
        actor_net = ElementActorNet(input_dim, output_dim, actor_cells, actor_layers)

        actor_net.load_state_dict(
            torch.load(state_dict_path, weights_only=True)
        )
        actor_net.eval()    # set the model to evaluation mode

        # create policy
        actor_module = TensorDictModule(
            actor_net, in_keys=["observation"], out_keys=["logits"]
        )


        # create policy based on actor_module : NN
        actor = ProbabilisticActor(
            module=actor_module,
            spec=env.action_spec,
            distribution_class=CategoricalDist,
            in_keys=["logits"],  # Key in the input tensor containing the observation
            out_keys=["action"],  # Key where the sampled action will be written
            return_log_prob=True,
        )    

    # ============================================================
    # Path B: Restore SOFT actor for evaluation
    # ============================================================
    elif actor_model == 'st':
        print(f"actor_model: {actor_model}")

        # soft tree parameters
        depth_soft = test_constants_carpol.depth_soft
        beta_soft = test_constants_carpol.beta_soft
        batchnorm_soft = test_constants_carpol.batchnorm_soft

        print(f"depth_soft: {depth_soft}")
        print(f"beta_soft: {beta_soft}")


        # restore soft actor
        actor_soft = ElementActorSoftTree(
            input_dim, output_dim,
            depth=depth_soft, beta=beta_soft, apply_batchNorm=batchnorm_soft
            )

        actor_soft.load_state_dict(
            torch.load(state_dict_path, weights_only=True)
        )
        actor_soft.eval()    # set the model to evaluation mode

        # create policy
        actor_module = TensorDictModule(
            actor_soft, in_keys=["observation"], out_keys=["log_probs"] # soft tree outputs log_probs
        )

        actor = ProbabilisticActor(
            module=actor_module,
            spec=env.action_spec,
            distribution_class=CategoricalDist,
            in_keys={"logits": "log_probs"},  # cause torch.distributions.Categorical does not have a log_probs= argument - I need to match with out_keys of actor_module
            out_keys=["action"],  # Key where the sampled action will be written
            return_log_prob=True,
        )    


    # gather experience
    logs = defaultdict(list)
    eval_str = ""






    # -----------------------------------------------------------------------------------------------------------------
    # original code:
    # we don’t use terminated / truncated at all in originl code
    # we sum the entire reward tensor from eval_rollout["next", "reward"], which may include:
    # steps after the environment is done (if rollout continues),
    # or even parts of a new episode if the env auto-resets inside rollout.
    # So this file does not yet enforce “episode return = sum of rewards up to terminated | truncated” the way I did in ele_ppo_training.py.

    # with tqdm(total=horizon * n_episodes) as pbar:
    #     with set_exploration_type(explore_type), torch.no_grad():
    #         for _ in range(n_episodes):
    #             # execute a rollout with the trained policy
    #             eval_rollout = env.rollout(horizon, actor)

    #             logs["observation"].append(eval_rollout["observation"].cpu().numpy())
    #             logs["action"].append(eval_rollout["action"].cpu().numpy())
    #             logs["reward"].append(eval_rollout["next", "reward"].cpu().numpy())
    #             logs["ep reward"].append(eval_rollout["next", "reward"].sum().item())

    #             eval_str = (
    #                 f"ep reward: {logs['ep reward'][-1]: 4.4f} "
    #                 f"(init: {logs['ep reward'][0]: 4.4f}), "
    #             )
    #             pbar.update(eval_rollout.numel())
    #             pbar.set_description(eval_str)

    #             del eval_rollout


    # The my code below enforces “episode return = sum of rewards up to terminated | truncated”
    with tqdm(total=eval_horizon  * n_episodes) as pbar: # This line is just for progress bar not affect the evaluation
        with set_exploration_type(explore_type), torch.no_grad():
            for _ in range(n_episodes):
                # execute a rollout with the trained policy
                eval_rollout = env.rollout(eval_horizon, actor) # execute a rollout with trained policy - may continue after done, so we’ll mask

                # Move tensors to CPU and flatten the time dimension
                obs = eval_rollout["observation"].cpu()
                act = eval_rollout["action"].cpu()
                rew = eval_rollout["next", "reward"].squeeze(-1).cpu() # squeeze to remove last dim

                terminated = eval_rollout["next", "terminated"].squeeze(-1).cpu()
                truncated  = eval_rollout["next", "truncated"].squeeze(-1).cpu()
                done_mask  = (terminated | truncated)

                # Episode length = up to first done, or full horizon if no done
                if done_mask.any():
                    t_end = int(done_mask.nonzero(as_tuple=True)[0][0].item()) + 1 # +1 to include the done step
                else:
                    t_end = rew.shape[0]

                # Store only the actual episode segment
                logs["observation"].append(obs[:t_end].numpy())
                logs["action"].append(act[:t_end].numpy())
                logs["reward"].append(rew[:t_end].numpy())

                ep_reward = float(rew[:t_end].sum().item()) # sum of rewards up to terminated | truncated = done
                logs["ep reward"].append(ep_reward)

                eval_str = (
                    f"ep reward: {logs['ep reward'][-1]: 4.4f} "
                    f"(init: {logs['ep reward'][0]: 4.4f}), "
                )
                pbar.update(t_end) # moves the bar after each episode by the number of completed steps.
                pbar.set_description(eval_str)

                del eval_rollout

    # -----------------------------------------------------------------------------------------------------------------


    print(f"Average reward: {np.mean(logs['ep reward'])}")
    print(f"Initial state for first episode: {logs['observation'][0][0]}")
    print(f"Final state for first episode: {logs['observation'][0][-1]}")


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
if actor_model == 'nn':
    npz_path = f"assets/{actor_version}/actor_net_init_params.npz"
    npz = np.load(npz_path, allow_pickle=True)  
    print("keys in actor_net_init_params.npz:", list(npz.keys()))

elif actor_model == 'st' or actor_model == 'ft':
    npz_path = f"assets/{actor_version}/actor_soft_init_params.npz"
    npz =  np.load(npz_path, allow_pickle=True)  
    print("keys in actor_soft_init_params.npz:", list(npz.keys()))


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


# action distribution summary
all_actions = np.concatenate(logs["action"])
id2name = {0: "Push Left", 1: "Push Right"}  

# ---- Action sequence for the single evaluation episode (t = 0..T-1 for steps actually taken) ----
ep0_actions = logs["action"][0].astype(int).flatten()          # shape: (horizon,)
ep0_obs     = logs["observation"][0]                           # shape: (horizon, obs_len)

ep0_action_names = [id2name.get(int(a), str(int(a))) for a in ep0_actions]
print("\nEvaluation action sequence (time-ordered):")
print(", ".join(ep0_action_names))


# --- Condition-state distribution per step (episode 0) ---
ep0_obs = logs["observation"][0]                 # shape: (horizon, obs_dim)
obs_dim = ep0_obs.shape[1]

# If include_step_count=True, the last obs entry is normalized time; otherwise there is no time column.
ncs_eff = int(obs_dim)

cs_traj = ep0_obs[:, :ncs_eff]                                # (horizon, ncs)

print("\nCondition-state distribution per step (episode 0):")
for t, (cs, a) in enumerate(zip(cs_traj, ep0_actions)):
    cs_str = ", ".join([f"cs{k}={p:.3f}" for k, p in enumerate(cs)])
    print(f"Step={t:02d}  act={id2name[int(a)]:<14} [{cs_str}]")
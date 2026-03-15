# %%
import os
import numpy as np
import torch
from torch import nn
import pickle

# environment
from torchrl_bridge import create_cartpole_env

# actor and critic networks
from torchrl_bridge import ElementActorNet, ElementActorSoftTree, ValueNet
from torchrl.modules import ProbabilisticActor, ValueOperator
from torch.distributions import Categorical as CategoricalDist
from tensordict.nn import TensorDictModule

# collector and replay buffer
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage


# training
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from tqdm import tqdm
from collections import defaultdict
from torchrl.envs.utils import set_exploration_type


if __name__ == "__main__":
    import datetime
    version = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    import test_constants_carpol
    import importlib
    importlib.reload(test_constants_carpol)

    # torch seed for reproducibility
    torch_seed = test_constants_carpol.ELE_PPO_TORCH_SEED
    torch.manual_seed(torch_seed)
    device = torch.device("cpu")
    
    # environment
    # include_step_count = test_constants_carpol.ELE_PPO_INC_STEP
    # random_state = test_constants_carpol.ELE_PPO_RANDOM_STATE
    actor_model = test_constants_carpol.actor_model 
    

    # ------------------------------------------------------------------------------
    # eval_seed = test_constants_carpol.ELE_TRAINING_ACTOR_RESET_SEED
    horizon = test_constants_carpol.ELE_PPO_HORIZON

    limit_for_cartpole_env = test_constants_carpol.limit_for_cartpole_env

    if limit_for_cartpole_env:
        # create cartpole env function with horizon argument
        env = create_cartpole_env(max_episode_steps=horizon)    
        # - If we limit the CartPole env, evaluate for `horizon` steps.
        eval_horizon = horizon

    elif not limit_for_cartpole_env:
        # create cartpole env function without horizon argument
        env = create_cartpole_env()
        # - Otherwise, use the default full CartPole horizon (500 steps).
        eval_horizon = 500  # CartPole-v1 default max_episode_steps


    # # Seed once before the collector (for reproducible first reset)
    # random_seed_before_collector = test_constants_carpol.ELE_TRAINING_ACTOR_RANDOM_STATE_CARTPOLE
    # env.reset(seed=random_seed_before_collector)
    # ------------------------------------------------------------------------------


    # region: create actor and critic ========================================
    input_dim = test_constants_carpol.ELE_PPO_INPUT_DIM
    output_dim = test_constants_carpol.ELE_PPO_OUTPUT_DIM

    value_cells = test_constants_carpol.ELE_PPO_VALUE_CELLS
    value_layers = test_constants_carpol.ELE_PPO_VALUE_LAYERS


    if actor_model == 'nn':
        actor_cells = test_constants_carpol.ELE_PPO_ACTOR_CELLS
        actor_layers = test_constants_carpol.ELE_PPO_ACTOR_LAYERS

        # Neural Networks 
        actor_net = ElementActorNet(
            input_dim, output_dim, actor_cells, actor_layers,
            device=device
        )

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


    elif actor_model == 'st':
        # soft tree parameters
        depth_soft = test_constants_carpol.depth_soft
        beta_soft = test_constants_carpol.beta_soft
        batchnorm_soft = test_constants_carpol.batchnorm_soft

            
        print(f"depth_soft: {depth_soft}")
        print(f"beta_soft: {beta_soft}")

        # Soft Tree
        actor_soft = ElementActorSoftTree(
            input_dim, output_dim,
            depth=depth_soft, beta=beta_soft, apply_batchNorm=batchnorm_soft,
            device=device
        )

        actor_module = TensorDictModule(
            actor_soft, in_keys=["observation"], out_keys=["log_probs"] # to avoid confusion with logits from nn cause soft tree outputs log_probs
        )      
        
        actor = ProbabilisticActor(
            module=actor_module,
            spec=env.action_spec,
            distribution_class=CategoricalDist,
            in_keys={"logits": "log_probs"},  # dist kw 'logits' <- TD key 'log_probs'(cause torch.distributions.Categorical does not have a log_probs= argument) - Key in the input tensor containing the observation - I need to match with out_keys of actor_module
            out_keys=["action"],  # Key where the sampled action will be written
            return_log_prob=True,
        )    


    # # test for actor
    # observation = env.observation_spec.rand()['observation']
    # action_logits = actor_net(observation)
    # print(f"acion_logits: {action_logits.detach().cpu().numpy()}")
    # td_reset = env.reset()
    # actor(td_reset)
    # sampled_action = td_reset["action"].detach().cpu().numpy()
    # print(f"sampled action: {sampled_action}")
    
    # create critic
    value_net = ValueNet(
        input_dim, value_cells, value_layers,
        device=device
    )
    critic = ValueOperator(module=value_net, in_keys=["observation"])
    # # critic
    # td_reset = env.reset()
    # critic(td_reset)
    # sample_value = td_reset["state_value"].detach().numpy()
    # print(f"sample value: {sample_value}")
    # endregion ==============================================================

    # region: set up advantage and loss =======================================
    # GAE
    GAE_gamma = test_constants_carpol.ELE_PPO_GAE_GAMMA
    GAE_lmbda = test_constants_carpol.ELE_PPO_GAE_LAMBDA
    average_GAE = test_constants_carpol.ELE_PPO_AVERAGE_GAE

    advantage_module = GAE(
        gamma=GAE_gamma, lmbda=GAE_lmbda,
        value_network=critic, average_gae=average_GAE,
        device=device,
    )

    clip_epsilon = test_constants_carpol.ELE_PPO_CLIP_EPSILON
    entropy_eps = test_constants_carpol.ELE_PPO_ENTROPY_EPS
    critic_coef = test_constants_carpol.ELE_PPO_CRITIC_COEF

    # PPO loss
    loss_module = ClipPPOLoss(
        actor_network=actor,
        critic_network=critic,
        clip_epsilon=clip_epsilon,
        entropy_bonus=bool(entropy_eps),
        entropy_coeff=entropy_eps,
        # these keys match by default but we set this for completeness
        critic_coeff=critic_coef,
        loss_critic_type="smooth_l1",
    )
    # endregion ==============================================================
    # region: set up collecor and replay_buffer ===========================================================
    frames_per_batch = test_constants_carpol.ELE_PPO_FRAMES_PER_BATCH
    total_frames = test_constants_carpol.ELE_PPO_TOTAL_FRAMES

    collector = SyncDataCollector(
        create_env_fn=lambda: env,
        policy=actor,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        split_trajs=False,
        device=device
    )

    replay_buffer = ReplayBuffer(
        storage=LazyTensorStorage(max_size=frames_per_batch),
        sampler=SamplerWithoutReplacement(),
    )
    # endregion ===============================================================

    # region: training ===========================================================
    lr = test_constants_carpol.ELE_PPO_LR
    lr_min = test_constants_carpol.ELE_PPO_LR_MIN
    training_epochs = test_constants_carpol.ELE_PPO_TRAINING_EPOCHS
    sub_batch_size = test_constants_carpol.ELE_PPO_SUB_BATCH_SIZE
    max_grad_norm = test_constants_carpol.ELE_PPO_MAX_GRAD_NORM
    eval_freq = test_constants_carpol.ELE_PPO_EVAL_FREQ
    eval_explore_type = test_constants_carpol.ELE_PPO_EVAL_EXPLORE_TYPE

    optim = torch.optim.Adam(loss_module.parameters(), lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, total_frames // frames_per_batch, lr_min
    )

    # training loop
    logs = defaultdict(list)
    eval_str = ""

    with tqdm(total=total_frames) as pbar: # This line is just for the progress bar, but total_frames is used in collector for training above. 
        # We iterate over the collector until it reaches the total number of frames it was
        # designed to collect:
        for i, tensordict_data in enumerate(collector):
            # we now have a batch of data to work with. Let's learn something from it.
            for _ in range(training_epochs):
                # We'll need an "advantage" signal to make PPO work.
                # We re-compute it at each epoch as its value depends on the value
                # network which is updated in the inner loop.
                advantage_module(tensordict_data)
                data_view = tensordict_data.reshape(-1)
                replay_buffer.extend(data_view.cpu())

                for _ in range(frames_per_batch // sub_batch_size):
                    subdata = replay_buffer.sample(sub_batch_size)
                    loss_vals = loss_module(subdata.to(device))
                    loss_value = (
                        loss_vals["loss_objective"]
                        + loss_vals["loss_critic"]
                        + loss_vals["loss_entropy"]
                    )

                    # Optimization: backward, grad clipping and optimization step
                    loss_value.backward()

                    # this is not strictly mandatory but it's good practice to keep
                    # your gradient norm bounded
                    torch.nn.utils.clip_grad_norm_(loss_module.parameters(), max_grad_norm)
                    optim.step()
                    optim.zero_grad()

            # ---------------------------------------------------------------------------------
            # original code
            # logs["reward"].append(tensordict_data["next", "reward"].mean().item())

            # my code to compute episode return from rewards and done flags
            rewards = tensordict_data["next", "reward"].detach().cpu().view(-1)
            terminated = tensordict_data["next", "terminated"].detach().cpu().view(-1)
            truncated = tensordict_data["next", "truncated"].detach().cpu().view(-1)
            done_mask = (terminated | truncated).to(torch.bool) #For each step t, treat an episode as “done” if either flag fires(terminal or time-limit end).

            episode_returns = []
            running_return = 0.0
            for reward, done in zip(rewards.tolist(), done_mask.tolist()):
                running_return += reward # Accumulates reward into running_return
                if done:
                    episode_returns.append(running_return)
                    running_return = 0.0  # equivalent to reset env and start a new episode in return accounting

            if not episode_returns and running_return: # if episode_returns is empty and running_return > 0
                # If the batch ends mid-episode, keep the partial return.
                episode_returns.append(running_return)

            if episode_returns:
                train_return_mean = float(np.mean(episode_returns))
            else:
                train_return_mean = 0.0

            
            logs["train return (mean)"].append(train_return_mean)
            # Backwards-compatible keys expected by plotting utilities.
            logs["reward"].append(train_return_mean)        
            logs["lr"].append(optim.param_groups[0]["lr"])
            # ---------------------------------------------------------------------------------

            #--------------------------------------------------------------------------
            # I modified the following line to avoid evaluation if eval_freq is None, because I got warning 
            # when I conisider the big horizon value(e.g., CartPole falls at step 10)
            # what was the reason and how can I fix it:
                # Answ: # In evaluation part of code we have env.rollout(horizon, actor), and this always tries to run exactly horizon steps, 
                        # even if the environment has already terminated early. Cause I ask for horizon steps in env.rollout(horizon, actor),
                        # I recieved warning when horizon is big case env ends earlier than horizon. 
                        # So, in the metrics I only count rewards up to where terminated or truncated fired. 
                        # Any extra steps after done are ignored in the evaluation return.
            # if i % eval_freq == 0:
            if eval_freq and (i % eval_freq == 0):
            #--------------------------------------------------------------------------
                # We evaluate the policy periodically.
                # Evaluation is rather simple: execute the policy without exploration
                # (take the expected value of the action distribution) for a given
                # number of steps (e.g., ``env`` horizon).
                # The ``rollout`` method of the ``env`` can take a policy as argument:
                # it will then execute this policy at each step.
                with set_exploration_type(eval_explore_type), torch.no_grad():

                    # -------------------------------------------------------------------------
                    # force identical initial state for evaluation in each iteration
                    # env.reset(seed=eval_seed)
                    # -------------------------------------------------------------------------

                    # execute a rollout with the trained policy
                    eval_rollout = env.rollout(eval_horizon, actor)

                    # -----------------------------------------------------------------------------
                    # my code to compute episode return from rewards and done flags
                    eval_rewards = eval_rollout["next", "reward"].detach().cpu().view(-1)
                    eval_terminated = eval_rollout["next", "terminated"].detach().cpu().view(-1)
                    eval_truncated = eval_rollout["next", "truncated"].detach().cpu().view(-1)
                    eval_done_mask = (eval_terminated | eval_truncated).to(torch.bool)  # For each step t, treat an episode as “done” if either flag fires (terminal or time-limit end).

                    eval_episode_returns = []
                    running_eval_return = 0.0
                    for reward, done in zip(eval_rewards.tolist(), eval_done_mask.tolist()):
                        running_eval_return += reward
                        if done:
                            eval_episode_returns.append(running_eval_return)
                            running_eval_return = 0.0

                    if not eval_episode_returns and running_eval_return:
                        eval_episode_returns.append(running_eval_return)

                    eval_return_sum = float(np.sum(eval_episode_returns)) if eval_episode_returns else 0.0
                    logs["eval return (sum)"].append(eval_return_sum)


                    eval_str = (
                        # f"eval cumulative reward: {logs['eval reward (sum)'][-1]: .4e} "
                        # f"(init: {logs['eval reward (sum)'][0]: .4e})"
                        f"eval cumulative reward: {logs['eval return (sum)'][-1]: .4e} "
                        f"(init: {logs['eval return (sum)'][0]: .4e})"                        
                    )
                    del eval_rollout

            pbar.update(tensordict_data.numel())
            cum_reward_str = (
                # f"average reward={logs['reward'][-1]:.4e} (init={logs['reward'][0]: .4e})"
                f"avg episode return={logs['train return (mean)'][-1]:.4e} "
                f"(init={logs['train return (mean)'][0]: .4e})"                
            )
            lr_str = f"lr policy: {logs['lr'][-1]: .4e}"
            pbar.set_description(", ".join([eval_str, cum_reward_str, lr_str]))

            # We're also using a learning rate scheduler. Like the gradient clipping,
            # this is a nice-to-have but nothing necessary for PPO to work.
            scheduler.step()
    # endregion ===============================================================

    # region: save results ====================================================
    # Save/Load state_dict (Recommended)
    # https://pytorch.org/tutorials/beginner/saving_loading_models.html

    # create folder with name of version
    if actor_model == 'nn':
        save_path = os.path.join('./assets', f"{version}_nn")
        os.makedirs(save_path, exist_ok=True)

        # save only the actor
        base_actor = getattr(actor_net, 'module', actor_net) #getattr(obj, "attr", default): tries to read obj.attr; if it doesn’t exist, it returns default.
                                                            # cause we have actor_network_params.module.0.module.layers.0.weight but we need layers.0.weight
        torch.save(
            base_actor.state_dict(),
            os.path.join(save_path, "actor_net_state_dict.pt")
        )

        np.savez(
            os.path.join(save_path, "actor_net_init_params.npz"),
            input_dim=input_dim, output_dim=output_dim,
            actor_cells=actor_cells,
            actor_layers=actor_layers,
            horizon=horizon,
        )
    elif actor_model == 'st':
        save_path = os.path.join('./assets', f"{version}_st")
        os.makedirs(save_path, exist_ok=True)

        # save only the actor
        base_actor = getattr(actor_soft, 'module', actor_soft) #getattr(obj, "attr", default): tries to read obj.attr; if it doesn’t exist, it returns default.
                                                            # cause we have actor_network_params.module.0.module.layers.0.weight but we need layers.0.weight
        torch.save(
            base_actor.state_dict(),
            os.path.join(save_path, "actor_soft_state_dict.pt")
        )
        np.savez(
            os.path.join(save_path, "actor_soft_init_params.npz"),
            input_dim=input_dim, output_dim=output_dim,
            depth=depth_soft, beta=beta_soft, apply_batchNorm=batchnorm_soft,
            horizon=horizon,
        )


    with open(os.path.join(save_path, "learning_logs.pkl"), 'wb') as file:
        pickle.dump(logs, file)
    # endregion ===============================================================

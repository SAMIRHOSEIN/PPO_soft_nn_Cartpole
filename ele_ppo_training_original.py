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
    env = create_cartpole_env()


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

    elif actor_model == 'st':
        # soft tree parameters
        depth_soft = test_constants_carpol.depth_soft
        beta_soft = test_constants_carpol.beta_soft
        batchnorm_soft = test_constants_carpol.batchnorm_soft

        # Soft Tree
        actor_net = ElementActorSoftTree(
            input_dim, output_dim,
            depth=depth_soft, beta=beta_soft, apply_batchNorm=batchnorm_soft,
            device=device
        )


    print(f"actor_model: {actor_model}")

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

    with tqdm(total=total_frames) as pbar:
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

            logs["reward"].append(tensordict_data["next", "reward"].mean().item())
            logs["lr"].append(optim.param_groups[0]["lr"])


            #--------------------------------------------------------------------------
            # I modified the following line to avoid evaluation if eval_freq is None, because I got warning 
            # when I conisider the big horizon value
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
                    eval_rollout = env.rollout(horizon, actor)
                    logs["eval reward"].append(eval_rollout["next", "reward"].mean().item())
                    logs["eval reward (sum)"].append(
                        eval_rollout["next", "reward"].sum().item()
                    )
                    eval_str = (
                        f"eval cumulative reward: {logs['eval reward (sum)'][-1]: .4e} "
                        f"(init: {logs['eval reward (sum)'][0]: .4e})"
                    )
                    del eval_rollout

            pbar.update(tensordict_data.numel())
            cum_reward_str = (
                f"average reward={logs['reward'][-1]:.4e} (init={logs['reward'][0]: .4e})"
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
    elif actor_model == 'st':
        save_path = os.path.join('./assets', f"{version}_st")
    os.makedirs(save_path, exist_ok=True)

    # save only the actor
    base_actor = getattr(actor_net, 'module', actor_net) #getattr(obj, "attr", default): tries to read obj.attr; if it doesn’t exist, it returns default.
                                                         # cause we have actor_network_params.module.0.module.layers.0.weight but we need layers.0.weight
    torch.save(
        base_actor.state_dict(),
        os.path.join(save_path, "actor_net_state_dict.pt")
    )


    if actor_model == 'nn':
        np.savez(
            os.path.join(save_path, "actor_net_init_params.npz"),
            input_dim=input_dim, output_dim=output_dim,
            actor_cells=actor_cells,
            actor_layers=actor_layers,
            horizon=horizon,         )
    elif actor_model == 'st':
        np.savez(
            os.path.join(save_path, "actor_net_init_params.npz"),
            input_dim=input_dim, output_dim=output_dim,
            depth=depth_soft, beta=beta_soft, apply_batchNorm=batchnorm_soft,
            horizon=horizon,
        )

    with open(os.path.join(save_path, "learning_logs.pkl"), 'wb') as file:
        pickle.dump(logs, file)
    # endregion ===============================================================

# %%
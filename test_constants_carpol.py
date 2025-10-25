from turtle import st
import numpy as np

from torchrl.envs.utils import ExplorationType



# region: constants for ele_ppo_training.py for Cartpole==================================
ELE_PPO_INPUT_DIM_CARTPOLE = 4   # cart position, cart velocity, pole angle, pole angular velocity
ELE_PPO_OUTPUT_DIM_CARTPOLE = 2  # push left or push right


# horizon for env and training
limit_for_cartpole_env = False
if limit_for_cartpole_env:
    """ horizon consider the max_episode_steps of cartpole env 
    & for training horizon """
    ELE_PPO_HORIZON = 35 #500 # Max=500 steps per episode in CartPole-v1 env

elif not limit_for_cartpole_env:
    """ horizon consider just for training horizon
    & the max_episode_steps of cartpole env is default 500 """
    ELE_PPO_HORIZON = 30


actor_model = 'st'  # 'st', 'nn' soft tree or neural network


# soft tree parameters
depth_soft = 8 #8 #5
beta_soft = 1.0
batchnorm_soft = False


ELE_PPO_INPUT_DIM = ELE_PPO_INPUT_DIM_CARTPOLE
ELE_PPO_OUTPUT_DIM = ELE_PPO_OUTPUT_DIM_CARTPOLE

# network parameters
ELE_PPO_TORCH_SEED = 0


if actor_model == 'nn':
    ELE_PPO_ACTOR_CELLS = 32
    ELE_PPO_ACTOR_LAYERS = 2

ELE_PPO_VALUE_CELLS = 32
ELE_PPO_VALUE_LAYERS = 2

# ELE_PPO_OUTPUT_DIM = NA



# GAE parameters
# gamma has to be 1 to avoid double counting gamma in the env
# lmbda=0 is equivalent to using TD0
# lmbda=1 is equivalent to using TD1
# so lmbda should be between 0 and 1
ELE_PPO_GAE_GAMMA = 1.0
ELE_PPO_GAE_LAMBDA = 0.95
ELE_PPO_AVERAGE_GAE = True

# PPO loss parameters
ELE_PPO_ENTROPY_EPS = 0.01
ELE_PPO_CLIP_EPSILON = (1e-3)
ELE_PPO_CRITIC_COEF = 1.0

# collector parameters
ELE_PPO_EPISODES_PER_BATCH = 32     # how many episodes we collect per iteration 









ELE_PPO_NUM_ITERATIONS = 1500 #1024       # how many iteration: collect→optimize cycles we run in total









ELE_PPO_FRAMES_PER_BATCH = ELE_PPO_HORIZON*ELE_PPO_EPISODES_PER_BATCH
ELE_PPO_TOTAL_FRAMES = ELE_PPO_FRAMES_PER_BATCH*ELE_PPO_NUM_ITERATIONS
ELE_PPO_SPLIT_TRAJS = False

# training parameters
ELE_PPO_TRAINING_EPOCHS = 50
ELE_PPO_SUB_BATCH_SIZE = ELE_PPO_HORIZON*32 # actually we consider one one mini-batch
ELE_PPO_MAX_GRAD_NORM = 1.0
ELE_PPO_LR = 1e-3
ELE_PPO_LR_MIN = 1e-5    # lr reduced to lr_min with total_frames // frames_per_batch



# I faced this warn when I put ELE_PPO_EVAL_FREQ = 1: WARN: You are calling 'step()' even though this environment has already returned terminated = True. You should always call 'reset()' once you receive 'terminated = True' -- any further steps are undefined behavior.
# Why I faced this warnn(due to this line of code: eval_rollout = env.rollout(horizon, actor) in ele_ppo_training.py?
# You create one env: env = create_cartpole_env() and pass that same instance to:
    # the collector (SyncDataCollector(create_env_fn=lambda: env, ...)), and
    # the evaluator (env.rollout(horizon, actor)).
# During evaluation you call env.rollout(horizon, actor) with horizon=50. If CartPole terminates in, say, 9 steps, the rollout still tries to keep stepping to 50 without an env reset → Gym warns:
    # You are calling step() even though this environment has already returned terminated=True…
# The warning you saw comes from the evaluation loop stepping past done, not from training. if I don't like see
# the warning I can set ELE_PPO_EVAL_FREQ to None. again, it doesn't affect training, just the evaluation frequency.
# and evaluation in training is not important for us because we evaluate the each actor in ele_exp_actor.py and also plot the learning curve in Plt_LC_nn_st.py
ELE_PPO_EVAL_FREQ = 1 # None or 1 : if I put it 1, I got warning because of big horizon value

# In carpole doen't change this to stochastic becasue the we need to repreduce the intial state and compare the soft tree and nn
ELE_PPO_EVAL_EXPLORE_TYPE = ExplorationType.DETERMINISTIC # This must be deterministic to choose greedy action because the frozen tree chooses the action with max prob
# endregion ==============================================================


# region: constants for ele_exp_actor.py ====================================

# Same horizon for env and training
# ELE_ACTOR_VERSION = '20251019-075631_nn' # This is nn horizon = 5 for env and horizen = 5 for training, and for new cartpole without reset seed 
# ELE_ACTOR_VERSION = '20251019-081756_nn' # This is nn horizon = 10 for env and horizen = 10 for training, and for new cartpole without reset seed 
# ELE_ACTOR_VERSION = '20251019-084320_nn' # This is nn horizon = 35 for env and horizen = 35 for training, and for new cartpole without reset seed
# ELE_ACTOR_VERSION = '20251019-033737_nn' # This is nn horizon = 75 for env and horizen = 75 for training, and for new cartpole without reset seed 
# ELE_ACTOR_VERSION = '20251019-102104_st' # This is st horizon = 5 for env and horizen = 5 for training, and for new cartpole without reset seed
# ELE_ACTOR_VERSION = '20251019-104556_st' # This is st horizon = 75 for env and horizen = 75 for training, and for new cartpole without reset seed
# ELE_ACTOR_VERSION = '20251019-150711_st' # This is st horizon = 10 for env and horizen = 10 for training, and for new cartpole without reset seed
# ELE_ACTOR_VERSION = '20251019-155201_st' # This is st horizon = 35 for env and horizen = 35 for training, and for new cartpole without reset seed
# ELE_ACTOR_VERSION = '20251020-130620_nn' # This is nn for default horizon for env and horizen = 30 for training, and for new cartpole without reset seed
# ELE_ACTOR_VERSION = '20251020-141936_st' # This is st for defaulthorizon for env and horizen = 30 for training, and for new cartpole without reset seed
# ELE_ACTOR_VERSION = '20251024-083802_st' # This is st with d=9 for defaulthorizon for env and horizen = 30 for training, and for new cartpole without reset seed, depth = 9
# ELE_ACTOR_VERSION = '20251024-122254_nn' # ELE_PPO_NUM_ITERATIONS = 1500
# ELE_ACTOR_VERSION = '20251024-101530_st' # This is st with d=9 for defaulthorizon for env and horizen = 30 for training, and for new cartpole without reset seed and ELE_PPO_NUM_ITERATIONS = 1500
# ELE_ACTOR_VERSION = '20251024-122254_nn' # ELE_PPO_NUM_ITERATIONS = 1500
ELE_ACTOR_VERSION = '20251024-172647_st' # This is st with d=8 for defaulthorizon for env and horizen = 30 for training, and for new cartpole without reset seed and ELE_PPO_NUM_ITERATIONS = 1500


ELE_ACTOR_HORIZON = 35 #64 #5 #75
ELE_ACTOR_N_EPISODES = 200 


# By design the CartPole environment does not let you pass an explicit starting state. 
# When you call reset(), each of the four state variables (cart position, cart velocity, pole angle and pole angular velocity) 
# is sampled independently and uniformly in (−0.05,0.05), so to compare different models i set the same random seed to just have the same initial state 
# to compare models. 



# ELE_ACTOR_RANDOM_STATE_CARTPOLE = 42

# In carpole doen't change this to stochastic becasue the we need to repreduce the intial state and compare the soft tree and nn
ELE_ACTOR_EXPLORE_TYPE = ExplorationType.DETERMINISTIC # This must be deterministic to choose greedy action because the frozen tree chooses the action with max prob
# endregion ==============================================================



# seeds for reproducible resets in Cartpole env
ELE_TRAINING_ACTOR_RANDOM_STATE_CARTPOLE = 1234 # seed once before the collector starts. It makes the very first reset deterministic so runs are reproducible.
ELE_TRAINING_ACTOR_RESET_SEED = 4321  # used before eval_rollout during training and evaluation for both ele_ppo_training.py and ele_exp_actor.py


# region: which actor model compared(leaning curve) for Plt_LC_nn_st.py ==================================
# for horizon = 5
# ELE_ACTOR_VERSION_nn = '20251019-075631_nn' # This is nn horizon = 5 for env and horizen = 5 for training, and for new cartpole without reset seed
# ELE_ACTOR_VERSION_st = '20251019-102104_st' # This is st horizon = 5 for env and horizen = 5 for training, and for new cartpole without reset seed

# # for horizon = 10
# ELE_ACTOR_VERSION_nn = '20251019-081756_nn' # This is nn horizon = 10 for env and horizen = 10 for training, and for new cartpole without reset seed
# ELE_ACTOR_VERSION_st = '20251019-150711_st' # This is st

# # for horizon = 35
# ELE_ACTOR_VERSION_nn = '20251019-084320_nn' # This is nn horizon = 35 for env and horizen = 35 for training, and for new cartpole without reset seed
# ELE_ACTOR_VERSION_st = '20251019-155201_st' # This is st horizon = 35 for env and horizen = 35 for training, and for new cartpole without reset seed

# # for horizon = 75
# ELE_ACTOR_VERSION_nn = '20251019-033737_nn' # This is nn horizon = 75 for env and horizen = 75 for training, and for new cartpole without reset seed
# ELE_ACTOR_VERSION_st = '20251019-104556_st' # This is st horizon = 75 for env and horizen = 75 for training, and for new cartpole without reset seed


# for deafult horizon
# ELE_ACTOR_VERSION_nn = '20251020-130620_nn' # This is nn for default horizon for env and horizen = 30 for training, and for new cartpole without reset seed
# ELE_ACTOR_VERSION_st = '20251020-141936_st' # This is st for defaulthorizon for env and horizen = 30 for training, and for new cartpole without reset seed


# for deafult horizon but soft tree has depth = 9
# ELE_ACTOR_VERSION_nn = '20251020-130620_nn' # This is nn for default horizon for env and horizen = 30 for training, and for new cartpole without reset seed, depth = 9
# ELE_ACTOR_VERSION_st = '20251024-083802_st' # This is st with d=9 for defaulthorizon for env and horizen = 30 for training, and for new cartpole without reset seed, depth = 9


# for deafult horizon but soft tree has depth = 9 and ELE_PPO_NUM_ITERATIONS = 1500
# ELE_ACTOR_VERSION_nn = '20251024-122254_nn' # ELE_PPO_NUM_ITERATIONS = 1500
# ELE_ACTOR_VERSION_st = '20251024-101530_st' # This is st with d=9 for defaulthorizon for env and horizen = 30 for training, and for new cartpole without reset seed and ELE_PPO_NUM_ITERATIONS = 1500


# for deafult horizon but soft tree has depth = 8 and ELE_PPO_NUM_ITERATIONS = 1500
ELE_ACTOR_VERSION_nn = '20251024-122254_nn' # ELE_PPO_NUM_ITERATIONS = 1500
ELE_ACTOR_VERSION_st = '20251024-172647_st' # This is st with d=8 for defaulthorizon for env and horizen = 30 for training, and for new cartpole without reset seed and ELE_PPO_NUM_ITERATIONS = 1500




WINDOW = 100 #50 100  # for rolling average - integer
# endregion ==============================================================
from turtle import st
import numpy as np

from torchrl.envs.utils import ExplorationType



# region: constants for ele_ppo_training.py for Cartpole==================================
ELE_PPO_INPUT_DIM_CARTPOLE = 4   # cart position, cart velocity, pole angle, pole angular velocity
ELE_PPO_OUTPUT_DIM_CARTPOLE = 2  # push left or push right


# env parameters
ELE_PPO_HORIZON = 20 #75

ELE_PPO_INC_STEP = True
# ELE_PPO_MAX_COST = unit_costs.max()

# ELE_PPO_RESET_PROB = None
# ELE_PPO_DIRICHLET_ALPHA = 0.5*np.ones(NCS)
# ELE_PPO_RANDOM_STATE = 42
ELE_PPO_RESET_PROB = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
ELE_PPO_DIRICHLET_ALPHA = None
ELE_PPO_RANDOM_STATE = 'off'


actor_model = 'st'  # 'st', 'nn' soft tree or neural network

NCS = ELE_PPO_INPUT_DIM_CARTPOLE
NA = ELE_PPO_OUTPUT_DIM_CARTPOLE

# network parameters
ELE_PPO_TORCH_SEED = 0
if ELE_PPO_INC_STEP:
    ELE_PPO_INPUT_DIM = NCS + 1
else:
    ELE_PPO_INPUT_DIM = NCS

if actor_model == 'nn':
    ELE_PPO_ACTOR_CELLS = 32
    ELE_PPO_ACTOR_LAYERS = 2

ELE_PPO_VALUE_CELLS = 32
ELE_PPO_VALUE_LAYERS = 2

ELE_PPO_OUTPUT_DIM = NA


# soft tree parameters
depth_soft = 8 #5
beta_soft = 1.0
batchnorm_soft = False


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
ELE_PPO_NUM_ITERATIONS = 1024       # how many iteration: collect→optimize cycles we run in total
ELE_PPO_FRAMES_PER_BATCH = ELE_PPO_HORIZON*ELE_PPO_EPISODES_PER_BATCH
ELE_PPO_TOTAL_FRAMES = ELE_PPO_FRAMES_PER_BATCH*ELE_PPO_NUM_ITERATIONS
ELE_PPO_SPLIT_TRAJS = False

# training parameters
ELE_PPO_TRAINING_EPOCHS = 50
ELE_PPO_SUB_BATCH_SIZE = ELE_PPO_HORIZON*32 # actually we consider one one mini-batch
ELE_PPO_MAX_GRAD_NORM = 1.0
ELE_PPO_LR = 1e-3
ELE_PPO_LR_MIN = 1e-5    # lr reduced to lr_min with total_frames // frames_per_batch
ELE_PPO_EVAL_FREQ = 1
ELE_PPO_EVAL_EXPLORE_TYPE = ExplorationType.DETERMINISTIC # This must be deterministic to choose greedy action because the frozen tree chooses the action with max prob
# endregion ==============================================================


# region: constants for ele_exp_actor.py ====================================
# ELE_ACTOR_VERSION = '20251007-161613_nn' # This is nn with horizon=5
# ELE_ACTOR_VERSION = '20251007-170244_st' # This is soft tree with horizon=5
# ELE_ACTOR_VERSION = '20251007-172544_nn' #This is nn with horizon=20
ELE_ACTOR_VERSION = '20251007-180714_st' # This is soft tree with horizon=20

ELE_ACTOR_HORIZON = 20 #75
ELE_ACTOR_N_EPISODES = 1 # modified to avoid confusion


# By design the CartPole environment does not let you pass an explicit starting state. 
# When you call reset(), each of the four state variables (cart position, cart velocity, pole angle and pole angular velocity) 
# is sampled independently and uniformly in (−0.05,0.05), so to compare different models i set the same random seed to just have the same initial state 
# to compare models. 



ELE_ACTOR_RANDOM_STATE_CARTPOLE = 42

ELE_ACTOR_EXPLORE_TYPE = ExplorationType.DETERMINISTIC # This must be deterministic to choose greedy action because the frozen tree chooses the action with max prob
# endregion ==============================================================
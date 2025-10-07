from turtle import st
import numpy as np


from torchrl.envs.utils import ExplorationType



# region: carPole env ==================================
ELE_PPO_HORIZON = 5

ELE_PPO_INPUT_DIM_CARTPOLE = 4   # cart position, cart velocity, pole angle, pole angular velocity
ELE_PPO_OUTPUT_DIM_CARTPOLE = 2  # push left or push right

# endregion ==============================================================


# region: constants for ele_ppo_training.py ==================================
# env parameters
ELE_PPO_HORIZON = 5 #75

ELE_PPO_INC_STEP = True
# ELE_PPO_MAX_COST = unit_costs.max()

# ELE_PPO_RESET_PROB = None
# ELE_PPO_DIRICHLET_ALPHA = 0.5*np.ones(NCS)
# ELE_PPO_RANDOM_STATE = 42
ELE_PPO_RESET_PROB = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
ELE_PPO_DIRICHLET_ALPHA = None
ELE_PPO_RANDOM_STATE = 'off'


actor_model = 'nn'  # 'st', 'nn' soft tree or neural network

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
# ELE_ACTOR_VERSION = '20250505-192030'   #David's model
# ELE_ACTOR_VERSION = '20250910-202015' # my model with 5 horizon
# ELE_ACTOR_VERSION = '20250917-102249' # my model with 75 horizon
# ELE_ACTOR_VERSION = '20250924-173308' # my model with 1 horizon with dirichlet alpha 0.5
# ELE_ACTOR_VERSION = '20250924-183258' # my model with 1 horizon with reset prob [1,0,0,0,0]
# ELE_ACTOR_VERSION = '20250924-184355' # my model with 5 horizon with reset prob [1,0,0,0,0]
# ELE_ACTOR_VERSION = '20250924-190413' # my model with 10 horizon with reset prob [1,0,0,0,0]
# ELE_ACTOR_VERSION = '20250925-100427'   # my model with 1 horizon with reset prob [1,0,0,0,0]
# ELE_ACTOR_VERSION = '20250925-101620'   # my model with 5 horizon with reset prob [1,0,0,0,0]
# ELE_ACTOR_VERSION = '20250930-152609'   # my model with 5 horizon with reset prob [1,0,0,0,0] - soft tree with depth 5 and beta 1.0
# ELE_ACTOR_VERSION = '20250930-163141_nn'   # my model with 5 horizon with reset prob [1,0,0,0,0] - neural network with 2 layers and 32 cells
# ELE_ACTOR_VERSION = '20250930-170010_st'   # my model with 5 horizon with reset prob [1,0,0,0,0] - soft tree with depth 5 and beta 1.0
# ELE_ACTOR_VERSION = '20251001-051210_nn'  # my model with 1 horizon with reset prob [1,0,0,0,0] - neural network with 2 layers and 32 cells
# ELE_ACTOR_VERSION = '20251001-052254_st'  # my model with 1 horizon with reset prob [1,0,0,0,0] - soft tree with depth 5 and beta 1.0
# ELE_ACTOR_VERSION = '20251001-062631_st'  # my model with 1 horizon with reset prob [1,0,0,0,0] - soft tree with depth 5 and beta 1.0
# ELE_ACTOR_VERSION = '20251001-093701_nn'  # my model with 1 horizon with reset prob [1,0,0,0,0] - neural network with 2 layers and 32 cells
# ELE_ACTOR_VERSION = '20251001-095252_st'  # my model with 1 horizon with reset prob [1,0,0,0,0] - soft tree with depth 5 and beta 1.0
# ELE_ACTOR_VERSION = '20251001-103449_st'  # my model with 1 horizon with reset prob [1,0,0,0,0] - soft tree with depth 6 and beta 1.0
# ELE_ACTOR_VERSION = '20251001-110432_st'  # my model with 1 horizon with reset prob [1,0,0,0,0] - soft tree with depth 8 and beta 1.0
# ELE_ACTOR_VERSION = '20251001-112507_st'  # my model with 1 horizon with reset prob [1,0,0,0,0] - soft tree with depth 10 and beta 1.0


# important files are:
# ELE_ACTOR_VERSION = '20251001-134624_nn' # my model with 1 horizon with reset prob [1,0,0,0,0] - neural network with 2 layers and 32 cells
# ELE_ACTOR_VERSION = '20251001-135623_st' # my model with 1 horizon with reset prob [1,0,0,0,0] - soft tree with depth 8 and beta 1.0
# ELE_ACTOR_VERSION = '20251001-141105_nn' # my model with 5 horizon with reset prob [1,0,0,0,0] - neural network with 2 layers and 32 cells
# ELE_ACTOR_VERSION = '20251001-142834_st' # my model with 5 horizon with reset prob [1,0,0,0,0] - soft tree with depth 8 and beta 1.0
ELE_ACTOR_VERSION = '20251001-150733_nn' # my model with 10 horizon with reset prob [1,0,0,0,0] - neural network with 2 layers and 32 cells
# ELE_ACTOR_VERSION = '20251001-153504_st' # my model with 10 horizon with reset prob [1,0,0,0,0] - soft tree with depth 8 and beta 1.0
# ELE_ACTOR_VERSION = '20251003-163151_nn' # my model with 5 horizon with reset prob [1,0,0,0,0] - neural network with 2 layers and 32 cells

ELE_ACTOR_HORIZON = 5 #75
# ELE_ACTOR_N_HORIZON = 1E
ELE_ACTOR_N_EPISODES = 1 # modified to avoid confusion
ELE_ACTOR_MAX_COST = 1.0

# ELE_DP_RESET_PROB = None
# ELE_DP_DIRICHLET_ALPHA = 0.5*np.ones(NCS)
# ELE_DP_RANDOM_STATE = 42
ELE_ACTOR_RESET_PROB = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
# ELE_ACTOR_RESET_PROB = np.array([0.3, 0.7, 0.0, 0.0, 0.0])
# ELE_ACTOR_RESET_PROB = np.array([0.0, 0.8, 0.2, 0.0, 0.0])

ELE_ACTOR_DIRICHLET_ALPHA = None
ELE_ACTOR_RANDOM_STATE = 'off'

ELE_ACTOR_EXPLORE_TYPE = ExplorationType.DETERMINISTIC # This must be deterministic to choose greedy action because the frozen tree chooses the action with max prob
# endregion ==============================================================
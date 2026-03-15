# %%
import numpy as np
import torch
from torch import device, nn

# environment
from torchrl.envs import GymWrapper

import test_constants_carpol

import gymnasium as gym
from torchrl.envs import GymWrapper


# constants for ConstantActionModule
_CONST_ACTION_DEFAULT = 0


# Note: 2 important points about carpole env:
    # 1. Gymnasium’s documentation explains that the randomness of the initial state is controlled by the seed argument of reset()
    # and recommends calling reset(seed=seed) once right after initialisation.

    # 2. I didnt consider the horizon in the cartpole env() it has a max episode length of 500 by default), 
    # cause in training(ele_ppo_training.py) we have eval_rollout = env.rollout(horizon, actor) that limits the rollout length to horizon anyway
    # so I removed the horizon argument from create_cartpole_env()

# 1- create cartpole env function with horizon argument
def create_cartpole_env(max_episode_steps: int = 500):
    base_env = gym.make("CartPole-v1", max_episode_steps=max_episode_steps)
    env = GymWrapper(base_env, categorical_action_encoding=True)
    print(f"max_episode_steps: {base_env.spec.max_episode_steps}")
    return env

# # 2- create cartpole env function without horizon argument
# def create_cartpole_env():
#     base_env = gym.make("CartPole-v1")
#     env = GymWrapper(base_env, categorical_action_encoding=True)
#     print(f"max_episode_steps: {base_env.spec.max_episode_steps}")
#     return env



class ValueNet(nn.Module):
    def __init__(
        self, input_dim,
        value_cells, value_layers,
        device=torch.device("cpu")
    ):
        # no need for input_dim due to LazyLinear
        super().__init__()
        layers = [nn.Linear(input_dim, value_cells, device=device), nn.ELU()]
        layers = layers + [nn.Linear(value_cells, value_cells, device=device), nn.ELU()] * value_layers
        layers.append(nn.Linear(value_cells, 1, device=device))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class ConstantModule(nn.Module):
    def __init__(self, constant_value: int = _CONST_ACTION_DEFAULT):
        super().__init__()
        self.constant = torch.as_tensor(constant_value)

    def forward(self, x):
        # ignore input x and always return the constant
        return self.constant


class ElementActorNet(nn.Module):
    def __init__(
        self, input_dim, output_dim,
        actor_cells, actor_layers,
        device=torch.device("cpu")
    ):
        # Change LazyLinear to Linear
        super().__init__()
        layers = [nn.Linear(input_dim, actor_cells, device=device), nn.ELU()]
        layers = layers + [nn.Linear(actor_cells, actor_cells, device=device), nn.ELU()] * actor_layers
        layers.append(nn.Linear(actor_cells, output_dim, device=device))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


# new version of the soft tree actor
# ============================================================
# ElementActorSoftTree: soft decision tree actor
# ============================================================
class ElementActorSoftTree(nn.Module):
    """
    Soft Decision Tree actor that outputs per-action log-probabilities.
    - Routing: sigmoid(β * logits)
    - Leaves: per-leaf class scores -> log_softmax -> logsumexp mix
    """
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 depth: int = 5,
                 beta: float = 1.0,
                 apply_batchNorm: bool = False,
                 device: torch.device | str | None = None):
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.depth = int(depth)
        self.beta = float(beta)
        self.apply_batchNorm = bool(apply_batchNorm)
        
        self._validate_parameters()

        self.internal_node_num_ = 2**self.depth - 1
        self.leaf_node_num_ = 2**self.depth

        self.inner_nodes = nn.Linear(self.input_dim + 1, self.internal_node_num_, bias=False)

        if self.apply_batchNorm: #a separate batch‐norm (with its own stats and affine parameters) after each internal-node logit
            self.inner_bn = nn.BatchNorm1d(self.internal_node_num_, affine=True, track_running_stats=True)
        else:
            self.inner_bn = None


        # This is the class defined to calculate the log of probability for each class
        class LeafLogMixtureHead(nn.Module):
            """
            Convert path probabilities mu (N, L) -> per-action log-probs (N, C).
            Learnable: leaf_scores (L, C).  logQ = log_softmax(leaf_scores, dim=1).
            y_log_pro[n, k] = logsumexp_l( log_mu[n,l] + logQ[l,k] )
            """

            def __init__(self, n_leaves, n_classes, eps=1e-12):
                super().__init__()
                self.eps = eps
                self.leaf_scores = nn.Parameter(torch.zeros(n_leaves, n_classes))

            @property #apply just to logQ
            def logQ(self) -> torch.Tensor:
                """(L, C) per-leaf log-probabilities."""
                return torch.log_softmax(self.leaf_scores, dim=1)

            def forward(self, mu: torch.Tensor) -> torch.Tensor:
                """
                mu: (N, L) path probabilities (sum to 1 across leaves per sample)
                returns: (N, C) log-probabilities per class 
                """
                # log(path_probability) - Clamping to a tiny eps: avoids -inf and NaNs
                log_mu = (mu.clamp_min(self.eps)).log()  # (N, L)
                logQ = self.logQ  # (L, C)
            
                # unsqueeze parts  prepares logQ for broadcasting with log_mu during the addition
                # To make PPO like supervised leaning part, I return the log of probabilities and then later on
                # in ProbabilisticActor, I set in_keys=[{"logits": log_probs}], so cause I consider the logit as key,
                # ProbabilisticActor will do softmax again on log-probabilites, so the training is right
                # So I returned the log-probabilites do to the following reasons: 
                # - I want to follow the same structure as supervised soft tree
                # - I want to have log scale to make the calculations numerically stable
                # - I want to use the same key "logits" in ProbabilisticActor, otherwise I could just calculate the sum of exp(log_mu + logQ) here and return the probabilities directly
                    # in that case I need to change the key in ProbabilisticActor to "probs" instead of "logits"
                y_log_pro = torch.logsumexp(log_mu.unsqueeze(-1) + logQ.unsqueeze(0), dim=1)  # (N, C)
                return y_log_pro
            

        self.head = LeafLogMixtureHead(self.leaf_node_num_, self.output_dim)

        if device is not None:
            self.to(device)            


    def set_beta(self, new_beta):
        self.beta = float(new_beta)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns per-action SCORES.
        - If x is (D,), return (C,)        (batchless step-time)
        - If x is (N, D), return (N, C)    (train-time)
        """
        single = (x.ndim == 1)          # remember original shape

        mu = self._forward(x)           # (1, L) if single else (N, L)
        log_prob = self.head(mu)          # (1, C) if single else (N, C) -  log-probabilities per class(C)

        if single: #removes the first dimension (only one sample in batch)
            log_prob = log_prob.squeeze(0)  # -> (C,) so policy outputs scalar action and this is compatible with CategoricalDist in ProbabilisticActor function

        return log_prob.contiguous() # get a tensor with standard memory layout - the logits is log-probabilities per class(C) not raw logits


    def _forward(self, X: torch.Tensor):
        """
        Compute path probabilities to leaves.
        Returns: mu (N, L)
        """
        X = self._data_augment(X)        # (1, D+1) if input was (D,) else X: (N, D+1)
        N = X.shape[0]                   # correct batch size

        logits = self.inner_nodes(X) # logits: (N, internal_node_num_)
        logits = self.beta * logits

        if self.apply_batchNorm:
            logits = self.inner_bn(logits) # batch normalization before sigmoid(ref: Hands on Machine learning book)

        path_prob = torch.sigmoid(logits).unsqueeze(-1)   # (N, L, 1)
        path_prob = torch.cat((path_prob, 1 - path_prob), dim=-1)  # (N, L, 2)

        _mu = X.new_ones((N, 1, 1)) # this is new one(root node)

        begin_idx = 0
        end_idx = 1
        for layer_idx in range(self.depth):
            _path_prob = path_prob[:, begin_idx:end_idx, :]

            _mu = _mu.view(N, -1, 1).repeat(1, 1, 2) # (duplicates each path prob to make slots for the two children)
            _mu = _mu * _path_prob                               #update path probabilities

            begin_idx = end_idx
            end_idx = begin_idx + 2 ** (layer_idx + 1)

        mu = _mu.view(N, self.leaf_node_num_) # flatten to 2D tensor (N, L)
        return mu

    # The following function is different from the supervised soft tree. In PPO cause when the collector calls the policy 
    # at each env step with a single environment(TorchRL’s collector calls the policy with one observation at a time), it passes the observation exactly as the env emits it: shape (D,) (a single sample, no batch dim). 
    # Later, during training, minibatches are (N, D). So PPO hits both cases. That is why I need to define if X.ndim == 1:. 
    def _data_augment(self, X: torch.Tensor) -> torch.Tensor:
        """
        step-time:   X: (D,) ──unsqueeze──> (1, D) ──concat bias──> (1, D+1)
        train-time:  X: (N, D) ───────────> (N, D) ──concat bias──> (N, D+1)
        """
        X = X.float()
        if X.ndim == 1:                 # (D,) -> (1, D)
            X = X.unsqueeze(0)
        N = X.shape[0]
        X = X.view(N, -1)               # (N, D)
        bias = X.new_ones((N, 1))       # (N, 1)
        X = torch.cat((bias, X), dim=1) # (N, D+1)
        return X


    def _validate_parameters(self):
        if not self.depth >= 1:
            raise ValueError("the tree depth should be at least 1, but got {} instead.")
        if not self.beta > 0:
            raise ValueError("The temperature, and beta should be positive, but got {} instead.")
        
# PPO Soft Tree vs Neural Network on CartPole

A PyTorch and TorchRL project for comparing **Proximal Policy Optimization (PPO)** with two policy representations on the **CartPole-v1** benchmark:

- a standard **Neural Network (NN) actor**
- an interpretable **Soft Decision Tree (ST) actor**

The repository focuses on training, evaluating, and comparing these actor types under a shared PPO framework, with saved experiment assets and plotting utilities for learning-curve analysis.

---

## Overview

This project investigates whether an interpretable **Soft Decision Tree** policy can serve as a competitive alternative to a conventional **Neural Network** policy in reinforcement learning.

The implementation uses:

- **PPO** for policy optimization
- **TorchRL** for environment handling, rollout collection, and PPO objectives
- **PyTorch** for actor and critic network definitions
- **Gymnasium / CartPole-v1** as the benchmark environment
- **Matplotlib** for visualization of training and evaluation behavior

The repository includes:

- a PPO training pipeline
- actor evaluation scripts
- reusable actor and critic model definitions
- experiment configuration through a centralized constants file
- saved model checkpoints and learning logs for multiple runs

---

## Main Features

- Train PPO with either:
  - **Neural Network actor**
  - **Soft Decision Tree actor**
- Evaluate saved actors using deterministic action selection
- Compare training returns and evaluation performance
- Save model weights, initialization parameters, and rollout logs
- Plot learning curves for NN and Soft Tree runs
- Study the effect of **tree depth** and **beta / temperature-style sharpness** in soft routing

---

## Repository Structure

```text
PPO_soft_nn_Cartpole/
├── ele_ppo_training.py          # Main PPO training script
├── ele_exp_actor.py             # Actor evaluation script for saved models
├── plt_nn_st.py                 # Plotting and comparison utilities
├── torchrl_bridge.py            # Environment wrapper + actor/critic model definitions
├── test_constants_carpol.py     # Centralized experiment settings and run selection
├── assets/                      # Saved trained models, logs, and experiment outputs
└── __pycache__/                 # Python cache files
```

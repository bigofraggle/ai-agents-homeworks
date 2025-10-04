# MountainCar Q-Learning

A simple Q-learning implementation for the MountainCar climbing task using Gymnasium.

## Overview

This project trains an agent to drive an underpowered car up a steep mountain by building momentum. The classic MountainCar environment has continuous state space, which is discretized into bins to make it compatible with tabular Q-learning.

**Goal**: Reach the flag at the top of the mountain by rocking back and forth to build momentum.

## Requirements

```bash
pip install -e .
```

This installs Gymnasium, NumPy, Matplotlib, and registers the custom `DiscretizedMountainCar-v0` environment.

## Training

Run the training script to learn the optimal policy:

```bash
python training.py
```

Training runs for 1000 episodes and saves the learned Q-table to `q_table_mountaincar.npy`. Two plots are generated:
- `training_progress.png`: Episode rewards over time
- `test_results.png`: Test performance after training

## Inference

After training, test the learned agent:

```bash
python main.py
```

This runs 5 test episodes with visualization enabled and displays the agent's performance.

## How It Works

- **State Space**: 2 continuous variables (car position, car velocity) discretized into 20 bins each
  - Position: 20 bins from -1.2 to 0.6
  - Velocity: 20 bins from -0.07 to 0.07
- **Action Space**: 3 discrete actions (push left, do nothing, push right)
- **Q-Table**: Shape (20, 20, 3) storing state-action values
- **Algorithm**: Standard Q-learning with epsilon-greedy exploration
- **Hyperparameters**:
  - Learning rate (α): 0.1
  - Discount factor (γ): 0.99
  - Epsilon decay: 1.0 → 0.01 over training
  - Episodes: 6000
- **Challenge**: The car must learn to use momentum by going backwards first to build speed

## Files

- `env.py`: Custom environment wrapper with state discretization
- `setup.py`: Environment registration with Gymnasium
- `training.py`: Q-learning training loop
- `main.py`: Inference script for testing trained agent
- `q_table_mountaincar.npy`: Saved Q-table (generated after training)

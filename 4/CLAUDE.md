# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Q-learning implementation for the MountainCar climbing task using Gymnasium. The agent learns to drive an underpowered car up a steep mountain by building momentum through discretized state space compatible with tabular Q-learning.

## Key Commands

**Training:**
```bash
python training.py
```
Trains the agent for 1000 episodes and saves the Q-table to `q_table_mountaincar.npy`.

**Inference/Testing:**
```bash
python main.py
```
Runs 5 test episodes using the trained Q-table with visualization.

## Architecture

### Core Components

**env.py - DiscretizedMountainCarEnv**
- Custom Gymnasium environment wrapper that discretizes MountainCar's continuous state space
- Converts 2 continuous state variables into discrete bins: (20, 20)
  - Car position: 20 bins from -1.2 to 0.6
  - Car velocity: 20 bins from -0.07 to 0.07
- The `discretize_state()` method maps continuous values to bin indices using `np.digitize()`
- Returns discretized states as tuples for Q-table indexing

**setup.py**
- Registers the custom environment as "DiscretizedMountainCar-v0" with Gymnasium
- Must be imported before creating the environment

**training.py**
- Implements standard Q-learning algorithm
- Q-table shape: (20, 20, 3) - 2 state dimensions × 3 actions
- Uses epsilon-greedy exploration with decay (1.0 → 0.01)
- Learning rate (alpha): 0.1, Discount factor (gamma): 0.99
- Runs 1000 episodes (more than CartPole due to harder task)
- Saves trained Q-table as numpy array

**main.py**
- Loads pre-trained Q-table for inference
- Runs greedy policy (no exploration, always argmax)
- Includes visualization and performance logging

## Important Implementation Details

- State discretization happens in both `reset()` and `step()` methods of the environment wrapper
- Q-table is indexed directly using discrete state tuples: `q_table[state][action]`
- The environment wrapper must be registered via `setup.py` import before use
- MountainCar is harder than CartPole - the car must learn to build momentum by rocking back and forth
- The saved Q-table (`q_table_mountaincar.npy`) is required for inference
- Uses finer discretization (20 bins) compared to CartPole (10 bins) for better precision in the 2D state space

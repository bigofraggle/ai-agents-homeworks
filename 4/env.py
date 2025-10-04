import gymnasium as gym
from gymnasium import spaces
import numpy as np


class DiscretizedMountainCarEnv(gym.Env):
    """
    A wrapper for MountainCar that discretizes the continuous state space
    for Q-learning compatibility.
    """

    def __init__(self, render_mode=None):
        super(DiscretizedMountainCarEnv, self).__init__()

        # Create the base MountainCar environment
        self.env = gym.make("MountainCar-v0", render_mode=render_mode)

        # Actions: 0 = push left, 1 = do nothing, 2 = push right
        self.action_space = spaces.Discrete(3)

        # Define discretization bins for each state variable
        # MountainCar state: [position, velocity]
        self.bins = [
            np.linspace(-1.2, 0.6, 20),    # Car position
            np.linspace(-0.07, 0.07, 20),  # Car velocity
        ]

        # Observation space is discrete
        self.observation_space = spaces.MultiDiscrete([20, 20])

        self.episode = 0
        self.total_reward = 0

    def discretize_state(self, state):
        """Convert continuous state to discrete state indices"""
        discrete_state = []
        for i, value in enumerate(state):
            discrete_state.append(np.digitize(value, self.bins[i]) - 1)
        return tuple(np.clip(discrete_state, 0, 19))

    def step(self, action):
        # Take action in the environment
        state, reward, terminated, truncated, info = self.env.step(action)

        # Store continuous state for visualization
        self.continuous_state = state

        # Discretize the state
        discrete_state = self.discretize_state(state)

        # Track total reward
        self.total_reward += reward

        # Print episode completion
        if terminated or truncated:
            print(f"Episode {self.episode} finished - Total Reward: {self.total_reward:.1f}")

        return discrete_state, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        # Reset the base environment
        state, info = self.env.reset(seed=seed)

        # Increment episode counter
        self.episode += 1
        self.total_reward = 0

        # Store continuous state for visualization
        self.continuous_state = state

        # Discretize the initial state
        discrete_state = self.discretize_state(state)

        return discrete_state, info

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()

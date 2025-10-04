import numpy as np
import gymnasium as gym
import setup
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Create environment
env = gym.make("DiscretizedMountainCar-v0")

# Q-learning parameters
alpha = 0.1  # Learning rate
gamma = 0.99  # Discount factor
epsilon = 1.0  # Initial exploration rate
epsilon_min = 0.01  # Minimum exploration rate
epsilon_decay = 0.995  # Decay rate for epsilon
episodes = 6000  # Number of training episodes

# Q-table: 20x20 state space, 3 actions
q_table = np.zeros((20, 20, 3))

# Track performance
rewards_history = []
steps_history = []
success_history = []  # Track if car reached the goal

for episode in range(episodes):
    state, info = env.reset()
    done = False
    total_reward = 0
    steps = 0

    while not done:
        steps += 1
        # Epsilon-greedy action selection
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # Explore
        else:
            action = np.argmax(q_table[state])  # Exploit

        # Take action
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward

        # Q-learning update
        best_next_action = np.argmax(q_table[next_state])
        q_table[state][action] = q_table[state][action] + alpha * (
            reward + gamma * q_table[next_state][best_next_action] - q_table[state][action]
        )

        state = next_state

    # Decay epsilon
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    # Track rewards, steps, and success (MountainCar succeeds when reward > -200)
    rewards_history.append(total_reward)
    steps_history.append(steps)
    success_history.append(1 if total_reward > -200 else 0)

    # Print progress
    if (episode + 1) % 100 == 0:
        avg_reward = np.mean(rewards_history[-100:])
        avg_steps = np.mean(steps_history[-100:])
        success_rate = np.mean(success_history[-100:]) * 100
        print(f"Episode {episode + 1}/{episodes} - Avg Reward: {avg_reward:.2f} - Avg Steps: {avg_steps:.0f} - Success Rate: {success_rate:.1f}% - Epsilon: {epsilon:.3f}")

# Save trained Q-table
np.save("q_table_mountaincar.npy", q_table)
print("\nTraining completed and Q-table saved!")
print(f"Final average reward (last 100 episodes): {np.mean(rewards_history[-100:]):.2f}")
print(f"Final success rate (last 100 episodes): {np.mean(success_history[-100:]) * 100:.1f}%")

env.close()

# Create visualization of training progress
print("\nGenerating training visualization...")

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('MountainCar Q-Learning Training Progress', fontsize=16, fontweight='bold')

# Calculate moving averages for smoother plots
window_size = 100

def moving_average(data, window):
    return np.convolve(data, np.ones(window)/window, mode='valid')

# Plot 1: Episode Rewards
ax1 = axes[0, 0]
ax1.plot(rewards_history, alpha=0.3, color='blue', label='Raw Rewards')
if len(rewards_history) >= window_size:
    ma_rewards = moving_average(rewards_history, window_size)
    ax1.plot(range(window_size-1, len(rewards_history)), ma_rewards,
             color='red', linewidth=2, label=f'{window_size}-Episode Moving Avg')
ax1.set_xlabel('Episode', fontsize=12)
ax1.set_ylabel('Total Reward', fontsize=12)
ax1.set_title('Rewards per Episode', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend()

# Plot 2: Steps per Episode
ax2 = axes[0, 1]
ax2.plot(steps_history, alpha=0.3, color='green', label='Raw Steps')
if len(steps_history) >= window_size:
    ma_steps = moving_average(steps_history, window_size)
    ax2.plot(range(window_size-1, len(steps_history)), ma_steps,
             color='darkgreen', linewidth=2, label=f'{window_size}-Episode Moving Avg')
ax2.set_xlabel('Episode', fontsize=12)
ax2.set_ylabel('Steps', fontsize=12)
ax2.set_title('Steps per Episode', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend()

# Plot 3: Success Rate Over Time
ax3 = axes[1, 0]
if len(success_history) >= window_size:
    success_rate_over_time = [np.mean(success_history[max(0, i-window_size):i+1]) * 100
                               for i in range(len(success_history))]
    ax3.plot(success_rate_over_time, color='purple', linewidth=2)
    ax3.fill_between(range(len(success_rate_over_time)), 0, success_rate_over_time,
                     alpha=0.3, color='purple')
ax3.set_xlabel('Episode', fontsize=12)
ax3.set_ylabel('Success Rate (%)', fontsize=12)
ax3.set_title(f'Success Rate (Rolling {window_size}-Episode Window)', fontsize=14, fontweight='bold')
ax3.set_ylim([0, 105])
ax3.grid(True, alpha=0.3)
ax3.axhline(y=50, color='orange', linestyle='--', linewidth=1, label='50% Success')
ax3.axhline(y=90, color='red', linestyle='--', linewidth=1, label='90% Success')
ax3.legend()

# Plot 4: Q-Value Heatmap (max Q-value at each state)
ax4 = axes[1, 1]
max_q_values = np.max(q_table, axis=2)
im = ax4.imshow(max_q_values.T, cmap='viridis', aspect='auto', origin='lower')
ax4.set_xlabel('Position Bin', fontsize=12)
ax4.set_ylabel('Velocity Bin', fontsize=12)
ax4.set_title('Learned Q-Values (Max per State)', fontsize=14, fontweight='bold')
cbar = plt.colorbar(im, ax=ax4)
cbar.set_label('Max Q-Value', fontsize=10)

plt.tight_layout()
plt.savefig('training_progress.png', dpi=300, bbox_inches='tight')
print("Training visualization saved as 'training_progress.png'")
plt.show()

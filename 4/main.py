import numpy as np
import gymnasium as gym
import setup
import time
import matplotlib.pyplot as plt

# Load trained Q-table
q_table = np.load("q_table_mountaincar.npy")

print("\n=== Q-Learning MountainCar Agent ===")
print("Dimensions: (20x20 discrete states, 3 actions)")
print("Actions: 0 = Push Left, 1 = Do Nothing, 2 = Push Right\n")

# Create environment without live visualization for faster execution
env = gym.make("DiscretizedMountainCar-v0")

# Run multiple test episodes
num_episodes = 5
total_rewards = []
all_episode_data = []  # Store data for visualization

for episode in range(num_episodes):
    state, info = env.reset()
    done = False
    episode_reward = 0
    steps = 0

    # Track trajectory for this episode
    positions = []
    velocities = []
    actions_taken = []
    q_value_maxes = []

    # Get reference to unwrapped environment
    unwrapped_env = env.unwrapped

    print(f"\n--- Episode {episode + 1} ---")

    while not done:
        # Always choose the best action (no exploration)
        action = np.argmax(q_table[state])

        # Get Q-values for current state
        q_values = q_table[state]

        # Get actual continuous state for tracking
        positions.append(unwrapped_env.continuous_state[0])
        velocities.append(unwrapped_env.continuous_state[1])
        actions_taken.append(action)
        q_value_maxes.append(np.max(q_values))

        # Print state and action info every 20 steps
        if steps % 20 == 0:
            action_names = ['Push Left', 'Do Nothing', 'Push Right']
            print(f"Step {steps}: State {state} | Position: {unwrapped_env.continuous_state[0]:.3f} | Velocity: {unwrapped_env.continuous_state[1]:.4f}")
            print(f"  Q-values: Left={q_values[0]:.3f}, Nothing={q_values[1]:.3f}, Right={q_values[2]:.3f}")
            print(f"  Action: {action_names[action]}")

        # Take action
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        episode_reward += reward
        steps += 1

        # Small delay for visualization
        time.sleep(0.02)

    total_rewards.append(episode_reward)
    all_episode_data.append({
        'positions': positions,
        'velocities': velocities,
        'actions': actions_taken,
        'q_values': q_value_maxes,
        'reward': episode_reward,
        'steps': steps
    })
    print(f"Episode {episode + 1} finished - Steps: {steps}, Total Reward: {episode_reward:.1f}")

env.close()

print("\n=== Testing Complete ===")
print(f"Average reward over {num_episodes} episodes: {np.mean(total_rewards):.2f}")
print(f"Best performance: {np.max(total_rewards):.1f} reward")
print(f"Worst performance: {np.min(total_rewards):.1f} reward")

# Create comprehensive visualization of test episodes
print("\nGenerating test episode visualization...")

fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Color scheme for episodes
colors = plt.cm.tab10(range(num_episodes))
action_colors = {0: 'red', 1: 'gray', 2: 'blue'}
action_names = {0: 'Left', 1: 'Nothing', 2: 'Right'}

# Plot 1: Position Trajectory over Time (all episodes)
ax1 = fig.add_subplot(gs[0, :2])
for i, data in enumerate(all_episode_data):
    ax1.plot(data['positions'], color=colors[i], linewidth=2,
             label=f"Episode {i+1} ({data['steps']} steps)", alpha=0.8)
ax1.axhline(y=0.5, color='green', linestyle='--', linewidth=2, label='Goal Position')
ax1.axhline(y=-1.2, color='orange', linestyle='--', linewidth=1, alpha=0.5)
ax1.set_xlabel('Step', fontsize=12)
ax1.set_ylabel('Position', fontsize=12)
ax1.set_title('Car Position Trajectory', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(loc='best', fontsize=9)

# Plot 2: Performance Summary
ax2 = fig.add_subplot(gs[0, 2])
episode_nums = list(range(1, num_episodes + 1))
bars = ax2.bar(episode_nums, [d['steps'] for d in all_episode_data],
               color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
ax2.set_xlabel('Episode', fontsize=12)
ax2.set_ylabel('Steps to Goal', fontsize=12)
ax2.set_title('Steps per Episode', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')
# Add value labels on bars
for i, (bar, data) in enumerate(zip(bars, all_episode_data)):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

# Plot 3: Phase Space Diagram (Position vs Velocity) - Best Episode
best_episode_idx = np.argmax(total_rewards)
best_data = all_episode_data[best_episode_idx]
ax3 = fig.add_subplot(gs[1, 0])
scatter = ax3.scatter(best_data['positions'], best_data['velocities'],
                     c=range(len(best_data['positions'])), cmap='viridis',
                     s=20, alpha=0.6, edgecolors='black', linewidth=0.5)
ax3.plot(best_data['positions'], best_data['velocities'],
         color='black', alpha=0.3, linewidth=1, linestyle='--')
ax3.set_xlabel('Position', fontsize=12)
ax3.set_ylabel('Velocity', fontsize=12)
ax3.set_title(f'Phase Space - Best Episode ({best_episode_idx+1})', fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3)
cbar = plt.colorbar(scatter, ax=ax3)
cbar.set_label('Step', fontsize=10)

# Plot 4: Action Distribution - Best Episode
ax4 = fig.add_subplot(gs[1, 1])
action_counts = [best_data['actions'].count(i) for i in range(3)]
bars = ax4.bar(['Push Left', 'Do Nothing', 'Push Right'], action_counts,
               color=['red', 'gray', 'blue'], alpha=0.7, edgecolor='black', linewidth=2)
ax4.set_ylabel('Count', fontsize=12)
ax4.set_title(f'Action Distribution - Episode {best_episode_idx+1}', fontsize=14, fontweight='bold')
ax4.grid(True, alpha=0.3, axis='y')
# Add value labels
for bar in bars:
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

# Plot 5: Q-Values over Time - Best Episode
ax5 = fig.add_subplot(gs[1, 2])
ax5.plot(best_data['q_values'], color='purple', linewidth=2)
ax5.fill_between(range(len(best_data['q_values'])), best_data['q_values'],
                 alpha=0.3, color='purple')
ax5.set_xlabel('Step', fontsize=12)
ax5.set_ylabel('Max Q-Value', fontsize=12)
ax5.set_title(f'Q-Value Evolution - Episode {best_episode_idx+1}', fontsize=14, fontweight='bold')
ax5.grid(True, alpha=0.3)

# Plot 6: Actions over Position - Best Episode
ax6 = fig.add_subplot(gs[2, :])
for action_type in range(3):
    action_positions = [pos for pos, act in zip(best_data['positions'], best_data['actions'])
                       if act == action_type]
    action_steps = [step for step, act in enumerate(best_data['actions']) if act == action_type]
    if action_positions:
        ax6.scatter(action_positions, [action_type]*len(action_positions),
                   c=action_steps, cmap='viridis', s=50, alpha=0.6,
                   label=action_names[action_type], edgecolors='black', linewidth=0.5)

ax6.set_xlabel('Position', fontsize=12)
ax6.set_ylabel('Action', fontsize=12)
ax6.set_yticks([0, 1, 2])
ax6.set_yticklabels(['Push Left', 'Do Nothing', 'Push Right'])
ax6.set_title(f'Actions vs Position - Best Episode ({best_episode_idx+1})', fontsize=14, fontweight='bold')
ax6.axvline(x=0.5, color='green', linestyle='--', linewidth=2, label='Goal', alpha=0.7)
ax6.axvline(x=-1.2, color='orange', linestyle='--', linewidth=2, label='Start Boundary', alpha=0.5)
ax6.grid(True, alpha=0.3)
ax6.legend(loc='best', fontsize=10)

fig.suptitle('MountainCar Q-Learning Test Results Analysis',
            fontsize=18, fontweight='bold', y=0.995)

plt.savefig('test_results.png', dpi=300, bbox_inches='tight')
print("Test visualization saved as 'test_results.png'")
plt.show()

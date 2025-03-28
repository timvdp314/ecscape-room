import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Callable

def plot_results(sarsa_values, q_values, sarsa_policy, q_policy, title="SARSA vs Q-Learning"):
    """
    Compares SARSA and Q-Learning by visualizing their reward values as heatmaps and overlaying arrows for policies.

    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    arrow_map = {0: (0, -0.3), 1: (0, 0.3), 2: (-0.3, 0), 3: (0.3, 0)}
    action_symbols = {0: "↑", 1: "↓", 2: "←", 3: "→"}
    
    for ax, values, policy, method in zip(axes, [sarsa_values, q_values], [sarsa_policy, q_policy], ["SARSA", "Q-Learning"]):
        img = ax.imshow(values, cmap="coolwarm", interpolation="nearest")
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                action = policy[i, j]
                dx, dy = arrow_map.get(action, (0, 0))
                ax.arrow(j, i, dx, dy, color="black", head_width=0.15)
                ax.text(j, i, action_symbols.get(action, ""), ha="center", va="center", fontsize=14, color="white")
        ax.set_title(f"{method} Policy")
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.suptitle(title)
    fig.colorbar(img, ax=axes.ravel().tolist(), location='right', label="State Values")
    plt.show()

def plot_total_rewards(rewards: dict[str, list[float]], title: str, alpha_factor: float, gamma_factor: float, num_episodes: int):
    """Plots the learning curves of SARSA and Q-Learning."""
    plt.figure(figsize=(8, 5))

    for key, vals in rewards.items():
        plt.plot(vals, label = key)

    plt.xlabel('Episodes')
    plt.ylabel('Cumulative reward per episode')
    plt.suptitle("Alpha = {:.2f}, Gamma = {:.2f}, Num. episodes = {}".format(alpha_factor, gamma_factor, num_episodes))
    plt.title(title)
    plt.legend()
    plt.show()

def plot_state_visits(grid: dict[tuple[int, int], int], grid_size: int, episode_length: int, title: str):
    heatmap_data = np.zeros((grid_size, grid_size))

    for (x, y), val in grid.items():
        heatmap_data[(y, x)] = float(val) / episode_length

    plt.figure(figsize=(6, 5))  # Adjust figure size
    sns.heatmap(heatmap_data, annot=True, cmap="coolwarm", fmt=".2f")

    plt.title(title)
    plt.show()

# Plot a Policy Iteration Heatmap.
def plot_pi_heatmap(grid_size: int, movement: dict[(int, int)], potential_rewards) -> None:
    plot_directional_heatmap("Policy Iteration Heatmap", grid_size, potential_rewards, movement, movement_cell_definition)

# Plot a Value Iteration Heatmap.
def plot_vi_heatmap(grid_size: int, optimal_actions: dict[(int, int)], potential_rewards: dict[(int, int)]) -> None:
    plot_directional_heatmap("Value Iteration Heatmap", grid_size, potential_rewards, optimal_actions, vi_cell_definition)

# Plot a Policy Iteration Heatmap.
def plot_sarsa_heatmap(grid_size: int, movement: dict[(int, int)], potential_rewards) -> None:
    plot_directional_heatmap("TD Sarsa Heatmap", grid_size, potential_rewards, movement, movement_cell_definition)

# This function gives the definition of a cell for a movement dictionary Heatmap.
# It is made a callback function in order to minimize duplicate code.
def movement_cell_definition(x: int, y: int, arrow_size_multiplier: float, movement: dict[(int, int)]) -> None:
    for action in movement[(x, y)]:
        plt.arrow(x, y,
                  action[0] * arrow_size_multiplier,
                  action[1] * arrow_size_multiplier,
                  head_width=0.2,
                  head_length=0.3,
                  fc='k',
                  ec='k')

# This function gives the definition of a cell in the Value Iteration Heatmap.
# It is made a callback function in order to minimize duplicate code.
def vi_cell_definition(x: int, y: int, arrow_size_multiplier: float, optimal_actions: dict[(int, int)]) -> None:
    action = optimal_actions[(x, y)]
    plt.arrow(x, y,
              action[0] * arrow_size_multiplier,
              action[1] * arrow_size_multiplier,
              head_width=0.2,
              head_length=0.3,
              fc='k',
              ec='k')

# The generalized code of plotting a directional heatmap. The cell_def_function is the callback function which defines the cell for a specific algorithm.
def plot_directional_heatmap(title: str, grid_size: int, potential_rewards: dict[(int, int)], actions: dict[(int, int)], cell_def_function: Callable) -> None:
    fig, ax = plt.subplots()

    rewards = [[0 for _ in range(grid_size)] for _ in range(grid_size)]
    for coordinates, reward in potential_rewards.items():
        rewards[coordinates[1]][coordinates[0]] = reward

    ax.imshow(rewards)

    ax.set_xticks(range(grid_size))
    ax.set_yticks(range(grid_size))

    arrow_size_multiplier: float = 0.15
    for x in range(grid_size):
        for y in range(grid_size):
            ax.text(x, y, round(potential_rewards[(x, y)], 2), ha="center", va="center", color="w")

            # Call the callback function to draw the specific cell.
            cell_def_function(x, y, arrow_size_multiplier, actions)

    ax.set_title(title)
    fig.tight_layout()
    plt.show()
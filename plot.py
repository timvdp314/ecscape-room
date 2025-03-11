import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

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

def plot_total_rewards(rewards: dict[str, list[float]], title: str):
    """Plots the learning curves of SARSA and Q-Learning."""
    plt.figure(figsize=(8, 5))

    for key, vals in rewards.items():
        plt.plot(vals, label = key)

    # plt.plot(sarsa_rewards, label='SARSA', color='blue')
    # plt.plot(q_rewards, label='Q-Learning', color='red')
    plt.xlabel('Episodes')
    plt.ylabel('Cumulative reward per episode')
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
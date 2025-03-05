import numpy as np
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

def plot_learning_curves(sarsa_rewards, q_rewards):
    """Plots the learning curves of SARSA and Q-Learning."""
    plt.figure(figsize=(8, 5))
    plt.plot(sarsa_rewards, label='SARSA', color='blue')
    plt.plot(q_rewards, label='Q-Learning', color='red')
    plt.xlabel('Episodes')
    plt.ylabel('Cumulative Reward')
    plt.title('SARSA vs Q-Learning Learning Curves')
    plt.legend()
    plt.show()

# test
grid_size = (4, 4)
sarsa_values = np.random.rand(*grid_size)  
q_values = np.random.rand(*grid_size)  
sarsa_policy = np.random.randint(0, 4, grid_size)  
q_policy = np.random.randint(0, 4, grid_size)  

plot_results(sarsa_values, q_values, sarsa_policy, q_policy, "SARSA vs Q-Learning")

# test
sarsa_rewards = np.cumsum(np.random.randn(100))  
q_rewards = np.cumsum(np.random.randn(100))  
plot_learning_curves(sarsa_rewards, q_rewards)

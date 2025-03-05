import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_values(values, title, ax):
    sns.heatmap(values, annot=True, fmt=".2f", cmap="coolwarm", ax=ax, cbar=True)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])

def plot_policies(policies, title, ax):
    direction_map = {0: "←", 1: "↓", 2: "→", 3: "↑", None: "·"}
    policy_grid = np.vectorize(direction_map.get)(policies)
    ax.table(cellText=policy_grid, loc='center', cellLoc='center')
    ax.set_title(title)
    ax.axis("off")

def compare_algorithms(value_iteration_values, value_iteration_policy,
                        policy_iteration_values, policy_iteration_policy,
                        random_values, random_policy):
    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    
    plot_values(value_iteration_values, "Value Iteration - Values", axes[0, 0])
    plot_policies(value_iteration_policy, "Value Iteration - Policy", axes[0, 1])
    
    plot_values(policy_iteration_values, "Policy Iteration - Values", axes[1, 0])
    plot_policies(policy_iteration_policy, "Policy Iteration - Policy", axes[1, 1])
    
    plot_values(random_values, "Random Policy - Values", axes[2, 0])
    plot_policies(random_policy, "Random Policy - Policy", axes[2, 1])
    
    plt.tight_layout()
    plt.show()

# Example dummy data for testing
size = (5, 5)
value_iteration_values = np.random.rand(*size)
policy_iteration_values = np.random.rand(*size)
random_values = np.random.rand(*size)

value_iteration_policy = np.random.randint(0, 4, size)
policy_iteration_policy = np.random.randint(0, 4, size)
random_policy = np.random.randint(0, 4, size)

compare_algorithms(value_iteration_values, value_iteration_policy,
                   policy_iteration_values, policy_iteration_policy,
                   random_values, random_policy)

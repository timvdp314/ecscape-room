import numpy as np
import matplotlib.pyplot as plt

def plot_results(values, policy, title="MDP Visualization"):
    """
    Visualizes the reward values as a heatmap and overlays arrows for policies.
    
    """
    plt.figure(figsize=(8, 8))
    plt.imshow(values, cmap="coolwarm", interpolation="nearest")
    
    # Define arrow directions based on policy outputs
    arrow_map = {0: (0, -0.3), 1: (0, 0.3), 2: (-0.3, 0), 3: (0.3, 0)}
    action_symbols = {0: "↑", 1: "↓", 2: "←", 3: "→"}
    
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            action = policy[i, j]
            dx, dy = arrow_map.get(action, (0, 0))
            plt.arrow(j, i, dx, dy, color="black", head_width=0.15)
            plt.text(j, i, action_symbols.get(action, ""), ha="center", va="center", fontsize=14, color="white")

    plt.title(title)
    plt.colorbar(label="State Values")
    plt.xticks([])
    plt.yticks([])
    plt.show()

#test
grid_size = (4, 4)
values = np.random.rand(*grid_size)  
policy = np.random.randint(0, 4, grid_size)  

plot_results(values, policy, "Policy Iteration Results")
plot_results(values, policy, "Value Iteration Results")

from policies.policy import Policy

import matplotlib.pyplot as plt

def plot_dp_heatmap(policy: Policy, potential_rewards) -> None:
    fig, ax = plt.subplots()

    rewards = [[0 for _ in range(policy.grid_size)] for _ in range(policy.grid_size)]
    for coordinates, reward in potential_rewards.items():
        rewards[coordinates[1]][coordinates[0]] = reward

    ax.imshow(rewards)

    ax.set_xticks(range(policy.grid_size))
    ax.set_yticks(range(policy.grid_size))

    arrow_size_multiplier: float = 0.15
    for x in range(policy.grid_size):
        for y in range(policy.grid_size):
            ax.text(x, y, round(potential_rewards[(x, y)], 2), ha="center", va="center", color="w")

            for action in policy.movement[(x, y)]:
                plt.arrow(x, y,
                      action[0] * arrow_size_multiplier,
                      action[1] * arrow_size_multiplier,
                      head_width=0.2,
                      head_length=0.3,
                      fc='k',
                      ec='k')

    ax.set_title("Dynamic Programming Heatmap")
    fig.tight_layout()
    plt.show()

def plot_vi_heatmap(grid_size: int, optimal_actions: dict[(int, int)], potential_rewards: dict[(int, int)]) -> None:
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

            action = optimal_actions[(x, y)]
            plt.arrow(x, y,
                      action[0] * arrow_size_multiplier,
                      action[1] * arrow_size_multiplier,
                      head_width=0.2,
                      head_length=0.3,
                      fc='k',
                      ec='k')

    ax.set_title("Value Iteration Heatmap")
    fig.tight_layout()
    plt.show()
import random
from typing import Callable

import matplotlib.pyplot as plt

from cell import Cell


def init_policy(grid: dict[tuple[int, int], Cell], grid_size: int, movement: dict[tuple[int, int], dict[tuple[int, int], float]]) -> None:
    for x in range(0, grid_size):
        for y in range(0, grid_size):
            movement[(x, y)] = dict()
            if x > 0:
                add_action(grid, movement, (x, y), (-1, 0))
            if x < grid_size - 1:
                add_action(grid, movement, (x, y), (1, 0))
            if y > 0:
                add_action(grid, movement, (x, y), (0, -1))
            if y < grid_size - 1:
                add_action(grid, movement, (x, y), (0, 1))

            probability: float = 1 / len(movement[(x, y)])
            for key in movement[(x, y)]:
                movement[(x, y)][key] = probability

def add_action(grid, movement, position, action) -> None:
    new_pos = (position[0] + action[0], position[1] + action[1])
    if not grid[new_pos].is_solid:
        movement[position][action] = 0

def select_prob_move(grid_pos: (int, int), movement: dict[tuple[int, int], dict[tuple[int, int], float]]) -> (int, int):
    num = random.uniform(0, 1)
    threshold = 0.0

    for act, prob in movement[grid_pos].items():
        threshold += prob
        if num <= threshold:
            return act

    assert "Something is wrong with the agent's policy!"
    return None

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

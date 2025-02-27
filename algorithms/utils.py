from typing import Callable
from typing import NamedTuple

import matplotlib.pyplot as plt
import random

from math_utils import tuple_addition

class Observation(NamedTuple):
    grid_pos: tuple[int, int]
    reward: float
    is_terminal: bool = False

def sample_policy_action(policy: dict[tuple[int, int], dict[tuple[int, int]]], grid_pos: tuple[int, int]):
    num = random.uniform(0, 1)
    threshold = 0.0

    for act, prob in policy[grid_pos].items():
        threshold += prob
        if num <= threshold:
            return act

    assert("Something is wrong with the agent's policy!")
    return None

def generate_episode(policy: dict[tuple[int, int], dict[tuple[int, int]]], grid_action_cb: Callable[[tuple[int, int], tuple[int, int]], Observation],
                     grid_pos: tuple[int, int]) -> list[tuple[tuple[int, int], int], tuple[int, int]]:
    episode: list[tuple[tuple[int, int], int], tuple[int, int]] = list()

    state = grid_pos

    while (True):
        act = sample_policy_action(policy, state)
        next_state = tuple_addition(state, act)

        obs = grid_action_cb(state, act)
        episode.append((state, act, obs.reward))

        if (obs.is_terminal):
            break

        state = next_state  

    return episode

# Plot a Policy Iteration Heatmap.
def plot_pi_heatmap(grid_size: int, movement: dict[(int, int)], potential_rewards) -> None:
    plot_directional_heatmap("Policy Iteration Heatmap", grid_size, potential_rewards, movement, pi_cell_definition)

# Plot a Value Iteration Heatmap.
def plot_vi_heatmap(grid_size: int, optimal_actions: dict[(int, int)], potential_rewards: dict[(int, int)]) -> None:
    plot_directional_heatmap("Value Iteration Heatmap", grid_size, potential_rewards, optimal_actions, vi_cell_definition)

# This function gives the definition of a cell in the Policy Iteration Heatmap.
# It is made a callback function in order to minimize duplicate code.
def pi_cell_definition(x: int, y: int, arrow_size_multiplier: float, movement: dict[(int, int)]) -> None:
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

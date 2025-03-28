import random
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

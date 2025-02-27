import matplotlib.pyplot as plt
from abc import ABC
from typing import Callable
import numpy as np

from algorithms.algorithm import Algorithm
from algorithms.utils import Observation
from cell import Cell
from algorithms.utils import generate_episode

class MonteCarloAlgorithm(Algorithm):
    def __init__(self, policy: dict[tuple[int, int], dict[tuple[int, int], float]], grid_action_cb: Callable[[tuple[int, int], tuple[int, int]], Observation], 
                 grid_pos: tuple[int, int], grid_size: int):
        super().__init__(policy, grid_action_cb, grid_pos, grid_size)
    
    def run(self, gamma_factor: float = 0.90, num_episodes: int = 500):
        """ Estimates the value of each state-action pair based on the current policy.
        Updates the policy accordingly, taking into account the epsilon factor
        """

        state_action_returns: dict[tuple[int, int], dict[tuple[int, int], list[float]]] = dict()
        state_action_vals: dict[tuple[int, int], dict[tuple[int, int], float]] = dict()

        for x in range(self.grid_size):
            for y in range(self.grid_size):
                state_action_returns[(x, y)] = dict()
                state_action_vals[(x, y)] = dict()

        episode: list[tuple[tuple[int, int], int]] = list()
        visited_state_actions: list[tuple[int, int, int, int]] = list()

        reward: float = 0.0

        # Traverse a number of episodes
        for n in range(num_episodes):
            # Reset all relevant variables for a new episode
            visited_state_actions.clear()

            # The epsilon factor (or exploration factor) determines how eager the algorithm is to randomly choose an unoptimal path
            # As the algorithm starts to converge, the algorithm should focus more on exploitation, thus the epsilon factor get smaller
            epsilon_factor: float = 1.0 / (n + 1)

            episode = generate_episode(self.policy, self.grid_action_cb, self.grid_pos)
            reward = 0.0

            for s in reversed(episode):
                # Calculate state-action pair value
                reward = gamma_factor * reward + s[2]

                # Only add return value if state-action pair has not been visited (i.e. first-visit only)
                # Note that "s[0] + s[1]" appends the two tuples of state and action (instead of summing their values), resulting in a 4-int tuple
                if (s[0] + s[1]) not in visited_state_actions:
                    visited_state_actions.append(s[0] + s[1])

                    if (s[1] not in state_action_returns[s[0]]):
                        state_action_returns[s[0]][s[1]] = list()

                    state_action_returns[s[0]][s[1]].append(reward)

                    # Calculate average state-action pair value based on average
                    state_action_vals[s[0]][s[1]] = np.mean(state_action_returns[s[0]][s[1]])

                    # Select best action from state-action pairs
                    best_action = max(state_action_vals[s[0]], key = state_action_vals[s[0]].get)
                    self.policy[s[0]][best_action] = (1.0 - epsilon_factor + epsilon_factor / len(self.policy[s[0]]))

                    # Assign exploration probabilities according to epsilon factor
                    for act in self.policy[s[0]].keys():
                        if (act != best_action):
                            self.policy[s[0]][act] = epsilon_factor / len(self.policy[s[0]])
        
    def terminate(self):
        super().terminate(self)
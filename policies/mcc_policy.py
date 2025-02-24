from abc import ABC

from policies.policy import Policy
from cell import Cell
from policies.policy_utils import generate_episode

import matplotlib.pyplot as plt

import numpy as np

class MonteCarloControlPolicy(Policy, ABC):
    def __init__(self, grid_pos: tuple[int, int], grid: dict[tuple[int, int], Cell], grid_size: int = 6):
        super().__init__(grid_pos, grid, grid_size)

        self.mc_control()
    
    def mc_control(self, gamma_factor: float = 0.90, num_episodes: int = 100):
        """ Estimates the value of each state-action pair based on the current policy.
        Updates the policy accordingly, taking into account the epsilon factor
        """

        state_action_returns: dict[tuple[int, int], dict[tuple[int, int], list[float]]] = {state: dict() for state in self.grid.keys()}
        state_action_vals: dict[tuple[int, int], dict[tuple[int, int], float]] = {state: dict() for state in self.grid.keys()}

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

            episode = generate_episode(self, self.grid_pos)
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
                    self.movement[s[0]][best_action] = (1.0 - epsilon_factor + epsilon_factor / len(self.movement[s[0]]))

                    # Assign exploration probabilities according to epsilon factor
                    for act in self.movement[s[0]].keys():
                        if (act != best_action):
                            self.movement[s[0]][act] = epsilon_factor / len(self.movement[s[0]])
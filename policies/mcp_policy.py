from abc import ABC

from policies.policy import Policy
from cell import Cell
from policies.policy_utils import generate_episode

import matplotlib.pyplot as plt

import numpy as np

class MonteCarloPredictionPolicy(Policy, ABC):
    def __init__(self, grid_pos: tuple[int, int], grid: dict[tuple[int, int], Cell], grid_size: int = 6):
        super().__init__(grid_pos, grid, grid_size)

        self.og_grid_pos = grid_pos

        self.mc_prediction()
    
    def mc_prediction(self, gamma_factor: float = 0.90, num_episodes: int = 100):
        """ Estimates the value of each state based on the CURRENT policy.
        Does not update the policy.
        """

        state_returns: dict[tuple[int, int], list[float]] = {state: [] for state in self.grid.keys()}
        state_vals: dict[tuple[int, int], float] = {state: 0.0 for state in self.grid.keys()}

        episode: list[tuple[tuple[int, int], int], tuple[int, int]] = list()
        visited_states: list[tuple[int, int]] = list()

        # Traverse a number of episodes
        for n in range(num_episodes):
            episode = generate_episode(self, self.grid_pos)

            # Reset reward and visited states for new episode
            reward: float = 0
            visited_states.clear()

            for s in reversed(episode):
                # Calculate state value
                reward = gamma_factor * reward + s[2]

                # Add state value only upon first visiting a state
                if s[0] not in visited_states:
                    visited_states.append(s[0])
                    state_returns[s[0]].append(reward)

                    # Update state value by calculating average return value
                    state_vals[s[0]] = np.mean(state_returns[s[0]])

        # Plot heatmap
        self.plot_heatmap("Test", self.grid_size, state_vals)

    def plot_heatmap(self, title: str, grid_size: int, potential_rewards: dict[(int, int)]) -> None:
        fig, ax = plt.subplots()

        rewards = [[0 for _ in range(grid_size)] for _ in range(grid_size)]
        for coordinates, reward in potential_rewards.items():
            rewards[coordinates[1]][coordinates[0]] = reward

        ax.imshow(rewards)

        ax.set_xticks(range(grid_size))
        ax.set_yticks(range(grid_size))

        ax.set_title(title)
        fig.tight_layout()
        plt.show()
from abc import ABC

from policies.policy import Policy
from policies.utility import iterate_values, plot_vi_heatmap


class OptimalPolicy(Policy, ABC):
    optimal_actions: dict[(int, int)] = {}

    def __init__(self, grid_pos: (int, int), grid: dict[tuple[int, int]], grid_size: int = 6):
        super().__init__(grid_pos, grid, grid_size)
        self.optimal_actions, potential_rewards = iterate_values(self)
        plot_vi_heatmap(grid_size, self.optimal_actions, potential_rewards)

    def move(self):
        return self.optimal_actions[self.grid_pos]

    def terminate(self):
        return super().terminate()
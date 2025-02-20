from abc import ABC
from policies.policy import Policy
import random

from policies.utility import improve_policy, plot_dp_heatmap


class DPPolicy(Policy, ABC):
    def __init__(self, grid_pos: (int, int), grid: dict[tuple[int, int]], grid_size: int = 6):
        super().__init__(grid_pos, grid, grid_size)
        possible_rewards = improve_policy(self)

        plot_dp_heatmap(self, possible_rewards)

    def move(self):
        num = random.uniform(0, 1)
        threshold = 0.0

        for act, prob in self.movement[self.grid_pos].items():
            threshold += prob
            if num <= threshold:
                return act

        assert ("Something is wrong with the agent's policy!")
        return None

    def terminate(self):
        return super().terminate()

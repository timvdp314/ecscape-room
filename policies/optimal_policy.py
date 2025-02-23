import operator
from abc import ABC

from cell import Cell
from policies.policy import Policy
from policies.utility import plot_vi_heatmap


class OptimalPolicy(Policy, ABC):
    optimal_actions: dict[(int, int)] = {}

    def __init__(self, grid_pos: (int, int), grid: dict[tuple[int, int]], grid_size: int = 6):
        super().__init__(grid_pos, grid, grid_size)
        self.optimal_actions, potential_rewards = self.iterate_values()
        plot_vi_heatmap(grid_size, self.optimal_actions, potential_rewards)

    def move(self):
        return self.optimal_actions[self.grid_pos]

    def terminate(self):
        return super().terminate()

    def iterate_values(self, gamma: float = 0.8, theta: float = 0.001) -> (dict[(int, int)], dict[(int, int)]):
        potential_rewards = {state: 0 for state in self.grid.keys()}
        policy_actions: dict[(int, int)] = {state: (0, 0) for state in self.grid.keys()}

        delta: float = theta + 1
        while delta > theta:
            delta = 0

            for s in self.grid.keys():
                cell: Cell = self.grid[s]

                if cell.is_terminal:
                    continue

                v = potential_rewards[s]

                best_actions = {}
                for a in self.movement[s]:
                    next_state: (int, int) = (a[0] + s[0], a[1] + s[1])
                    best_actions[a] = cell.reward + gamma * potential_rewards[next_state]

                best_action, best_value = max(best_actions.items(), key=operator.itemgetter(1))
                policy_actions[s] = best_action
                potential_rewards[s] = best_value
                delta = max(delta, abs(v - best_value))

        return policy_actions, potential_rewards
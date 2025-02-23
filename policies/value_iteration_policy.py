import operator
from abc import ABC

from cell import Cell
from policies.policy import Policy
from policies.utility import plot_vi_heatmap

class ValueIterationPolicy(Policy, ABC):

    def __init__(self, grid_pos: (int, int), grid: dict[tuple[int, int]], grid_size: int = 6):
        super().__init__(grid_pos, grid, grid_size)
        optimal_actions, potential_rewards = self.iterate_values()

        # Update the movement of the policy based on the best actions found.
        for position in optimal_actions:
            self.movement[position] = optimal_actions[position]

        plot_vi_heatmap(grid_size, optimal_actions, potential_rewards)

    def move(self):
        return self.movement[self.grid_pos]

    def terminate(self):
        return super().terminate()

    def iterate_values(self, gamma: float = 0.8, theta: float = 0.001) -> (dict[(int, int)], dict[(int, int)]):
        potential_rewards = {state: 0 for state in self.grid.keys()} # A dictionary of the potential rewards for each Cell.
        policy_actions: dict[(int, int)] = {state: (0, 0) for state in self.grid.keys()} # The best actions for each given Cell.

        # Make delta higher than theta to start while loop.
        delta: float = theta + 1
        while delta > theta:
            # Reset delta.
            delta = 0

            # Iterate over each Cell in the grid (s).
            for s in self.grid.keys():
                cell: Cell = self.grid[s]

                # Skip terminal cells.
                if cell.is_terminal:
                    continue

                # Get old value
                v = potential_rewards[s]

                actions = {}
                # Iterate over each action to find the value for each action.
                for a in self.movement[s]:
                    # Create the new position given an action and the current position.
                    next_state: (int, int) = (a[0] + s[0], a[1] + s[1])
                    # Calculate the potential reward for an action.
                    actions[a] = cell.reward + gamma * potential_rewards[next_state]

                # Get the action with the highest value.
                best_action, best_value = max(actions.items(), key=operator.itemgetter(1))
                # Update the best actions policy dictionary.
                policy_actions[s] = best_action
                # Update the potential reward for this cell.
                potential_rewards[s] = best_value
                # Update delta to the difference of the old value and the new value, if larger than delta.
                delta = max(delta, abs(v - best_value))

        # Return the best actions and the potential rewards of the cells.
        return policy_actions, potential_rewards
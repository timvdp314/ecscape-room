import operator
from typing import Callable

from cell import Cell
from algorithms.algorithm import Algorithm
from algorithms.utils import Observation, plot_vi_heatmap

class ValueIterationAlgorithm(Algorithm):
    def __init__(self, policy: dict[tuple[int, int], dict[tuple[int, int], float]], grid_action_cb: Callable[[tuple[int, int], tuple[int, int]], Observation], 
                 grid_pos: tuple[int, int], grid_size: int):
        super().__init__(policy, grid_action_cb, grid_pos, grid_size)

    def run(self):
        optimal_actions, possible_rewards = self.iterate_values()

        # Update the movement of the policy based on the best actions found.
        for position in optimal_actions:
            self.policy[position].clear()
            self.policy[position][optimal_actions[position]] = 1.0

        plot_vi_heatmap(self.grid_size, optimal_actions, possible_rewards)

    def terminate(self):
        return super().terminate()

    def iterate_values(self, gamma: float = 0.8, theta: float = 0.001):
        potential_rewards: dict[tuple[int, int], float] = dict() # A dictionary of the potential rewards for each Cell.
        policy_actions: dict[(int, int)] = dict() # The best actions for each given Cell

        for x in range(self.grid_size):
            for y in range(self.grid_size):
                potential_rewards[(x,y)] = 0
                policy_actions[(x,y)] = (0, 0)

        # Make delta higher than theta to start while loop.
        delta: float = theta + 1
        while delta > theta:
            # Reset delta.
            delta = 0

            # Iterate over each Cell in the grid (s).
            for x in range(self.grid_size):
                for y in range(self.grid_size):
                    curr_state: Observation = self.grid_action_cb((x,y))

                    # Skip terminal cells.
                    if curr_state.is_terminal:
                        continue

                    s: tuple[int, int] = (x,y)

                    # Get old value
                    v = potential_rewards[s]

                    actions = {}
                    # Iterate over each action to find the value for each action.
                    for action in self.policy[s]:
                        # Create the new position given an action and the current position.
                        next_state = self.grid_action_cb(s, action)
                        # Calculate the potential reward for an action.
                        actions[action] = next_state.reward + gamma * potential_rewards[next_state.grid_pos]

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
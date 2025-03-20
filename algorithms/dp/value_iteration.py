import operator
from typing import Callable

from algorithms.algorithm import Algorithm
from algorithms.utils import Observation

class ValueIterationAlgorithm(Algorithm):
    def __init__(self, policy: dict[tuple[int, int], dict[tuple[int, int], float]], grid_action_cb: Callable[[tuple[int, int], tuple[int, int]], Observation], 
                 grid_pos: tuple[int, int], grid_size: int):
        super().__init__(policy, grid_action_cb, grid_pos, grid_size)

        self.potential_rewards: dict[tuple[int, int], float] = dict()
        self.optimal_actions: dict[tuple[int, int], tuple[int, int]] = dict()

    def terminate(self):
        return super().terminate()

    def run(self, gamma_factor: float = 0.8, theta_factor: float = 0.001):
        self.potential_rewards.clear() # A dictionary of the potential rewards for each Cell.
        self.optimal_actions.clear() # The best actions for each given Cell

        for x in range(self.grid_size):
            for y in range(self.grid_size):
                self.potential_rewards[(x,y)] = 0
                self.optimal_actions[(x,y)] = (0, 0)

        # Make delta higher than theta to start while loop.
        delta: float = theta_factor + 1.0
        while delta > theta_factor:
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
                    v = self.potential_rewards[s]

                    actions = {}
                    # Iterate over each action to find the value for each action.
                    for action in self.policy[s]:
                        # Create the new position given an action and the current position.
                        next_state = self.grid_action_cb(s, action)
                        # Calculate the potential reward for an action.
                        actions[action] = next_state.reward + gamma_factor * self.potential_rewards[next_state.grid_pos]

                    # Get the action with the highest value.
                    best_action, best_value = max(actions.items(), key=operator.itemgetter(1))
                    # Update the best actions policy dictionary.
                    self.optimal_actions[s] = best_action
                    # Update the potential reward for this cell.
                    self.potential_rewards[s] = best_value
                    # Update delta to the difference of the old value and the new value, if larger than delta.
                    delta = max(delta, abs(v - best_value))

        for position in self.optimal_actions:
            self.policy[position].clear()
            self.policy[position][self.optimal_actions[position]] = 1.0
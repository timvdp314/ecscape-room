from abc import ABC

from cell import Cell
from policies.policy import Policy
import random

from policies.utility import plot_pi_heatmap


class PolicyIterationPolicy(Policy, ABC):
    def __init__(self, grid_pos: (int, int), grid: dict[tuple[int, int]], grid_size: int = 6):
        super().__init__(grid_pos, grid, grid_size)
        possible_rewards = self.improve_policy()

        plot_pi_heatmap(grid_size, self.movement, possible_rewards)

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

    def evaluate_policy(self, potential_rewards: dict[(int, int)] = None, gamma: float = 0.8,
                        theta: float = 0.001) -> dict[(int, int)]:
        # If not provided with a list of possible rewards, initialize a list for the entire grid with value 0.
        if potential_rewards is None:
            potential_rewards = {state: 0 for state in self.grid.keys()}

        delta: float = theta + 1
        while delta > theta:
            delta = 0

            # Iterate for each cell in the grid.
            for s in self.grid.keys():
                v: float = 0
                # Get the cell associated with the coordinate (s).
                cell: Cell = self.grid[s]
                if cell.is_terminal:
                    continue

                # Iterate over each action possible in this cell.
                for action in self.movement[s]:
                    # Combine coordinate and action to create the new position following the action.
                    next_state = (action[0] + s[0], action[1] + s[1])
                    # In this simulation the actions are deterministic therefore it is multiplied by 1 instead of a probability.
                    v += self.movement[s][action] * (
                                self.grid[next_state].reward + gamma * potential_rewards[next_state])

                # Make the difference a positive number and replace it with delta if it is bigger.
                delta = max(delta, abs(v - potential_rewards[s]))
                # Add v as the possible reward for the coordinate s.
                potential_rewards[s] = v

        return potential_rewards

    def improve_policy(self, gamma: float = 0.8) -> dict[(int, int)]:
        policy_stable = False

        # While the policy is not stable, it is not optimal yet.
        while not policy_stable:
            # The policy is set to stable so that if no value is changed the loop stops.
            policy_stable = True
            potential_rewards: dict[(int, int)] = self.evaluate_policy()

            # Iterate for each cell in the grid.
            for s in self.grid.keys():
                # Get the cell associated with the coordinate (s).
                cell: Cell = self.grid[s]
                if cell.is_terminal:
                    continue

                # Save the current actions.
                old_actions: dict[(int, int)] = self.movement[s]
                # A dictionary of the new evaluated actions.
                eval_actions: dict[(int, int)] = {}

                # Iterate over each action possible in this cell.
                for action in self.movement[s]:
                    # Create the new position based on the action and the current position.
                    next_state: (int, int) = (action[0] + s[0], action[1] + s[1])
                    # Calculate the new reward given the action.
                    next_reward: float = self.grid[next_state].reward + gamma * potential_rewards[next_state]
                    eval_actions[action] = next_reward

                # Get the best actions out of the new actions, if there are multiple best actions give them all.
                best_actions: list[(int, int)] = [action for action, reward in eval_actions.items() if
                                                  reward == max(eval_actions.values())]
                # Distribute the probability evenly over the remaining actions.
                action_prob: float = 1 / len(best_actions)
                new_actions: dict[(int, int)] = {}

                for action in best_actions:
                    new_actions[action] = action_prob

                # If the actions have updated, mark the policy as stable to continue iterating.
                if old_actions != new_actions:
                    self.movement[s] = new_actions
                    policy_stable = False

            # Once the policy is done improving, return the possible rewards.
            if policy_stable:
                return potential_rewards

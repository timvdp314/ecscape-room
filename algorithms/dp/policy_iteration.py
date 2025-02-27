from abc import ABC
from typing import Callable

from algorithms.algorithm import Algorithm
from algorithms.utils import Observation, plot_pi_heatmap

class PolicyIterationAlgorithm(Algorithm):
    def __init__(self, policy: dict[tuple[int, int], dict[tuple[int, int], float]], grid_action_cb: Callable[[tuple[int, int], tuple[int, int]], Observation], 
                 grid_pos: tuple[int, int], grid_size: int):
        super().__init__(policy, grid_action_cb, grid_pos, grid_size)

        # possible_rewards = self.improve_policy()

        # plot_pi_heatmap(grid_size, self.movement, possible_rewards)

    def run(self):
        possible_rewards = self.improve_policy()

        plot_pi_heatmap(self.grid_size, self.policy, possible_rewards)

    def terminate(self):
        return super().terminate()

    def evaluate_policy(self, gamma: float = 0.8,
                        theta: float = 0.001) -> dict[(int, int)]:
        # If not provided with a list of possible rewards, initialize a list for the entire grid with value 0.

        potential_rewards: dict[tuple[int, int], float] = dict()

        for x in range(self.grid_size):
            for y in range (self.grid_size):                
                potential_rewards[(x, y)] = 0.0

        delta: float = theta + 1
        while delta > theta:
            delta = 0

            # Iterate for each cell in the grid.
            for x in range(self.grid_size):
                for y in range(self.grid_size):
                    v: float = 0
                    # Get the cell associated with the coordinate (s).
                    curr_state: Observation = self.grid_action_cb((x,y))

                    if curr_state.is_terminal:
                        continue

                    s: tuple[int, int] = (x,y)

                    # Iterate over each action possible in this cell.
                    for action in self.policy[s]:
                        # Combine coordinate and action to create the new position following the action.
                        next_state = self.grid_action_cb(s, action)
                        # In this simulation the actions are deterministic therefore it is multiplied by 1 instead of a probability.
                        v += self.policy[s][action] * (
                                    next_state.reward + gamma * potential_rewards[next_state.grid_pos])

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
            for x in range(self.grid_size):
                for y in range(self.grid_size):
                    # Get the cell associated with the coordinate (s).
                    curr_state: Observation = self.grid_action_cb((x,y), None)

                    if curr_state.is_terminal:
                        continue

                    # Save the current actions.
                    old_actions: dict[(int, int)] = self.policy[curr_state.grid_pos]
                    # A dictionary of the new evaluated actions.
                    eval_actions: dict[(int, int)] = {}

                    s: tuple[int, int] = curr_state.grid_pos

                    # Iterate over each action possible in this cell.
                    for action in self.policy[s]:
                        # Create the new position based on the action and the current position.
                        next_state: Observation = self.grid_action_cb(action, s)
                        # Calculate the new reward given the action.
                        next_reward: float = next_state.reward + gamma * potential_rewards[next_state.grid_pos]
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
                        self.policy[s] = new_actions
                        policy_stable = False

                # Once the policy is done improving, return the possible rewards.
                if policy_stable:
                    return potential_rewards

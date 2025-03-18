from abc import ABC
import random

from cell import Cell
from math_utils import tuple_addition
from policies.policy import Policy
from policies.policy_utils import generate_episode
from policies.utility import plot_pi_heatmap, init_policy, select_prob_move, plot_sarsa_heatmap


class TDSarsaPolicy(Policy, ABC):

    def __init__(self, grid_pos: (int, int), grid: dict[tuple[int, int], Cell], grid_size: int = 8):
        super().__init__(grid_pos, grid, grid_size)

        self.random_movement: dict[tuple[int, int], dict[tuple[int, int], float]] = dict()
        init_policy(grid, grid_size, self.random_movement)
        
        self.state_values = self.sarsa()
        self.process_state_values(self.state_values)

        plot_sarsa_heatmap(grid_size, self.movement, self.state_values)

    def move(self, grid_pos: tuple[int, int] = None) -> (int, int):
        return select_prob_move(self.grid_pos, self.movement)

    # Process the state values in order to update the movement strategy.
    def process_state_values(self, state_values: dict[(int, int), float]) -> None:
        # For each Cell in the grid update the movement strategy based on the state values.
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                eval_actions: dict[(int, int), float] = {}

                # Construct position based on the x and y coordinates.
                pos = (x, y)
                for a in self.movement[pos]:
                    # Create the new position based on the action and the current position.
                    next_state: (int, int) = tuple_addition(pos, a)

                    # If a final state is among the actions, it is taken as the only actions.
                    if self.grid[next_state].is_terminal:
                        eval_actions.clear()
                        eval_actions[a] = self.grid[next_state].reward
                        break

                    next_reward: float = state_values[next_state]
                    eval_actions[a] = next_reward

                # Get the best actions of the current state, based on the highest value associated.
                best_actions: list[(int, int)] = [action for action, reward in eval_actions.items() if
                                                  reward == max(eval_actions.values())]
                # Get the new probability distribution based on the number of best actions.
                action_prob: float = 1 / len(best_actions)
                new_actions: dict[(int, int), float] = {}

                for action in best_actions:
                    new_actions[action] = action_prob

                self.movement[pos] = new_actions

    # Runs the TD Sarsa algorithm and returns a list of state values.
    def sarsa(self, alpha: int = 1, gamma: float = 0.9, epsilon: int = 0.05, num_episodes: int = 100):
        # Create a dictionary with all cells of the grid, having a value of 0.
        state_values = {state: 0 for state in self.grid.keys()}

        # Iterate for the number of episodes defined.
        for n in range(num_episodes):
            current_pos: (int, int) = self.grid_pos
            # Get an initial action for the episode using epsilon greedy.
            action = self.epsilon_greedy(state_values, current_pos, epsilon)

            # Iterate until a terminal state is found.
            while True:
                # Get the name state for the given action.
                state_prime: (int, int) = tuple_addition(current_pos, action)
                # Get the reward for the given action.
                reward: float = self.grid[state_prime].reward
                # Get a new action given the new state, using epsilon greedy.
                action_prime = self.epsilon_greedy(state_values, state_prime, epsilon)
                # Update the state values.
                state_values[current_pos] += alpha * (reward + gamma * state_values[state_prime] - state_values[current_pos])
                # Update the position and action for the next iteration.
                current_pos = state_prime
                action = action_prime

                # Break if the new state is terminal.
                if self.grid[current_pos].is_terminal:
                    break

        return state_values

    # Selects an optimal or random action based on epsilon greedy.
    def epsilon_greedy(self, state_values: dict[(int, int), float], pos: (int, int), epsilon: float = 0.3) -> (int, int):
        # Get a random probability between 0 and 1.
        p = random.uniform(0, 1)
        # If the probability is smaller than epsilon, take a random move.
        if p < epsilon:
            action: (int, int) = select_prob_move(pos, self.random_movement)
        else:
            eval_actions: dict[(int, int), float] = {}

            # Select the optimal move.
            for a in self.movement[pos]:
                # Create the new position based on the action and the current position.
                next_state: (int, int) = tuple_addition(pos, a)

                # If an action leads to a terminal state, return it as the optimal action.
                if self.grid[next_state].is_terminal:
                    return a

                next_reward: float = state_values[next_state]
                eval_actions[a] = next_reward

            # Get the best actions of the current state, based on the highest value associated.
            best_actions: list[(int, int)] = [action for action, reward in eval_actions.items() if
                                              reward == max(eval_actions.values())]
            # Choose a random action out of the best actions.
            action = random.choice(best_actions)
        return action

    # Runs TD(0) and returns a list of state values.
    def td_zero(self, alpha: int = 1, gamma: float = 0.8, num_episodes: int = 100):
        # Create a dictionary with all cells of the grid, having a value of 0.
        state_values = {state: 0 for state in self.grid.keys()}
        current_pos = self.grid_pos

        # Iterate based on the number of episodes.
        for n in range(num_episodes):
            episode: list[tuple[tuple[int, int], int], tuple[int, int]] = generate_episode(self, current_pos)

            # Iterate for each step taken in the episode.
            for step in episode:
                # Get the action from the step.
                a = step[1]
                # Get the new position based on the action.
                new_pos = tuple_addition(current_pos, a)
                # Get the reward based on the action.
                new_reward = self.grid[new_pos].reward

                # Update the state values.
                state_values[current_pos] += alpha * (new_reward + gamma * state_values[new_pos] - state_values[current_pos])
                current_pos = new_pos

            # Break if the state is terminal.
            if self.grid[current_pos].is_terminal:
                break

        return state_values

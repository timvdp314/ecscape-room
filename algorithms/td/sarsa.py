import random
from typing import Callable

from algorithms.algorithm import Algorithm, Observation
from algorithms.utils import generate_episode, sample_policy_action

class TDSarsaAlgorithm(Algorithm):
    def __init__(self, policy: dict[tuple[int, int], dict[tuple[int, int], float]], grid_action_cb: Callable[[tuple[int, int], tuple[int, int]], Observation], 
                 grid_pos: tuple[int, int], grid_size: int):

        super().__init__(policy, grid_action_cb, grid_pos, grid_size)

        self.q_table: dict[tuple[int, int], dict[tuple[int, int], float]] = dict()
        self.total_rewards: list[float] = list()
        self.total_state_visits: dict[tuple[int, int], int] = dict()
        self.total_state_visits_tracker: dict[tuple[int, int], bool] = dict()

        self.state_values: dict[tuple[int, int], float] = dict()

    def move(self, grid_pos: tuple[int, int] = None) -> tuple[int, int]:
        return sample_policy_action(grid_pos, self.policy)

    # Process the state values in order to update the movement strategy.
    def process_state_values(self, state_values: dict[(int, int), float]) -> None:
        # For each Cell in the grid update the movement strategy based on the state values.
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                eval_actions: dict[(int, int), float] = {}

                # Construct position based on the x and y coordinates.
                pos = (x, y)
                for a in self.policy[pos]:
                    # Create the new position based on the action and the current position.
                    next_state: Observation = self.grid_action_cb(pos, a)

                    # If a final state is among the actions, it is taken as the only actions.
                    if next_state.is_terminal:
                        eval_actions.clear()
                        eval_actions[a] = next_state.reward
                        break

                    next_reward: float = state_values[next_state.grid_pos]
                    eval_actions[a] = next_reward

                # Get the best actions of the current state, based on the highest value associated.
                best_actions: list[(int, int)] = [action for action, reward in eval_actions.items() if
                                                  reward == max(eval_actions.values())]
                # Get the new probability distribution based on the number of best actions.
                action_prob: float = 1 / len(best_actions)
                new_actions: dict[(int, int), float] = {}

                for action in best_actions:
                    new_actions[action] = action_prob

                self.policy[pos] = new_actions


    # Runs the TD Sarsa algorithm and returns a list of state values.
    def run(self, alpha_factor: float = 0.10, gamma_factor: float = 0.95, epsilon_factor: int = 0.05, num_episodes: int = 100):
        # Create a dictionary with all cells of the grid, having a value of 0.
        self.state_values: dict[tuple[int, int], float] = dict()

        self.total_rewards.clear()

        # Initialize value table to all 0.0
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                self.state_values[(x,y)] = 0.0
                self.total_state_visits[(x,y)] = 0

        # Iterate for the number of episodes defined.
        for n in range(num_episodes):
            total_reward: float = 0.0

            current_pos: Observation = self.grid_action_cb(self.grid_pos)

            # Get an initial action for the episode using epsilon greedy.
            action = self.epsilon_greedy(self.state_values, current_pos.grid_pos, epsilon_factor)

            # Purely for plotting, not functional
            self.total_state_visits_tracker.clear()

            # Iterate until a terminal state is found.
            while True:
                
                # Purely for plotting, not functional
                if (not current_pos.grid_pos in self.total_state_visits_tracker):
                    self.total_state_visits_tracker[current_pos.grid_pos] = True
                    self.total_state_visits[current_pos.grid_pos] += 1

                # Get the name state for the given action.
                state_prime: Observation = self.grid_action_cb(current_pos.grid_pos, action)
                # Get the reward for the given action.
                reward: float = state_prime.reward
                # Get a new action given the new state, using epsilon greedy.
                action_prime = self.epsilon_greedy(self.state_values, state_prime.grid_pos, epsilon_factor)
                # Update the state values.
                self.state_values[current_pos.grid_pos] += alpha_factor * (reward + gamma_factor * self.state_values[state_prime.grid_pos] - self.state_values[current_pos.grid_pos])
                # Update the position and action for the next iteration.
                current_pos = state_prime
                action = action_prime

                total_reward += state_prime.reward

                # Break if the new state is terminal.
                if state_prime.is_terminal:
                    break

            self.total_rewards.append(total_reward)

        self.process_state_values(self.state_values)

    # Selects an optimal or random action based on epsilon greedy.
    def epsilon_greedy(self, state_values: dict[(int, int), float], pos: tuple[int, int], epsilon: float = 0.3) -> tuple[int, int]:
        # Get a random probability between 0 and 1.
        p = random.uniform(0, 1)
        # If the probability is smaller than epsilon, take a random move.
        if p < epsilon:
            action: tuple[int, int] = sample_policy_action(self.policy, pos)
        else:
            eval_actions: dict[(int, int), float] = {}

            # Select the optimal move.
            for a in self.policy[pos]:
                # Create the new position based on the action and the current position.
                next_state: Observation = self.grid_action_cb(pos, a)

                # If an action leads to a terminal state, return it as the optimal action.
                if (next_state.is_terminal):
                    return a

                next_reward: float = state_values[next_state.grid_pos]
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
        state_values: dict[tuple[int, int], float] = dict()

        # Initialize value table to all 0.0
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                state_values[(x,y)] = 0.0
                self.total_state_visits[(x,y)] = 0

                for action in self.policy[(x,y)].keys():
                    self.q_table[(x,y)][action] = 0.0

        current_pos = self.grid_pos

        # Iterate based on the number of episodes.
        for n in range(num_episodes):
            episode: list[tuple[tuple[int, int], int], tuple[int, int]] = generate_episode(self, current_pos)

            # Iterate for each step taken in the episode.
            for step in episode:
                # Get the action from the step.
                a = step[1]
                # Get the new position based on the action.
                new_pos: Observation = self.grid_action_cb(current_pos, a)
                # Get the reward based on the action.
                new_reward = new_pos.reward

                # Update the state values.
                state_values[current_pos] += alpha * (new_reward + gamma * state_values[new_pos.grid_pos] - state_values[current_pos])
                current_pos = new_pos.grid_pos

            # Break if the state is terminal.
            if new_pos.is_terminal:
                break

        return state_values
    
    def terminate(self):
        pass

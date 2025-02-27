import random
from typing import Callable
import logging

from algorithms.algorithm import Algorithm, Observation
from algorithms.utils import sample_policy_action

class QLearningAlgorithm(Algorithm):
    def __init__(self, policy: dict[tuple[int, int], dict[tuple[int, int], float]], grid_action_cb: Callable[[tuple[int, int], tuple[int, int]], Observation], 
                 grid_pos: tuple[int, int], grid_size: int):
        super().__init__(policy, grid_action_cb, grid_pos, grid_size)

        self.q_table: dict[tuple[int, int], dict[tuple[int, int], float]] = dict()

    def run(self, alpha_factor: float = 0.10, gamma_factor: float = 0.95, num_episodes: int = 200):
        episode: list[tuple[tuple[int, int], int]] = list()

        total_rewards: list[float] = list()

        for x in range(self.grid_size):
            for y in range(self.grid_size):
                self.q_table[(x,y)] = dict()

                for action in self.policy[(x,y)].keys():
                    self.q_table[(x,y)][action] = 0.0

        for n in range(num_episodes):
            total_reward: float = 0.0
            episode.clear()

            epsilon_factor: float = 1.0 / (n + 1)
            curr_state = self.grid_action_cb(self.grid_pos)

            while (True):
                if (curr_state.is_terminal):
                    break

                s: tuple[int, int] = curr_state.grid_pos

                action: tuple[int, int] = self.get_action(s, epsilon_factor)
                next_state: Observation = self.grid_action_cb(s, action)
                total_reward += next_state.reward

                best_next_action: tuple[int, int] = max(self.q_table[next_state.grid_pos], key = self.q_table[next_state.grid_pos].get)
                target = next_state.reward + (gamma_factor * self.q_table[next_state.grid_pos][best_next_action] * (not next_state.is_terminal))
                self.q_table[s][action] += alpha_factor * (target - self.q_table[s][action])

                curr_state = next_state
            
            total_rewards.append(total_reward)

        logging.info(total_rewards)
        self.get_best_policy()

    def get_action(self, state: tuple[int, int], epsilon_factor: float):
        if (random.uniform(0, 1) < epsilon_factor):
            return sample_policy_action(self.policy, state)
        else:
            return max(self.q_table[state], key = self.q_table[state].get)
        
    def get_best_policy(self):
        for state in self.q_table.keys():
            best_action = max(self.q_table[state], key = self.q_table[state].get)
            self.policy[state].clear()
            self.policy[state][best_action] = 1.0
    
    def terminate(self):
        pass
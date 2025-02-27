from typing import Callable
import random

from algorithms.dp.policy_iteration import PolicyIterationAlgorithm
from algorithms.dp.value_iteration import ValueIterationAlgorithm
from algorithms.algorithm import Algorithm
from algorithms.mc.mcc import MonteCarloAlgorithm
from algorithms.utils import Observation, sample_policy_action

class Agent:
    def __init__(self, grid_action_cb: Callable[[tuple[int, int], tuple[int, int]], Observation], 
                 grid_pos: tuple[int, int] = (0, 0), grid_size: int = 8, img: str = "robot.png"):
        self.img = img
        self.grid_pos: tuple[int, int] = grid_pos
        self.grid_size = grid_size
        self.grid_action_cb: Callable[[tuple[int, int], tuple[int, int]], Observation] = grid_action_cb

        self.policy: dict[tuple[int, int], dict[tuple[int, int], float]] = dict()
        # self.algorithm = MonteCarloAlgorithm(self.policy, self.grid_action_cb, self.grid_pos, self.grid_size)
        # self.algorithm = PolicyIterationAlgorithm(self.policy, self.grid_action_cb, self.grid_pos, self.grid_size)
        self.algorithm = ValueIterationAlgorithm(self.policy, self.grid_action_cb, self.grid_pos, self.grid_size)

        self.init_policy()

    def init_policy(self):
        for x in range(0, self.grid_size):
            for y in range(0, self.grid_size):
                self.policy[(x, y)] = dict()
                if x > 0:
                    self.add_policy_action((x, y), (-1, 0))
                if x < self.grid_size - 1:
                    self.add_policy_action((x, y), (1, 0))
                if y > 0:
                    self.add_policy_action((x, y), (0, -1))
                if y < self.grid_size - 1:
                    self.add_policy_action((x, y), (0, 1))

                probability: float = 1 / len(self.policy[(x, y)])
                for key in self.policy[(x, y)]:
                    self.policy[(x, y)][key] = probability

    def add_policy_action(self, state: tuple[int, int], action: tuple[int, int]) -> tuple[int, int]:
        obs: Observation = self.grid_action_cb(state, action)

        if (state != obs.grid_pos):
            self.policy[state][action] = 0
    
    def sample_action(self, grid_pos: tuple[int, int] = None) -> tuple[int, int]:
        return sample_policy_action(self.policy, self.grid_pos if grid_pos is None else grid_pos)

    def run_algorithm(self):
        self.algorithm.run()
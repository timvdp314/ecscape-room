from abc import ABC, abstractmethod
import random

from algorithms.algorithm import Policy
from cell import Cell

class QLearning(Policy, ABC):
    def __init__(self, grid_pos: tuple[int, int], grid: dict[tuple[int, int], Cell], grid_size: int = 6):
        super().__init__()
        self.grid_pos = grid_pos
        self.grid_size = grid_size
        self.movement: dict[tuple[int, int], dict[tuple[int, int], float]] = dict()
        self.grid: dict[tuple[int, int], Cell] = grid
        self.init_policy()

    def init_policy(self):
        for x in range(0, self.grid_size):
            for y in range(0, self.grid_size):
                self.movement[(x, y)] = dict()
                if x > 0:
                    self.add_action((x, y), (-1, 0))
                if x < self.grid_size - 1:
                    self.add_action((x, y), (1, 0))
                if y > 0:
                    self.add_action((x, y), (0, -1))
                if y < self.grid_size - 1:
                    self.add_action((x, y), (0, 1))

                probability: float = 1 / len(self.movement[(x, y)])
                for key in self.movement[(x, y)]:
                    self.movement[(x, y)][key] = probability


    def add_action(self, position, action):
        new_pos = (position[0] + action[0], position[1] + action[1])
        if not self.grid[new_pos].is_solid:
            self.movement[position][action] = 0
    
    def move(self, grid_pos: tuple[int, int] = None):
        pos = (self.grid_pos) if grid_pos is None else grid_pos

        num = random.uniform(0, 1)
        threshold = 0.0

        for act, prob in self.movement[pos].items():
            threshold += prob
            if num <= threshold:
                return act
    
        assert("Something is wrong with the agent's policy!")
        return None
    
    def terminate(self):
        pass
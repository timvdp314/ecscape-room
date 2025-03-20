from abc import ABC, abstractmethod
import random

from cell import Cell
from policies.utility import init_policy, select_prob_move


# The default policy is a random policy with uniform action probabilities
class Policy(ABC):
    def __init__(self, grid_pos: tuple[int, int], grid: dict[tuple[int, int], Cell], grid_size: int = 6):
        super().__init__()
        self.grid_pos = grid_pos
        self.grid_size = grid_size
        self.movement: dict[tuple[int, int], dict[tuple[int, int], float]] = dict()
        self.grid: dict[tuple[int, int], Cell] = grid
        init_policy(grid, grid_size, self.movement)
    
    def move(self, grid_pos: tuple[int, int] = None):
        return select_prob_move(self.grid_pos, self.movement)
    
    def terminate(self):
        pass
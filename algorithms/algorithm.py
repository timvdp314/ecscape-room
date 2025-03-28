import random
from typing import Callable

from abc import ABC, abstractmethod

from algorithms.utils import Observation
from cell import Cell

class Algorithm(ABC):
    def __init__(self, policy: dict[tuple[int, int], dict[tuple[int, int], float]], grid_action_cb: Callable[[tuple[int, int], tuple[int, int]], Observation], 
                 grid_pos: tuple[int, int], grid_size: int):
        self.policy: dict[tuple[int, int], dict[tuple[int, int], float]] = policy
        self.grid_action_cb: Callable[[tuple[int, int], tuple[int, int]], Observation] = grid_action_cb
        self.grid_pos: tuple[int, int] = grid_pos
        self.grid_size: int = grid_size

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def terminate(self):
        pass
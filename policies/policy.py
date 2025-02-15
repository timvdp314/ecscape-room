from abc import ABC, abstractmethod

class Policy(ABC):
    def __init__(self, grid_pos: (int, int), grid: dict[tuple[int, int]], grid_size: int = 6):
        super().__init__()
        self.grid_pos = grid_pos
        self.grid_size = grid_size
        self.movement: dict[tuple[int, int]] = dict()
        self.grid: dict[tuple[int, int]] = grid
        self.init_policy()

    def init_policy(self):
        for x in range(0, self.grid_size):
            for y in range(0, self.grid_size):
                self.movement[(x, y)] = dict()
                if x > 0:
                    self.movement[(x, y)][(-1, 0)] = 0.25
                if x < self.grid_size - 1:
                    self.movement[(x, y)][(1, 0)] = 0.25
                if y > 0:
                    self.movement[(x, y)][(0, -1)] = 0.25
                if y < self.grid_size - 1:
                    self.movement[(x, y)][(0, 1)] = 0.25
    
    @abstractmethod
    def move(self):
        pass
    
    @abstractmethod
    def terminate(self):
        pass
from abc import ABC, abstractmethod

class Policy(ABC):
    def __init__(self, grid_pos, grid_size = 6):
        super().__init__()
        self.grid_pos = grid_pos
        self.grid_size = grid_size
        self.movement: dict[tuple[int, int]] = dict()
        self.init_policy()

    def init_policy(self):
        for x in range(0, self.grid_size):
            for y in range(0, self.grid_size):
                self.movement[(x, y)] = dict()
                self.movement[(x, y)][(-1, 0)] = 0.25
                self.movement[(x, y)][(1, 0)] = 0.25
                self.movement[(x, y)][(0, -1)] = 0.25
                self.movement[(x, y)][(0, 1)] = 0.25
    
    @abstractmethod
    def move(self):
        pass
    
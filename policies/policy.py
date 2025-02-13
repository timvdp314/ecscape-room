from abc import ABC, abstractmethod

class Policy(ABC):
    def __init__(self, grid_size = 6):
        super().__init__()
        self.grid_size = grid_size
        self.movement: dict[tuple[int, int]] = dict()
        self.init_policy()

    def init_policy(self):
        for x in range(0, self.grid_size):
            for y in range(0, self.grid_size):
                self.policy[(x, y)] = dict()
                self.policy[(x, y)][(-1, 0)] = 0.25
                self.policy[(x, y)][(1, 0)] = 0.25
                self.policy[(x, y)][(0, -1)] = 0.25
                self.policy[(x, y)][(0, 1)] = 0.25
    
    @abstractmethod
    def movement(self):
        pass
    
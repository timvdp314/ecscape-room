class Cell:
    def __init__(self, grid_pos, reward, img = None, is_terminal = False, is_solid = False):
        self.grid_pos = grid_pos
        self.reward = reward
        self.img = img
        self.is_terminal = is_terminal 
        self.is_solid = is_solid
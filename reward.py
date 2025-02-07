class RewardObject:
    def __init__(self, grid_pos, reward, img, is_terminal = False):
        self.grid_pos = grid_pos
        self.reward = reward
        self.img = img
        self.is_terminal = is_terminal

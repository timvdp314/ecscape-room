from policies.policy import Policy
from policies.random_policy import RandomPolicy

class Agent:
    def __init__(self, grid_pos = (0, 0), grid_size = 5, img = "robot.png"):
        self.img = img

        self.policy: Policy = RandomPolicy(grid_pos, grid_size)
        self.last_reward_val = 0

    def action(self) -> (int, int):
        return self.policy.move()

    def set_position(self, grid_pos: (int, int)):
        self.policy.grid_pos = grid_pos

    def get_position(self) -> (int, int):
        return self.policy.grid_pos

    def set_last_reward(self, reward_val : int):
        self.last_reward_val = reward_val

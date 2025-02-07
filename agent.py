from enum import Enum
import random

class Agent:
    def __init__(self, grid_pos = (0, 0), grid_size = 5, img = "robot.png"):
        self.grid_pos = grid_pos
        self.grid_size = grid_size
        self.img = img

        self.policy: dict[tuple[int, int]] = dict()
        self.last_reward_val = 0

    def reset_policy(self):
        for x in range(0, self.grid_size):
            for y in range(0, self.grid_size):
                self.policy[(x, y)] = dict()
                self.policy[(x, y)][(-1, 0)] = 0.25
                self.policy[(x, y)][(1, 0)] = 0.25
                self.policy[(x, y)][(0, -1)] = 0.25
                self.policy[(x, y)][(0, 1)] = 0.25

    def sample_action(self):
        for n in range(5):
            num = random.uniform(0, 1)
            threshold = 0.0

            for act, prob in self.policy[self.grid_pos].items():
                if (num <= threshold + prob):
                    return act
                
                threshold += prob

        assert("Something is wrong with the agent's policy!")
        return None

    def set_last_reward(self, reward_val : int):
        self.last_reward_val = reward_val

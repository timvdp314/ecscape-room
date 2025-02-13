from abc import ABC

from policies.policy import Policy
import random

class RandomPolicy(Policy, ABC):
    def move(self):
        num = random.uniform(0, 1)
        threshold = 0.0

        for act, prob in self.movement[self.grid_pos].items():
            threshold += prob
            if (num <= threshold):
                return act
    
        assert("Something is wrong with the agent's policy!")
        return None
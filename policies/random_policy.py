from policy import Policy
import random

class RandomPolicy(Policy):
    def movement(self):
        num = random.uniform(0, 1)
        threshold = 0.0

        for act, prob in self.policy[self.grid_pos].items():
            threshold += prob
            if (num <= threshold):
                return act
    
        assert("Something is wrong with the agent's policy!")
        return None
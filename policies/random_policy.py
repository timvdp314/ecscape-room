from abc import ABC

from policies.policy import Policy
import random

from policies.utility import evaluate_policy


class RandomPolicy(Policy, ABC):
    def move(self):
        possible_rewards = evaluate_policy(self)

        num = random.uniform(0, 1)
        threshold = 0.0

        for act, prob in self.movement[self.grid_pos].items():
            threshold += prob
            if num <= threshold:
                return act
    
        assert("Something is wrong with the agent's policy!")
        return None
    
    def terminate(self):
        return super().terminate()
    

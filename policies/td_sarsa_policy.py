from abc import ABC

from policies.policy import Policy


class TDSarsaPolicy(Policy, ABC):

    def td_zero(self, step_size: int = 1):
        state_values = {state: 0 for state in self.grid.keys()}
        current_pos = self.grid_pos

        while True:
            for action in self.movement[current_pos]:


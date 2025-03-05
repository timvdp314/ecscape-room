from policies.policy_iteration_policy import PolicyIterationPolicy
from policies.value_iteration_policy import ValueIterationPolicy
from policies.policy import Policy
from policies.mcp_policy import MonteCarloPredictionPolicy
from policies.mcc_policy import MonteCarloControlPolicy

class Agent:
    def __init__(self, grid: dict[tuple[int, int]], grid_pos: tuple[int, int] = (0, 0), grid_size: int = 5, img: str = "robot.png"):
        self.img = img
        self.grid_pos = grid_pos

        self.policy: Policy = MonteCarloControlPolicy(grid_pos, grid, grid_size)
        self.last_reward_val = 0

    def action(self) -> tuple[int, int]:
        return self.policy.move()

    def set_position(self, grid_pos: tuple[int, int]):
        self.grid_pos = grid_pos
        self.policy.grid_pos = grid_pos

    def get_position(self) -> tuple[int, int]:
        return self.grid_pos

    def set_last_reward(self, reward_val : int):
        self.last_reward_val = reward_val

import gymnasium as gym
import numpy as np
from cell import Cell
import math_utils

from algorithms.utils import Observation

class SchoolEnv(gym.Env):
    def __init__(self, agent_location: tuple[int, int], target: Cell, grid_size = 5):
        super(SchoolEnv, self).__init__()

        self.grid_size = grid_size  # The size of the square grid

        self.agent_location: tuple[int, int] = agent_location
        self.agent_reset_location: tuple[int, int] = agent_location

        # Locations of the solids, primarily for visualization.
        self.solids: list[tuple[int, int]] = []

        # Target location for constant lookup times.
        self.target: Cell = target

        # Reward locations are stored for possible moving rewards.
        self.reward_locations: dict[tuple[int, int]] = dict()
        self.reward_reset_locations: dict[tuple[int, int]] = dict()

        # Base reward for moving to an empty Cell in the grid.
        self.base_reward: float = -1.0

        # Dictionary of the grid mapping coordinates to Cells.
        self.grid: dict[tuple[int, int], Cell] = {}
        self.init_grid() # Creates a grid with empty Cells.

    def register_object(self, cell: Cell) -> None:
        if cell.is_solid:
            # Register the grid position of a solid for visualization.
            self.solids.append(cell.grid_pos)
        else:
            # If not a solid, it must be a reward.
            self.reward_locations[cell.grid_pos] = cell.reward
            self.reward_reset_locations[cell.grid_pos] = cell.reward

        # Register the cell at the given cell position.
        self.grid[cell.grid_pos] = cell

    def init_grid(self) -> None:
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                self.grid[(x, y)] = Cell((x, y), self.base_reward)

        self.grid[self.target.grid_pos] = self.target

    def reset(self, seed = None, options = None):
        """Reset the environment to an initial state."""
        self.agent_location = self.agent_reset_location
        self.reward_locations = self.reward_reset_locations

        return self.get_obs()
    
    def step(self, action: tuple[int, int], state: tuple[int, int] = None) -> Observation:
        if (state is None):
            backup_state = self.agent_location
        else:
            backup_state = state
        
        backup_state = (self.agent_location) if state is None else state
        temp_state = (self.agent_location) if state is None else state
        temp_state = np.array(temp_state) + np.array(action)
        temp_state = np.clip(temp_state, 0, self.grid_size - 1)
        temp_state = math_utils.np_to_tuple(temp_state)

        cell: Cell = self.grid[temp_state]

        if cell.is_solid:
            temp_state = backup_state
            cell: Cell = self.grid[temp_state]

        obs = Observation(temp_state, cell.reward, cell.is_terminal)

        if (state is None):
            self.agent_location = temp_state

        return obs

    def get_obs(self, state: tuple[int, int] = None, action: tuple[int, int] = None) -> Observation:
        if (action is None):
            return Observation(self.agent_location if state is None else state, self.grid[self.agent_location].reward, self.grid[self.agent_location].is_terminal)
        else:
            return self.step(action, state)

    def close(self):
        """Clean up resources (optional)."""
        pass
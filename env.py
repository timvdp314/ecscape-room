import gymnasium as gym
import numpy as np
from cell import Cell

class SchoolEnv(gym.Env):
    def __init__(self, agent_location: (int, int), target: Cell, grid_size = 5):
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
        self.base_reward: float = -1

        # Dictionary of the grid mapping coordinates to Cells.
        self.grid: dict[tuple[int, int]] = {}
        self.init_grid() # Creates a grid with empty Cells.


    def get_obs(self):
        obs_dict = dict()

        obs_dict["agent"] = None if self.agent_location is None else self.agent_location
        obs_dict["target"] = None if self.target.grid_pos is None else self.target.grid_pos
        obs_dict["rewards"] = None if len(self.reward_locations) == 0 else [rw for rw in self.reward_locations]

        return obs_dict

    def get_info(self):
        return {
            "distance": np.linalg.norm(
                np.array(self.agent_location) - np.array(self.target.grid_pos), ord=1
            )
        }

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

        observation = self.get_obs()
        info = self.get_info()

        return observation, info
    
    def step(self, action):
        temp_pos = self.agent_location
        self.agent_location = np.array(self.agent_location) + np.array(action)
        self.agent_location = np.clip(self.agent_location, 0, self.grid_size - 1)
        self.agent_location = tuple(self.agent_location)

        cell: Cell = self.grid[self.agent_location]

        if cell.is_solid:
            self.agent_location = temp_pos
            cell: Cell = self.grid[self.agent_location]

        observation = self.get_obs()
        info = self.get_info()

        return observation, cell.reward, cell.is_terminal, info

    def close(self):
        """Clean up resources (optional)."""
        pass
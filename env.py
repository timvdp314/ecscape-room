import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame

from agent import Agent
from reward import RewardObject

from enum import Enum

class SchoolEnv(gym.Env):
    def __init__(self, grid_size = 5):
        super(SchoolEnv, self).__init__()

        self.grid_size = grid_size  # The size of the square grid
        self.agent_location: tuple[int, int] = None
        self.agent_reset_location: tuple[int, int] = None

        self.target_location: tuple[int, int] = None
        self.target_reset_location: tuple[int, int] = None
        self.target_reward: int = 0
        self.reward_locations: dict[tuple[int, int]] = dict()
        self.reward_reset_locations: dict[tuple[int, int]] = dict()
        self.solids: dict[tuple[int, int]] = dict()
        self.base_reward = -1

        self.grid: dict[tuple] = dict()


    def get_obs(self):
        obs_dict = dict()

        obs_dict["agent"] = None if self.agent_location is None else self.agent_location
        obs_dict["target"] = None if self.target_location is None else self.target_location
        obs_dict["rewards"] = None if len(self.reward_locations) == 0 else [rw for rw in self.reward_locations]

        return obs_dict

    def get_info(self):
        return {
            "distance": np.linalg.norm(
                np.array(self.agent_location) - np.array(self.target_location), ord=1
            )
        }

    def register_agent(self, agent_pos: tuple[int, int]):
        self.agent_location = agent_pos
        self.agent_reset_location = agent_pos

    def register_target(self, target_pos: tuple[int, int], target_reward):
        self.target_location = target_pos
        self.target_reset_location = target_pos
        self.target_reward = target_reward

    def register_reward(self, reward_location, reward_value):
        self.reward_locations[reward_location] = reward_value
        self.reward_reset_locations[reward_location] = reward_value

    def register_solid(self, grid_pos):
        self.solids[grid_pos] = True

    def grid_clear(self):
        self.grid = dict()
        self.solids = dict()

    def reset(self, seed = None, options = None):
        """Reset the environment to an initial state."""
        self.agent_location = self.agent_reset_location
        self.target_location = self.target_reset_location
        self.reward_locations = self.reward_reset_locations

        observation = self.get_obs()
        info = self.get_info()

        return observation, info
    
    def step(self, action):
        temp_pos = self.agent_location
        self.agent_location = np.array(self.agent_location) + np.array(action)
        self.agent_location = np.clip(self.agent_location, 0, self.grid_size - 1)
        self.agent_location = tuple(self.agent_location)

        if (not self.solids.get(self.agent_location) is None):
            self.agent_location = temp_pos

        reward = self.base_reward
        terminal = False

        if (self.agent_location == self.target_location):
            terminal = True
            reward = self.target_reward

        for reward_pos, reward_val in self.reward_locations.items():
            if (self.agent_location == reward_pos):
                reward = reward_val
                break

        observation = self.get_obs()
        info = self.get_info()

        return observation, reward, terminal, info

    def close(self):
        """Clean up resources (optional)."""
        pass
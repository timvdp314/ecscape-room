import logging

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from env import SchoolEnv
from reward import RewardObject
from agent import Agent

import pygame

from gymnasium.envs.registration import register

from disp import EnvRenderer

# Configure the logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

if __name__ == "__main__":
    register(
    id="SchoolEnv-v0",
    entry_point="env:SchoolEnv",
    )

    pygame.init()
    pygame.display.init()

    window_size = 512
    grid_size = 8
    window = pygame.display.set_mode(
            (window_size, window_size)
        )
    clock = pygame.time.Clock()

    renderer = EnvRenderer(window, window_size, grid_size)
    school_env: SchoolEnv = SchoolEnv(grid_size)
    pill = RewardObject((1, 3), 2, "pill.png", False)
    pill2 = RewardObject((3, 4), 2, "pill.png", False)
    agent = Agent((2, 2), grid_size = grid_size)
    agent.reset_policy()

    target = RewardObject((4,5), 10, "exam.png", True)

    school_env.register_agent(agent.grid_pos)
    school_env.register_target(target.grid_pos, 5)

    school_env.register_solid((1,1))
    school_env.register_solid((2,1))
    school_env.register_solid((3,1))

    school_env.register_reward(pill.grid_pos, -2)
    school_env.register_reward(pill2.grid_pos, -4)

    running = True

    obs, info = school_env.reset()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    action = agent.sample_action()
                    obs, reward, done, info = school_env.step(action)
                    if done:
                        running = False

                    agent.grid_pos = obs["agent"]

                    logging.info("Obtained reward: {}".format(reward))

                if event.key == pygame.K_r:
                    obs, info = school_env.reset()

                    agent.grid_pos = obs["agent"]

        
        renderer.clear_frame()
        renderer.draw_gridlines()

        for val, key in school_env.solids.items():
            renderer.draw_solid_square(val, (0,0,0))

        renderer.draw_object(agent.grid_pos, agent.img)
        renderer.draw_object(pill.grid_pos, pill.img)
        renderer.draw_object(pill2.grid_pos, pill.img)
        renderer.draw_object(target.grid_pos, target.img)

        renderer.render_frame()
        clock.tick(60)

    school_env.close()

import logging

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from cell import Cell
from env import SchoolEnv
from reward import RewardObject
from agent import Agent
import time
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
    agent_pos = (2, 2)
    target = Cell((4,5), 10, "exam.png", True)

    school_env: SchoolEnv = SchoolEnv(agent_pos, target, grid_size)
    agent = Agent(school_env.grid, agent_pos, grid_size)

    pill1 = Cell((1, 3), -2, "pill.png")
    pill2 = Cell((3, 4), -2, "pill.png")

    school_env.register_object(Cell((1,1), 0, None, False, True))
    school_env.register_object(Cell((2,1), 0, None, False, True))
    school_env.register_object(Cell((3,1), 0, None, False, True))

    school_env.register_object(pill1)
    school_env.register_object(pill2)

    running = True

    obs, info = school_env.reset()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                match event.key:
                    case pygame.K_SPACE:
                        action = agent.action()
                    case pygame.K_LEFT:
                        action = (-1, 0)
                    case pygame.K_RIGHT:
                        action = (1, 0)
                    case pygame.K_UP:
                        action = (0, -1)
                    case pygame.K_DOWN:
                        action = (0, 1)
                    case default:
                        action = (0, 0)

                obs, reward, done, info = school_env.step(action)
                if done:
                    running = False

                logging.info("Obtained reward: {}".format(reward))
                agent.set_position(obs["agent"])

                if event.key == pygame.K_r:
                    obs, info = school_env.reset()

                    agent.grid_pos = obs["agent"]
          
        renderer.clear_frame()
        renderer.draw_gridlines()

        for solid in school_env.solids:
            renderer.draw_solid_square(solid, (0,0,0))

        renderer.draw_object(agent.get_position(), agent.img)
        renderer.draw_object(pill1.grid_pos, pill1.img)
        renderer.draw_object(pill2.grid_pos, pill2.img)
        renderer.draw_object(target.grid_pos, target.img)

        renderer.render_frame()
        clock.tick(60)

        action = agent.action()
        obs, reward, done, info = school_env.step(action)
        if done:
            running = False

        agent.set_position(obs["agent"])

        logging.info("Obtained reward: {}".format(reward))
        time.sleep(1)

    school_env.close()

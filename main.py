import logging

from cell import Cell
from env import SchoolEnv
from agent import Agent
import pygame

from gymnasium.envs.registration import register
import gymnasium as gym

from disp import EnvRenderer

# Configure the logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s: [%(levelname)s] %(message)s')

def sim_step(env: gym.Env, agent: Agent, action: tuple[int, int] = None):
    real_action = action

    if (action is None):
        real_action = agent.action()

    obs, reward, done, info = env.step(real_action)

    step_str: dict[tuple[int, int], str] = {
        (-1, 0): "LEFT",
        (1, 0): "RIGHT",
        (0, -1): "UP",
        (0, 1): "DOWN"
    }

    agent.set_position(obs["agent"])

    logging.info("[SIMULATION STEP, {}] Obtained reward: {}".format(step_str[real_action], reward))

    return reward, done, info


def sim_reset(env: gym.Env, agent: Agent):
    obs, info = env.reset()
    agent.set_position(obs["agent"])

    logging.info("[SIMULATION RESET]")

    return info


def sim_render(renderer: EnvRenderer, objects: dict[str, Cell], agent: Agent):
    renderer.clear_frame()
    renderer.draw_gridlines()

    for key, cell in env_objects.items():
        if (cell.is_solid):
            renderer.draw_solid_square(cell.grid_pos, (0,0,0))
        else:
            renderer.draw_object(cell.grid_pos, cell.img)

    renderer.draw_object(agent.get_position(), agent.img)
    renderer.render_frame()


if __name__ == "__main__":
    register(
    id="SchoolEnv-v0",
    entry_point="env:SchoolEnv",
    )

    pygame.display.init()

    window_size = 512
    grid_size = 8
    window = pygame.display.set_mode(
            (window_size, window_size)
        )
    clock = pygame.time.Clock()

    renderer = EnvRenderer(window, window_size, grid_size)
    agent_pos = (2, 2)

    env_objects: dict[str, Cell] = dict()
    env_objects["target"] = Cell((7,7), 15.0, "exam.png", True)
    env_objects["pill1"] = Cell((3, 2), -15.0, "pill.png")
    env_objects["pill2"] = Cell((0, 1), -8.0, "pill.png")
    env_objects["pill4"] = Cell((1, 5), -8.0, "pill.png")
    env_objects["pill5"] = Cell((3, 3), -8.0, "pill.png")
    env_objects["pill6"] = Cell((3, 7), -8.0, "pill.png")
    env_objects["solid1"] = Cell((1,1), 0.0, None, False, True)
    env_objects["solid2"] = Cell((2,1), 0.0, None, False, True)
    env_objects["solid3"] = Cell((3,1), 0.0, None, False, True)

    school_env: SchoolEnv = SchoolEnv(agent_pos, env_objects["target"], grid_size)

    for key, cell in env_objects.items():
        school_env.register_object(cell)

    # The agent needs to be created AFTER all of the objects have been added to the environment.
    agent = Agent(school_env.grid, agent_pos, grid_size)
    running = True

    obs, info = school_env.reset()

    fps = 60
    auto_run = False
    auto_run_timer = 0
    auto_run_max_time = fps

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if (auto_run):
                    match event.key:
                        case pygame.K_p:
                            auto_run = not auto_run
                        case pygame.K_r:
                            info = sim_reset(school_env, agent)
                        case pygame.K_EQUALS:
                            auto_run_max_time = max(auto_run_max_time - 5, 1)
                            logging.info("Simulation speed: {:.2f} steps per second".format(fps / auto_run_max_time))
                        case pygame.K_MINUS:
                            auto_run_max_time = min(auto_run_max_time + 5, 100)
                            logging.info("Simulation speed: {:.2f} steps per second".format(fps / auto_run_max_time))
                else:
                    match event.key:
                        case pygame.K_p:
                            auto_run = not auto_run
                            continue
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
                        case pygame.K_r:
                            info = sim_reset(school_env, agent)
                            action = None
                        case default:
                            action = None

                    if (action is None):
                        continue

                    reward, done, info = sim_step(school_env, agent, action)

                    if done:
                        running = False
        
        sim_render(renderer, env_objects, agent)

        if (auto_run):
            auto_run_timer += 1

            if (auto_run_timer >= auto_run_max_time):
                reward, done, info = sim_step(school_env, agent)

                if done:
                    running = False

                auto_run_timer = 0

        clock.tick(fps)

    school_env.close()

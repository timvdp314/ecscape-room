import logging
import sys

from cell import Cell
from env import SchoolEnv
from agent import Agent, AgentAlgorithm
import pygame

from gymnasium.envs.registration import register
import gymnasium as gym

import plot
from disp import EnvRenderer
from algorithms.utils import Observation

# Configure the logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s: [%(levelname)s] %(message)s')

def sim_step(env: gym.Env, agent: Agent, action: tuple[int, int] = None):
    real_action = action

    if (action is None):
        real_action = agent.sample_action()

    obs: Observation = env.step(real_action)

    step_str: dict[tuple[int, int], str] = {
        (-1, 0): "LEFT",
        (1, 0): "RIGHT",
        (0, -1): "UP",
        (0, 1): "DOWN"
    }

    agent.grid_pos = obs.grid_pos

    logging.info("[SIMULATION STEP, {}] Obtained reward: {}".format(step_str[real_action], obs.reward))

    return obs.reward, obs.is_terminal


def sim_reset(env: gym.Env, agent: Agent):
    obs: Observation = env.reset()
    agent.grid_pos = obs.grid_pos

    logging.info("[SIMULATION RESET]")

def sim_render(renderer: EnvRenderer, objects: dict[str, Cell], agent: Agent):
    renderer.clear_frame()
    renderer.draw_gridlines()

    for key, cell in env_objects.items():
        if (cell.is_solid):
            renderer.draw_solid_square(cell.grid_pos, (0,0,0))
        else:
            renderer.draw_object(cell.grid_pos, cell.img)

    renderer.draw_object(agent.grid_pos, agent.img, (2, 2))
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
    env_objects["pill1"] = Cell((4, 1), -4.0, "virus.png")
    env_objects["pill2"] = Cell((1, 5), -4.0, "virus.png")
    env_objects["pill3"] = Cell((3, 3), -4.0, "virus.png")
    env_objects["pill4"] = Cell((5, 5), -4.0, "virus.png")
    env_objects["pill5"] = Cell((3, 7), -4.0, "virus.png")
    env_objects["pill6"] = Cell((6, 6), -4.0, "virus.png")
    env_objects["pill7"] = Cell((0, 0), -4.0, "virus.png")
    env_objects["solid1"] = Cell((1,1), 0.0, None, False, True)
    env_objects["solid2"] = Cell((2,1), 0.0, None, False, True)
    env_objects["solid3"] = Cell((3,1), 0.0, None, False, True)

    school_env: SchoolEnv = SchoolEnv(agent_pos, env_objects["target"], grid_size)

    for key, cell in env_objects.items():
        school_env.register_object(cell)

    obs: Observation = school_env.reset()

    agent = Agent(school_env.get_obs, agent_pos, grid_size)
    rewards: dict = dict()

    # Default hyperparameters
    alpha_factor = 0.10
    gamma_factor = 0.90
    num_episodes = 200

    agent.init_policy()

    if (len(sys.argv) < 2):
        print("Error: no algorithm given\n\t- Usage: python main.py <algorithm> (PI, VI, MC, SARSA, QLEARNING, CUMU_REWARDS)")
        sys.exit()

    # Parse input parameter
    if (sys.argv[1] == "PI"):
        agent.set_algorithm(AgentAlgorithm.POLICY_ITERATION)
        agent.run_algorithm(gamma_factor = gamma_factor, theta_factor = 0.001)
        plot.plot_pi_heatmap(grid_size, agent.policy, agent.algorithm.potential_rewards)
    elif (sys.argv[1] == "VI"):
        agent.set_algorithm(AgentAlgorithm.VALUE_ITERATION)
        agent.run_algorithm(gamma_factor = gamma_factor, theta_factor = 0.001)
        plot.plot_vi_heatmap(grid_size, agent.algorithm.optimal_actions, agent.algorithm.potential_rewards)
    elif (sys.argv[1] == "MC"):
        agent.set_algorithm(AgentAlgorithm.MONTE_CARLO)
        agent.run_algorithm(gamma_factor = gamma_factor, num_episodes = num_episodes, epsilon_mod = 0.05)
        rewards["Monte Carlo"] = agent.algorithm.total_rewards
        plot.plot_state_visits(agent.algorithm.total_state_visits, grid_size, num_episodes, "Monte Carlo total state visits (normalized)")
    elif (sys.argv[1] == "SARSA"):
        agent.set_algorithm(AgentAlgorithm.SARSA)
        agent.run_algorithm(alpha_factor = alpha_factor, gamma_factor = gamma_factor, epsilon_factor = 0.05, num_episodes = num_episodes)
        rewards["SARSA"] = agent.algorithm.total_rewards
        plot.plot_sarsa_heatmap(grid_size, agent.policy, agent.algorithm.state_values)
        plot.plot_state_visits(agent.algorithm.total_state_visits, grid_size, num_episodes, "SARSA total state visits (normalized)")
    elif (sys.argv[1] == "QLEARNING"):
        agent.set_algorithm(AgentAlgorithm.Q_LEARNING)
        agent.run_algorithm(alpha_factor = alpha_factor, gamma_factor = gamma_factor, num_episodes = num_episodes)
        rewards["Q-Learning"] = agent.algorithm.total_rewards
        plot.plot_state_visits(agent.algorithm.total_state_visits, grid_size, num_episodes, "Q-Learning total state visits (normalized)")
    elif (sys.argv[1] == "CUMU_REWARDS"):
        agent.set_algorithm(AgentAlgorithm.MONTE_CARLO)
        agent.run_algorithm(gamma_factor = gamma_factor, num_episodes = num_episodes, epsilon_mod = 0.05)
        rewards["Monte Carlo"] = agent.algorithm.total_rewards

        agent.init_policy()
        agent.set_algorithm(AgentAlgorithm.SARSA)
        agent.run_algorithm(alpha_factor = alpha_factor, gamma_factor = gamma_factor, epsilon_factor = 0.05, num_episodes = num_episodes)
        rewards["SARSA"] = agent.algorithm.total_rewards

        agent.init_policy()
        agent.set_algorithm(AgentAlgorithm.Q_LEARNING)
        agent.run_algorithm(alpha_factor = alpha_factor, gamma_factor = gamma_factor, num_episodes = num_episodes)
        rewards["Q-Learning"] = agent.algorithm.total_rewards

        plot.plot_total_rewards(rewards, "Monte-Carlo vs. Q-Learning vs. SARSA total cumulative rewards per episode", alpha_factor, gamma_factor, num_episodes)
    else:
        print("Error: incorrect parameter '{}'\n\t- Usage: python main.py <algorithm> (PI, VI, MC, SARSA, QLEARNING, CUMU_REWARDS)".format(sys.argv[1]))
        sys.exit()

    fps = 60
    auto_run = True
    auto_run_timer = 0
    auto_run_max_time = fps / 2 # Default speed is 0.5 sec/step

    running = True

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
                            sim_reset(school_env, agent)
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
                            action = agent.sample_action()
                        case pygame.K_LEFT:
                            action = (-1, 0)
                        case pygame.K_RIGHT:
                            action = (1, 0)
                        case pygame.K_UP:
                            action = (0, -1)
                        case pygame.K_DOWN:
                            action = (0, 1)
                        case pygame.K_r:
                            sim_reset(school_env, agent)
                            action = None
                        case default:
                            action = None

                    if (action is None):
                        continue

                    reward, done = sim_step(school_env, agent, action)

                    if done:
                        running = False
        
        sim_render(renderer, env_objects, agent)

        if (auto_run):
            auto_run_timer += 1

            if (auto_run_timer >= auto_run_max_time):
                reward, done = sim_step(school_env, agent)

                if done:
                    running = False

                auto_run_timer = 0

        clock.tick(fps)

    school_env.close()

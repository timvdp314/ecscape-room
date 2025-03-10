import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def plot_state_visits(state_visits, grid_size):
    visit_counts = np.zeros((grid_size, grid_size))
    for state, count in state_visits.items():
        visit_counts[state[1], state[0]] = count  # Flip (x, y) to (row, col)
    
    plt.figure(figsize=(6,6))
    plt.imshow(visit_counts, cmap='Blues', interpolation='nearest')
    for i in range(grid_size):
        for j in range(grid_size):
            plt.text(j, i, int(visit_counts[i, j]), ha='center', va='center', color='black')
    
    plt.title("State Visit Frequency")
    plt.colorbar()
    plt.show()

def plot_total_rewards(reward_per_episode):
    plt.figure(figsize=(10,5))
    plt.plot(reward_per_episode, label="Total Reward per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Total Reward Per Episode")
    plt.legend()
    plt.show()

def plot_final_policy(policy, grid_size):
    plt.figure(figsize=(6,6))
    for state, actions in policy.items():
        best_action = max(actions, key=actions.get)  # Get the action with the highest probability
        plt.arrow(state[0], state[1], best_action[0] * 0.3, best_action[1] * 0.3, 
                  head_width=0.2, head_length=0.2, fc='k', ec='k')
    
    plt.xlim(-0.5, grid_size-0.5)
    plt.ylim(-0.5, grid_size-0.5)
    plt.xticks(range(grid_size))
    plt.yticks(range(grid_size))
    plt.gca().invert_yaxis()
    plt.title("Final Policy Arrows")
    plt.grid()
    plt.show()

def run_mc_simulation(env, agent, episodes=500):
    state_visits = defaultdict(int)
    reward_per_episode = []
    final_policy = agent.policy.movement  

    for episode in range(episodes):
        state = env.agent_location
        total_reward = 0
        done = False
        
        while not done:
            action = agent.action()
            new_state = (state[0] + action[0], state[1] + action[1])
            
            if new_state in env.grid:
                reward = env.grid[new_state].reward
                state = new_state
            else:
                reward = -1  
            
            total_reward += reward
            state_visits[state] += 1
            done = env.grid[state].is_terminal
        
        reward_per_episode.append(total_reward)
        env.reset()
    
    plot_state_visits(state_visits, env.grid_size)
    plot_total_rewards(reward_per_episode)
    plot_final_policy(final_policy, env.grid_size)



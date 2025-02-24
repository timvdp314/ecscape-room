import numpy as np
import random
from env import SchoolEnv
from cell import Cell

class MonteCarloControl:
    def __init__(self, env, discount_factor=0.9, epsilon=0.1, episodes=1000):
        self.env = env  # The game environment
        self.discount_factor = discount_factor  # Discount factor for future rewards
        self.epsilon = epsilon  # Probability of exploring instead of exploiting
        self.episodes = episodes  # Number of episodes to train on
        
        self.Q = {}  # Action-value function dictionary
        self.returns = {}  # Stores returns for state-action pairs
        
        # Initialize Q-values for all state-action pairs
        for x in range(env.grid_size):
            for y in range(env.grid_size):
                self.Q[(x, y)] = {
                    (-1, 0): 0,  # Left
                    (1, 0): 0,   # Right
                    (0, -1): 0,  # Up
                    (0, 1): 0    # Down
                }
                self.returns[(x, y)] = {a: [] for a in self.Q[(x, y)]}  # Track returns

    def generate_episode(self):
        """Generates an episode using epsilon-greedy policy."""
        episode = []  # Stores (state, action, reward) tuples
        state = self.env.agent_location  # Start state
        self.env.reset()
        done = False
        
        while not done:
            if random.uniform(0, 1) < self.epsilon:
                action = random.choice(list(self.Q[state].keys()))  # Explore
            else:
                action = max(self.Q[state], key=self.Q[state].get)  # Exploit
            
            next_state, reward, done, _ = self.env.step(action)  # Take action
            episode.append((state, action, reward))  # Store transition
            state = next_state  # Move to next state
        
        return episode
    
    def update_Q_values(self, episode):
        """Updates Q-values using first-visit Monte Carlo method."""
        G = 0  # Initialize return
        visited_state_actions = set()  # Track visited state-action pairs
        
        for state, action, reward in reversed(episode):  # Iterate from end to start
            G = reward + self.discount_factor * G  # Compute return
            if (state, action) not in visited_state_actions:
                visited_state_actions.add((state, action))  # Mark as visited
                self.returns[state][action].append(G)  # Store return
                self.Q[state][action] = np.mean(self.returns[state][action])  # Update Q
    
    def train(self):
        """Trains the agent using Monte Carlo Control."""
        for episode in range(self.episodes):
            ep = self.generate_episode()  # Generate an episode
            self.update_Q_values(ep)  # Update Q-values
    
    def get_policy(self):
        """Returns the optimal policy based on trained Q-values."""
        policy = {}
        for state in self.Q:
            policy[state] = max(self.Q[state], key=self.Q[state].get)  # Best action
        return policy

# test
# if __name__ == "__main__":
#     agent_pos = (2, 2)
#     target = Cell((7,7), 10, "exam.png", True)
#     school_env = SchoolEnv(agent_pos, target, grid_size=8)
    
#     mc_agent = MonteCarloControl(school_env, episodes=5000)
#     mc_agent.train()
#     optimal_policy = mc_agent.get_policy()
    
#     for state, action in optimal_policy.items():
#         print(f"State {state} -> Best Action: {action}")

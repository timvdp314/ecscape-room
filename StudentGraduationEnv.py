import numpy as np
import gymnasium as gym
from gymnasium import spaces

class StudentGraduationEnv(gym.Env):
    """
    A stochastic reinforcement learning environment where a student aims to collect 60 ECTS to graduate.
    """
    def __init__(self):
        super(StudentGraduationEnv, self).__init__()
        
        # Define action space (Discrete actions)
        # 0: Study, 1: Take an Energy Drink, 2: Rest, 3: Skip Class
        self.action_space = spaces.Discrete(4)
        
        # Define observation space (State: [ECTS, Energy Level, Health])
        self.observation_space = spaces.Box(low=np.array([0, 0, 0]), 
                                            high=np.array([60, 10, 10]), 
                                            dtype=np.float32)
        
        self.state = None
        self.reset()
    
    def reset(self, seed=None, options=None):
        """ Reset the environment to an initial state """
        super().reset(seed=seed)
        self.ects = 0  # Credits earned
        self.energy = 5  # Energy level (0-10)
        self.health = 5  # Health level (0-10)
        
        self.state = np.array([self.ects, self.energy, self.health], dtype=np.float32)
        return self.state, {}
    
    def step(self, action):
        """ Apply an action with stochastic effects """
        reward = 0
        done = False
        
        probabilities = {
            0: 1.0,  
            1: 1.0,  
            2: 1.0,  
            3: 1.0   
        }
        
        if np.random.rand() < probabilities[action]:
            if action == 0:  # Study
                if self.energy > 1 and self.health > 1:
                    self.ects += 5
                    self.energy -= 2
                    self.health -= 1
                    reward = 5
                else:
                    reward = -2  # Penalize if low energy or health
            elif action == 1:  # Take an Energy Drink
                self.energy = min(10, self.energy + 3)
                reward = 1
            elif action == 2:  # Rest
                self.energy = min(10, self.energy + 2)
                self.health = min(10, self.health + 2)
                reward = 2
            elif action == 3:  # Skip Class
                reward = -5  # Penalize skipping class
        
        # Stochastic negative effects
        if np.random.rand() < 0.1:  # Alarm Clock fails, lose energy
            self.energy = max(0, self.energy - 2)
            reward -= 2
        
        if np.random.rand() < 0.05:  # Angry Teacher
            reward -= 3
        
        if np.random.rand() < 0.07:  # Sickness reduces health
            self.health = max(0, self.health - 3)
            reward -= 2
        
        # Update state
        self.state = np.array([self.ects, self.energy, self.health], dtype=np.float32)
        
        # Check if the student has graduated or failed
        if self.ects >= 60:
            done = True  # Graduated
            reward += 20  # Bonus for graduating
        elif self.energy <= 0 or self.health <= 0:
            done = True  # Failed due to burnout or illness
            reward -= 10  # Penalty for failing
        
        return self.state, reward, done, False, {}
    
    def transition_probability(self, s, a, s_prime, r):
        """ Computes the probability of transitioning to state s' with reward r given state s and action a """
        return 1.0 if np.array_equal(s_prime, self.state) and r == self.reward_function(a) else 0.0
    
    def reward_function(self, action):
        """ Computes expected reward for a given action """
        if action == 0:
            return 5 if self.energy > 1 and self.health > 1 else -2
        elif action == 1:
            return 1
        elif action == 2:
            return 2
        elif action == 3:
            return -5
        return 0
    
    def sample_action(self, policy, s):
        """ Samples an action based on a given policy """
        return np.random.choice(len(policy), p=policy)
    
    def sample_episode(self, policy, max_T=100):
        """ Samples a full episode until termination or max time """
        episode = []
        s, _ = self.reset()
        for _ in range(max_T):
            a = self.sample_action(policy, s)
            s_prime, r, done, _, _ = self.step(a)
            episode.append((s, a, r))
            s = s_prime
            if done:
                break
        return episode
    
    def render(self):
        """ Render the current state """
        print(f"ECTS: {self.ects}, Energy: {self.energy}, Health: {self.health}")
    
    def close(self):
        """ Clean up resources """
        pass
    
# Register the environment
gym.envs.registration.register(
    id='StudentGraduation-v0',
    entry_point='__main__:StudentGraduationEnv',
)

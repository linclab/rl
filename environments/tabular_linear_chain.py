"""
Environment where you have a linear states that's risky. Wrong action and you get to the start with a negative reward. 
"""


import gym
from gym import spaces
from gym.utils import seeding
from gym.envs.registration import register
import numpy as np

class TabularLinearChain():
    """
    """
    def __init__(self, n=4):

        # Initialzing required parameters
        self.update_count = 0
        self.n = n # Length of the chain
        self.state = 0  # Start at beginning of the chain
        self.action_space = spaces.Discrete(2) # Number of actions: 2 - [0: step forward, 1: reset]
        self.observation_space = spaces.Discrete(self.n) # number of states is equal to chain length
        
        # Setting reward values
        self.step_reward = 0
        self.final_reward = 1
        
        self.seed() # not sure what this does, so not changing it

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        '''
        takes an action as an argument and returns the next_state, reward, done, info.
        '''
        
        # Making sure valid action is chosen
        assert self.action_space.contains(action)

        self.update_count += 1

        # Stepping along on the chain
        if(action == 0):
            self.state = self.state + 1
            reward = self.step_reward

        # Because this is a continuing case
        if(self.state == self.n - 1):
            done = True
            reward = self.final_reward
        else:
            done = False

        return self.state, reward, done, {}

    def reset(self):
        '''
        transitions back to first state
        '''
        self.update_count = 0 
        self.state = 0
        return self.state
    
    def vstar(self, gamma):
        k = gamma ** (np.arange(self.n-2, -1, -1))
        return np.append(k, np.zeros(1), axis=0)
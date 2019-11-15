"""
Code for running tabular SR agent on the gridworld.
"""

# Loading required libraries
import sys
sys.path.insert(0, '../../environments/')
sys.path.insert(0, '../../agents/')

import numpy as np
import gym
import matplotlib.pyplot as plt
from tabular_grid_world import TabularGridWorld
from tabularSR import TabularSRAgent
from tqdm import tqdm
from random import sample

# Setting up the environment
init_state = 40
env = TabularGridWorld(n=9, loc_r=[72], loc_t=[72], init_state=init_state)

# Creating the agent and setting up other params
agent = TabularSRAgent(env)
no_states = env.observation_space.n
no_episodes = 200
horizon = 1000
lr = 0.2

# Running a loop for each episode
for i in tqdm(range(no_episodes), leave=True):
	# reset environment
    state = env.reset(state=init_state)

    for t in range(horizon):
    	# select action and update it's state
        action = agent.select_action()
        state2, r, done, _ = env.step(action)
        agent.update(state, action, r, state2)
        if(done):
            break
        state = state2

# Learning finished
plt.figure(figsize=(10,10))
agent.visualize_M(s=0)

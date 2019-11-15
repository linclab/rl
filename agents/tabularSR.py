"""
Code implementing a tabular SR agent for control
"""

import numpy as np
import gym
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook as tqdm
from PIL import Image, ImageDraw, ImageFont
from random import sample

class TabularSRAgent():
    """Class for tabular SR agent for control
    """
    def __init__(self, env):

        # Initialization of params
        self.env = env
        self.M = np.random.random((self.env.observation_space.n, self.env.action_space.n, self.env.observation_space.n))
        self.R = np.random.random(self.env.observation_space.n)
        self.updates = 0
        self.gamma = 0.95
        
        # Setting the SR values of terminal states
        for i in range(self.env.observation_space.n):
            self.M[i, :, i] = np.ones(self.env.action_space.n) # TODO: do this for all states
        
        # Exploration parameters
        self.eps_decay = 5e4
        self.eps_start = 1
        self.eps_end = 0.05
        self.eps = self.eps_start
        pass
    
    def update_eps_linear(self):
        """
        Updating exploration rate
        """
        if(self.updates > self.eps_decay):
            return
        self.eps = self.eps_start + (self.eps_end - self.eps_start) * self.updates / self.eps_decay
    
    def select_action(self, greedy=False):
        """
        Function to choose the action for agent
        """
        thresh = np.random.rand()
        if(greedy):
            self.Q = self.M[self.env.state, :, :] @ self.R
            return np.argmax(self.Q)
        if(not greedy):
            self.update_eps_linear()
            
        if(greedy or thresh > self.eps):
            self.Q = self.M[self.env.state, :, :] @ self.R # CHECK
            greedy_actions = np.where(self.Q==np.amax(self.Q))
            return np.random.choice(greedy_actions[0])
        else:
            return self.env.action_space.sample()
    
    def update(self, s, a, r, s2, lr=0.01):
        """
        Function to update the SR and reward for the agent.
        Assumes deterministic state dependant reward.
        Updates the SR with a TD style update
        """
        Qtmp = self.M[self.env.state, :, :] @ self.R
        amax = np.where(Qtmp==np.amax(Qtmp))
        self.M[s, a, :] = (1 - lr) * self.M[s, a, :] + lr * \
                        ((np.arange(self.env.observation_space.n)==s).astype(dtype='int') \
                         + self.gamma * np.mean(self.M[s2, amax[0], :], axis=0))
        self.R[s2] = r # reward when you enter the state
        self.updates = self.updates + 1
        
    def visualize_Q(self):
        self.Q = self.M @ self.R
        plt.subplot(2,2,1)
        plt.imshow(np.flip(self.Q[:,0].reshape((self.env.n, self.env.n)), axis=0))
        plt.subplot(2,2,2)
        plt.imshow(np.flip(self.Q[:,1].reshape((self.env.n, self.env.n)), axis=0))
        plt.subplot(2,2,3)
        plt.imshow(np.flip(self.Q[:,2].reshape((self.env.n, self.env.n)), axis=0))
        plt.subplot(2,2,4)
        plt.imshow(np.flip(self.Q[:,3].reshape((self.env.n, self.env.n)), axis=0))
        plt.show()
        
    def ind2xy(self, ind):
        return self.env.n - 1 - ind // self.env.n, ind % self.env.n
    
    def dir2coords(self, val, x, y, cs):
        """Converts policy direction to coordinates
        for plotting a triangle.
        
        """
        if(val == 0):
            return [x, y, x, y-cs*0.45]
        elif(val == 1):
            return [x, y, x+cs*0.45, y]
        elif(val == 2):
            return [x, y, x, y+cs*0.45]
        elif(val == 3):
            return [x, y, x-cs*0.45, y]
        
    def visualize_policy(self, mode='visual'):
        h=self.env.n*self.env.cs
        w=self.env.n*self.env.cs
        cs = self.env.cs
        
        self.Q = self.M @ self.R
        if(mode=='value'):
            best_action_value = np.max(self.M @ self.R, axis=1)
        if(mode=='text'):
            best_action_value = np.argmax(self.M @ self.R, axis=1)
        if(mode=='visual'):
            best_action_value = np.argmax(self.M @ self.R, axis=1)
        img = self.env.render(mode='rgb_array', printR=False)
        fnt = ImageFont.truetype("arial.ttf", h//20)
        draw = ImageDraw.Draw(img)
        
        
        for i in range(self.env.n * self.env.n):
            y, x = self.ind2xy(i)
            x = x * cs + cs//2
            y = y * cs + cs//2
            if(mode=='value' or mode=='text'):
                draw.text([x-cs/3, y-cs/3], "{:.2f}".format(best_action_value[i]), font=fnt, fill="blue")
            if(mode=='visual'):
                Qtmp = self.M[i,:,:] @ self.R
                actions = np.where(Qtmp==np.amax(Qtmp))
                for action in actions[0]:
                    draw.line(self.dir2coords(action, x, y, cs), fill="darkblue", width=3)
                
        plt.imshow(img)
        plt.grid(color='k', linestyle='-', linewidth=2)
        plt.show()
        
    def get_Mpi(self):
        """Returns the SR matrix for current policy
        """
        self.Mpi = np.zeros((self.env.n**2, self.env.n**2))
        for i in range(self.env.n**2):
            # calculating best actions
            Qtmp = self.M[i,:,:] @ self.R
            actions = np.where(Qtmp==np.amax(Qtmp))
            for j in range(self.env.n**2):
                tmp = 0
                for action in actions[0]:
                    tmp = tmp + self.M[i, action, j]/actions[0].shape[0]
                self.Mpi[i, j] = tmp
                
        return self.Mpi
    
    def visualize_M(self, s):
        h = self.env.n*self.env.cs
        w = self.env.n*self.env.cs
        cs = self.env.cs
        
        self.get_Mpi()
        
        img = self.env.render(mode='rgb_array', printR=False)
        fnt = ImageFont.truetype("arial.ttf", h//20)
        draw = ImageDraw.Draw(img)
        
        for i in range(self.env.n * self.env.n):
            y, x = self.ind2xy(i)
            x = x * cs + cs//2
            y = y * cs + cs//2
            draw.text([x-cs/3, y-cs/3], "{:.2f}".format(self.Mpi[s, i]), font=fnt, fill="blue")
                
        plt.imshow(img)
        plt.title('M(s,:) for s = ' + str(s))
        plt.show()
    
    def visualize_R(self):
        plt.imshow(np.flip(self.R.reshape((5,5)), axis=0))
        plt.colorbar()
        plt.show()
        
    def evaluate(self, env, no_seeds=10, horizon=100):
        for i in range(no_seeds):
            s = env.reset()
            env.seed(i)
            t_vec = []
            for t in range(horizon):
                s2, r, done, _ = env.step(self.select_action(greedy=True))
                if(done):
                    t_vec.append(env.update_count)
                    
        return t_vec

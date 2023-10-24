import os
import random
import copy
import numpy as np
from modules.constants import constants
import gym
from gym import Env
from gym.spaces import Discrete, Box


random.seed(constants.SEED)
np.random.seed(constants.SEED)
os.environ['PYTHONHASHSEED']=str(constants.SEED)


class LupusEnv(Env):
    def __init__(self, X, Y, random=True):
        super(LupusEnv, self).__init__()
        self.X = X
        self.Y = Y
        self.feat_num = self.X.shape[1]
        self.actions = constants.ACTION_SPACE
        self.action_space = Discrete(len(self.actions))
        self.observation_space = Box(np.inf, np.inf, (self.feat_num,))
        self.random = random
        self.sample_num = len(X)
        self.idx =-1
        self.x = np.zeros((self.feat_num,), dtype=np.float32)
        self.y = np.nan
        self.state = np.full((self.feat_num,), -1, dtype=np.float32)
        self.num_classes = constants.CLASS_NUM
        self.episode_length = 0
        self.trajectory = []
        self.total_reward = 0
        self.seed()

    def seed(self, seed=constants.SEED): 
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]
        

    def step(self, action):
        if isinstance(action, np.ndarray):
            action == int(action)
        self.episode_length += 1
        reward = 0
            
        if action < self.num_classes: # if diagnosis action
            if action == self.y:
                reward += constants.CORRECT_DIAGNOSIS_REWARD
                self.total_reward += constants.CORRECT_DIAGNOSIS_REWARD
                is_success = True
            else:
                reward += constants.INCORRECT_DIAGNOSIS_REWARD
                self.total_reward += constants.INCORRECT_DIAGNOSIS_REWARD
                is_success = False
            terminated = False
            done = True
            y_actual = self.y 
            y_pred = int(action)
            self.trajectory.append(self.actions[action])
        
        elif self.actions[action] in self.trajectory: #repeated action
            terminated = True
            reward += constants.REPEATED_ACTION_REWARD
            self.total_reward += constants.REPEATED_ACTION_REWARD
            done = True
            y_actual = self.y 
            y_pred = constants.CLASS_DICT['Inconclusive diagnosis']
            is_success = True if y_actual == y_pred else False
            self.trajectory.append('Inconclusive diagnosis')
            
        else:
            reward += -1/(constants.LAMBDA*constants.FEATURE_SCORES[self.actions[action]])
            self.total_reward += -1/(constants.LAMBDA*constants.FEATURE_SCORES[self.actions[action]])
            terminated = False
            done = False
            y_actual = np.nan
            y_pred = np.nan
            is_success = None
            self.state = self.get_next_state(action - self.num_classes)
            self.trajectory.append(self.actions[action])

        info = {'index': self.idx, 'episode_length':self.episode_length, 'reward':self.total_reward, 'y_pred':y_pred, 'y_actual':y_actual, 
        'trajectory':self.trajectory, 'terminated':terminated, 'is_success': is_success}

        return self.state, reward, done, info


    def render(self):
        print(f'STEP {self.episode_length} for index {self.idx}')
        print(f'Current state: {self.state}')
        print(f'Total reward: {self.total_reward}')
        print(f'Trajectory: {self.trajectory}')

    
    def reset(self, idx=None):
        if idx is not None:
            self.idx = idx
        elif self.random:
            self.idx = random.randint(0, self.sample_num-1)
        else:
            self.idx += 1
            if self.idx >= self.sample_num:
                raise StopIteration()
        self.x, self.y = self.X[self.idx], self.Y[self.idx]
        self.state = np.full((self.feat_num,), -1, dtype=np.float32)
        self.trajectory = []
        self.episode_length = 0
        self.total_reward = 0
        return self.state
        
    def get_next_state(self, feature_idx):
        self.x = self.x.reshape(-1, self.feat_num)
        x_value = self.x[0, feature_idx]
        next_state = copy.deepcopy(self.state)
        next_state[feature_idx] = x_value
        return next_state
    





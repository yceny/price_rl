import sys
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import pandas as pd 
import numpy as np 
from scipy.stats import weibull_min
from scipy.stats import truncnorm
import math
import random
import pdb 

class PricingBaseEnv(gym.Env):

    def __init__(self):
        self.arrivalRate = 70 # arrival rate of shoppers per day, estimated visits to Walmart per day
        self.priceSensitivity = 0.004
        self.priceScale = 4 # k in Weibull distribution
        self.qualitySensitivity = 4 # 1/lambda in Weibull distribution
        self.qualityScale = 1

        self.numberOrder = 500 # assume we order the same quantity of products each time
        self.orderingCost = 500 * 3.
        self.priceLow = -1.
        self.priceHigh = 1.

        self._max_episode_steps = 12 # time before which products have to be sold
        self._cur_episode_step = 0
        self.unitOrderingCost = 3.

        self.qualityDeteriorateRate = 0.1
        self.priorQualityRate1 = 0.05
        self.priorQualityRate2 = 0.1
        self.priorQualityRate3 = 0.15
        self.priorQualitySigma0 = 0.001
        self.qualitySigma = 0.004

        self.action_space = spaces.Box(low = self.priceLow, high = self.priceHigh, shape=(1,), dtype=np.float32)
        
        self.observation_space = spaces.Box(low=np.array([0.]), high=np.array([self.numberOrder]))
        
        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        action = 3. * (np.array(action)+ 1.)
        
        self._cur_episode_step += 1
        
        inventoryLevel = self.state[0]

        # sample number of shoppers from Poisson process, among those shoppers, some buy apples, others not
        numberSP = np.random.poisson(self.arrivalRate)
        
        # calculate number of buyers, among the shoppers
        numberBuySP = 0

        for _ in range(numberSP):
            rn = random.choice([1,2,3,4,5,6,7,8])
            if rn == 1:
                bayesian_quality = math.exp(-self.priorQualityRate1 * self._cur_episode_step) + np.random.normal(0, self.qualitySigma)
            elif rn == 2:
                bayesian_quality = math.exp(-self.priorQualityRate2 * self._cur_episode_step) + np.random.normal(0, self.qualitySigma)
            else:
                bayesian_quality = math.exp(-self.priorQualityRate3 * self._cur_episode_step) + np.random.normal(0, self.qualitySigma)

            bayesian_quality = np.clip(bayesian_quality, 0., 1.0)
            probBuySP = math.exp(-(self.priceSensitivity*action)**self.priceScale)*(1-math.exp(-(self.qualitySensitivity*bayesian_quality)**self.qualityScale))

            rn = np.random.uniform()
            if rn < probBuySP:
                numberBuySP += 1

                    
        # test whether demand exceeds inventory
        if numberBuySP  < inventoryLevel: 
            inventoryLevel = inventoryLevel - numberBuySP
            self.state[0] = inventoryLevel
            reward = (action[0] - self.unitOrderingCost) * numberBuySP
        else:
            reward = (action[0] - self.unitOrderingCost) * inventoryLevel
            inventoryLevel = 0
            self.state[0] = inventoryLevel
        
        done = inventoryLevel == 0   
        
        if self._cur_episode_step == self._max_episode_steps:
            done = True
            reward = reward - (inventoryLevel * self.unitOrderingCost)

        new_obs = np.array([self.state[0] / (float(self.numberOrder / 1.))], dtype=np.float)

        return np.array(new_obs, dtype=np.float), reward / 100., done, {'buyer': numberBuySP}

    def reset(self):
        self.state = np.array([self.numberOrder])
        self._cur_episode_step = 0
        new_obs = np.array([self.state[0] / (float(self.numberOrder / 1.))], dtype=np.float)
        return np.array(new_obs, dtype=np.float)

    def render(self, mode = 'human', close = False):
        pass

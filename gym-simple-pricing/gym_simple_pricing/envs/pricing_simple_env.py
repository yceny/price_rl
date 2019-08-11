import sys
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import pandas as pd 
import numpy as np 
from scipy.stats import weibull_min
from scipy.stats import truncnorm
import math
import pdb 

class PricingSimpleEnv(gym.Env):

    #####################################################################################################
    # A monopoly retailer sells fresh produce to customers who come to retail stores, see the posted price, 
    # examine the quality of products and decide whether to purchase the product
    #
    # The goal of the retailer is to maximize revenue by setting optimal price over time and decide when
    # to donate the remaining product. 
    # 
    # Assumptions: 
    # (1) The retailer starts with N unit of products for each episode
    # (2) Customers arrive with Poisson distribution at a rate of arrival_rate, which is constant over time.
    # (3) Customers decision making depends on price sensitivity and quality sensitivity, which are 
    # heterogeneous among customers. We assume
    # Weibull distribution for both reservation price and reservation quality. 
    # (4) The quality of products deteriorate over time according to q = Qexp(-alpha * t).
    # (5) Products have a life span of time T steps. At the end of T, whatever left is donated to charity 
    # organization, and the monopoly receives tax benefits. 
    # (6) In each time step, a customer buys at most one unit of product. 
    # (7) Terminal state if either of the following occurs
    #     - N units of products have been sold out 
    #     - By the end of time T
    #     - the reatiler decides to donate the remaining products
    # 
    #
    # States: 
    # (1) inventory at hand n
    # (2) quality of product
    #
    # Actions: 
    # (1) price, continuous from 0 to 3
    # 
    #     
    #
    #####################################################################################################

    def __init__(self):
        self.arrivalRate = 70 # arrival rate of shoppers per day, estimated visits to Walmart per day
        self.priceSensitivity = 0.0045 # 1/lambda in Weibull distribution
        self.priceScale = 4 # k in Weibull distribution
        self.qualitySensitivity = 4 # 1/lambda in Weibull distribution
        self.qualityScale = 4 # k in Weibull distribution

        self.numberOrder = 500 # assume we order the same quantity of products each time
        self.orderingCost = 50
        self.priceLow = -1.
        self.priceHigh = 1.
        self.qualityLow = 0.
        self.qualityHigh = 1.
        # self.taxRate = 0.3 # tax benefit from donating
        
        self._max_episode_steps = 15
        self._cur_episode_step = 0
        self.unitOrderingCost = 150.

        # self.qualityDeteriorateRate = 0.1
        # self.priorQualityRate = 0.12
        # self.priorQualitySigma0 = 0.0002
        # self.qualitySigma = 0.0005
        self.qualityDeteriorateRate = 0.1
        self.priorQualityRate = 0.15
        self.priorQualitySigma0 = 0.001
        self.qualitySigma = 0.004
        
        self.action_space = spaces.Box(low = self.priceLow, high = self.priceHigh, shape=(1,), dtype=np.float32)
        #TODO: FIXME: not sure if it is better to use a normlized version of numberOrder
        ### quality need to start at high, inventory level need to start at numberOrder
        # self.observation_space = spaces.Tuple([spaces.Discrete(self.numberOrder + 1), 
        #                                        spaces.Box(low = self.qualityLow, high = self.qualityHigh, shape=(1,), dtype = np.float32)])
        self.observation_space = spaces.Box(low=np.array([0., self.qualityLow]), high=np.array([self.numberOrder, self.qualityHigh]))
        
        self.seed()
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        # shift to range [1, 3]
        action = np.clip(action, self.action_space.low, self.action_space.high)
        # action = 75 * (np.array(action)+ 3.)
        action = 150.0 * (np.array(action)+ 1.)

        self._cur_episode_step += 1
        tmp_quality = math.exp(-self.qualityDeteriorateRate * self._cur_episode_step) + np.random.normal(0, self.qualitySigma)
        tmp_quality = np.clip(tmp_quality, 0., 1.0)
        self.state[1] = tmp_quality
        
        inventoryLevel = self.state[0]

        # sample number of shoppers from Poisson process, among those shoppers, some buy apples, others not
        numberSP = np.random.poisson(self.arrivalRate)

        bayesian_quality = math.exp(-self.priorQualityRate * self._cur_episode_step) + np.random.normal(0, self.qualitySigma)
        bayesian_quality = np.clip(bayesian_quality, 0., 1.0)
        # probability of buying product for each individual customer
        probBuySP = math.exp(-(self.priceSensitivity*action[0])**self.priceScale)*(1-math.exp(-(self.qualitySensitivity*bayesian_quality)**self.qualityScale))
        
        # calculate number of buyers, among the shoppers
        numberBuySP = 0
        
        for i in range(numberSP):
            # probBuySP = (1 - weibull_min.cdf(c = 0.05, x = action[0], scale = 0.1 / float(PS[i]))) * weibull_min.cdf(c = 0.1, x = quality, scale = 1 / float(QS[i]))  
            
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
        
        if self._cur_episode_step >= self._max_episode_steps:
            done = True
            reward = reward - (inventoryLevel * self.unitOrderingCost)

        new_obs = np.array([self.state[0] / (float(self.numberOrder / 1.)) , self.state[1]], dtype=np.float)

        return np.array(new_obs, dtype=np.float), reward / 100., done, {}

    def reset(self):
        self.state = np.array([self.numberOrder, self.qualityHigh])
        self._cur_episode_step = 0
        new_obs = np.array([self.state[0] / (float(self.numberOrder / 1.)) , self.state[1]], dtype=np.float)
        self.hist_y = [np.array(self.state[1])]
        return np.array(new_obs, dtype=np.float)

    def render(self, mode = 'human', close = False):
        pass

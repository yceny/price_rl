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


def dnlogp_grad(data_x, data_y, c_alpha, ratio_sigma2, mu_0):
    XY = (data_y - np.exp(- data_x * c_alpha)) * data_x *  np.exp( - data_x * c_alpha)
    prior = ratio_sigma2 * (c_alpha - mu_0)
    return 2. * (np.sum(XY) + prior)
    

def nlogp_loss(data_x, data_y, c_alpha, ratio_sigma2, mu_0):
    XY =  (data_y - np.exp( - data_x * c_alpha)) ** 2 
    prior = ratio_sigma2 *((c_alpha - mu_0) ** 2)
    return np.sum(XY) + prior

def perform_SGD(data_x, data_y, init_alpha, ratio_sigma2, mu_0, lr=1e-3):
    c_alpha = init_alpha
    loss = nlogp_loss(data_x, data_y, c_alpha, ratio_sigma2, mu_0)
    for tt in range(50):
        c_alpha -= lr * dnlogp_grad(data_x, data_y, c_alpha, ratio_sigma2, mu_0)
        loss = nlogp_loss(data_x, data_y, c_alpha, ratio_sigma2, mu_0)
    return c_alpha

class PricingModel2Env(gym.Env):

    def __init__(self):

        # arrival rate for each group of customers
        self.arrivalRate = 70 # arrival rate of price-sensitive shoppers per day


        self.priceSensitivity = 0.004
        self.priceScale = 4 # k in Weibull distribution
        self.qualitySensitivity = 4 # 1/lambda in Weibull distribution
        self.qualityScale = 1

        self.numberOrder = 500 # assume we order the same quantity of products each time
        self.orderingCost = 500 * 3.
        self.priceLow = -1.
        self.priceHigh = 1.
        self.qualityLow = 0.
        self.qualityHigh = 1.
        # self.taxRate = 0.3 # tax benefit from donating
        
        self._max_episode_steps = 12 # time before which products have to be sold
        self._cur_episode_step = 0
        self.unitOrderingCost = 3.

        self.qualityDeteriorateRate = 0.1
        self.priorQualityRate1 = 0.05
        self.priorQualityRate2 = 0.1
        self.priorQualityRate3 = 0.15
 
        self.priorQualitySigma0 = 0.001
        self.qualitySigma = 0.004

        self.action_space = spaces.Box(low = self.priceLow, high = self.priceHigh, shape=(2,), dtype=np.float32)

        self.observation_space = spaces.Box(low=np.array([0., self.qualityLow]), high=np.array([self.numberOrder, self.qualityHigh]))
        
        self.seed()
        self.reset()
    

    def get_bayesian_alpha(self, hist_y, hist_x, alpha):
        assert (len(hist_y) > 0) and (len(hist_y) <= 22)
        data_y = np.array(hist_y)
        data_x = np.array(hist_x)

        init_alpha = alpha
        ratio_sigma2 = (self.qualitySigma / self.priorQualitySigma0) ** 2
        c_alpha = perform_SGD(data_x, data_y, init_alpha, ratio_sigma2, alpha, lr=1e-3)
        return c_alpha


    def get_bayesian_alpha_linear(self, hist_y, hist_x, alpha):
        assert (len(hist_y) > 0) and (len(hist_y) <= 22)
        data_y = np.array(hist_y)
        data_x = np.array(hist_x)
        
        init_alpha = alpha
        ratio_sigma2 = (self.qualitySigma / self.priorQualitySigma0) ** 2
        c_alpha = (np.sum((1 - data_y) * data_x)) / (ratio_sigma2 + np.sum(data_x*data_x))
        return c_alpha

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        price = 3. * ( action[0] + 1.)
        act1 = (action[1] + 1.) / 2.
        
        
        self._cur_episode_step += 1
        tmp_quality = math.exp(-self.qualityDeteriorateRate * self._cur_episode_step) + np.random.normal(0, self.qualitySigma)
        tmp_quality = np.clip(tmp_quality, 0., 1.0)
        self.state[1] = tmp_quality
        
        inventoryLevel = self.state[0]

        ms_flag = False
       
        if np.random.uniform() < act1:
            self.hist_y.append(float(tmp_quality))
            self.hist_x.append(float(self._cur_episode_step))
            ms_flag = True

        numberSP = np.random.poisson(self.arrivalRate)

        # sample number of shoppers from Poisson process, among those shoppers, some buy apples, others not
        alpha1 = self.get_bayesian_alpha(self.hist_y, self.hist_x, self.priorQualityRate1)
        quality1 = math.exp(-alpha1 * self._cur_episode_step) + np.random.normal(0, self.qualitySigma)
        quality1 = np.clip(quality1, 0., 1.0)
        self.hist_alpha1.append(alpha1)

        alpha2 = self.get_bayesian_alpha(self.hist_y, self.hist_x, self.priorQualityRate2)
        quality2 = math.exp(-alpha2 * self._cur_episode_step) + np.random.normal(0, self.qualitySigma)
        quality2 = np.clip(quality2, 0., 1.0)
        self.hist_alpha2.append(alpha2)

        alpha3 = self.get_bayesian_alpha(self.hist_y, self.hist_x, self.priorQualityRate3)
        quality3 = math.exp(-alpha3 * self._cur_episode_step) + np.random.normal(0, self.qualitySigma)
        quality3 = np.clip(quality3, 0., 1.0)
        self.hist_alpha3.append(alpha3)

        numberBuySP = 0
        
        for _ in range(numberSP):
            rn = random.choice([1,2,3,4,5,6,7,8])

            if rn == 1:
                probBuySP = math.exp(-(self.priceSensitivity*price)**self.priceScale)*(1-math.exp(-(self.qualitySensitivity*quality1)**self.qualityScale))
                
            elif rn == 2:
                probBuySP = math.exp(-(self.priceSensitivity*price)**self.priceScale)*(1-math.exp(-(self.qualitySensitivity*quality2)**self.qualityScale))
                
            else:
                probBuySP = math.exp(-(self.priceSensitivity*price)**self.priceScale)*(1-math.exp(-(self.qualitySensitivity*quality3)**self.qualityScale))
               

            rn = np.random.uniform()
            if rn < probBuySP:
                numberBuySP += 1

        # numberBuySP = PSnumberBuySP + QSnumberBuySP
        
        # test whether demand exceeds inventory
        if numberBuySP  < inventoryLevel: 
            inventoryLevel = inventoryLevel - numberBuySP
            self.state[0] = inventoryLevel
            reward = (price - self.unitOrderingCost)* numberBuySP 
        else:
            reward = (price - self.unitOrderingCost) * inventoryLevel 
            inventoryLevel = 0
            self.state[0] = inventoryLevel

        done = inventoryLevel == 0   
        
        if self._cur_episode_step >= self._max_episode_steps:
            done = True
            reward = reward - (inventoryLevel * self.unitOrderingCost)
        
        new_obs = np.array([self.state[0] / float(self.numberOrder * 1.0), self.state[1]], dtype=np.float)

        # return new_obs, reward / 100., done, {}
        return new_obs, reward / 100., done, {'buyer': numberBuySP}

    def reset(self):
        self.state = np.array([self.numberOrder, self.qualityHigh])
        self._cur_episode_step = 0
        self.hist_y = [self.state[1]]
        self.hist_x = [0]
        self.hist_alpha1 = []
        self.hist_alpha2 = []
        self.hist_alpha3 = []
        new_obs = np.array([self.state[0] / float(self.numberOrder * 1.0), self.state[1]], dtype=np.float)
        return new_obs

    def render(self, mode = 'human', close = False):
        pass

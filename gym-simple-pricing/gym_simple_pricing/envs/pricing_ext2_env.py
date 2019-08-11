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


## NOTE; the clanss name has been changed

class PricingExt2Env(gym.Env):

    #####################################################################################################
    # A monopoly retailer sells fresh produce to customers who come to retail stores, see the posted price, 
    # examine the quality of products and decide whether to purchase the product
    #
    # The goal of the retailer is to maximize revenue by setting optimal price over time and decide when
    # to donate the remaining product. 
    # 
    # Extension 2: add another action of informing customers. informing customers can increase arrival rate 
    # of customers
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
        self.orderingCost = 500 * 150
        self.unitOrderingCost = 150.
        self.priceLow = -1.
        self.priceHigh = 1.
        self.qualityLow = 0.
        self.qualityHigh = 1.
        self.messageCost = 500
        # self.taxRate = 0.3 # tax benefit from donating
        # self.maxTime = 21 # time before which products have to be sold
        self._max_episode_steps = 15
        self._cur_episode_step = 0

        # self.qualityDeteriorateRate = 0.1
        # self.priorQualityRate = 0.12
        # self.priorQualitySigma0 = 0.0002
        # self.qualitySigma = 0.0005
        self.qualityDeteriorateRate = 0.1
        self.priorQualityRate = 0.15
        self.priorQualitySigma0 = 0.001
        self.qualitySigma = 0.004

        self.action_space = spaces.Box(low = self.priceLow, high = self.priceHigh, shape=(2,), dtype=np.float32)
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
    
    def get_bayesian_alpha(self):
        assert (len(self.hist_y) > 0) and (len(self.hist_y) <= 22)
        data_y = np.array(self.hist_y)
        data_x = np.array(self.hist_x)

        init_alpha = self.priorQualityRate
        ratio_sigma2 = (self.qualitySigma / self.priorQualitySigma0) ** 2
        c_alpha = perform_SGD(data_x, data_y, init_alpha, ratio_sigma2, self.priorQualityRate, lr=1e-3)
        return c_alpha

    def step(self, action):
        # shift to range [1, 3]
        # NOTE: i used price instead of action here, and changed it to price in the following
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        # price = 75 * ( action[0] + 3.)
        price = 150 * ( action[0] + 1.)
        act1 = (action[1] + 1.) / 2.
        

        
        self._cur_episode_step += 1
        tmp_quality = math.exp(-self.qualityDeteriorateRate * self._cur_episode_step) + np.random.normal(0, self.qualitySigma)
        tmp_quality = np.clip(tmp_quality, 0., 1.0)
        self.state[1] = tmp_quality
        
        inventoryLevel = self.state[0]
        
        Message_flag = False
       
        if np.random.uniform() < act1:
            self.hist_y.append(float(tmp_quality))
            self.hist_x.append(float(self._cur_episode_step))
            Message_flag = True



        # sample number of shoppers from Poisson process, among those shoppers, some buy apples, others not
        numberSP = np.random.poisson(self.arrivalRate)

        bayes_alpha = self.get_bayesian_alpha()
        bayesian_quality = math.exp(-bayes_alpha * self._cur_episode_step) + np.random.normal(0, self.qualitySigma)
        bayesian_quality = np.clip(bayesian_quality, 0., 1.0)
        # probability of buying product for each individual customer
        probBuySP = math.exp(-(self.priceSensitivity*price)**self.priceScale)*(1-math.exp(-(self.qualitySensitivity*bayesian_quality)**self.qualityScale))
        self.hist_alpha.append(float(bayes_alpha))
        # calculate number of buyers, among the shoppers
        numberBuySP = 0
        
        for i in range(numberSP):
            
            rn = np.random.uniform()
            if rn < probBuySP:
                numberBuySP += 1

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
        
        if self._cur_episode_step == self._max_episode_steps:
            done = 1
            reward = reward - (inventoryLevel * self.unitOrderingCost)
        
        #TODO: FIXME: try different messageCost including zero
        if Message_flag:
            reward = reward - self.messageCost * 1
        
        new_obs = np.array([self.state[0] / float(self.numberOrder * 1.), self.state[1]], dtype=np.float)

        return new_obs, reward / 100., done, {}

    def reset(self):
        self.state = np.array([self.numberOrder, self.qualityHigh])
        self._cur_episode_step = 0
        new_obs = np.array([self.state[0] / float(self.numberOrder * 1.), self.state[1]], dtype=np.float)
        self.hist_y = [self.state[1]]
        self.hist_x = [0]
        self.hist_alpha = []
        return new_obs

    def render(self, mode = 'human', close = False):
        pass

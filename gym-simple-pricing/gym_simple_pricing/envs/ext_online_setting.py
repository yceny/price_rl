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

## NOTE; the clanss name has been changed


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

## NOTE: change the environment name
class PricingExtOnlineEnv(gym.Env):

    #####################################################################################################
    # A monopoly retailer sells fresh produce to customers who come to retail stores, see the posted price, 
    # examine the quality of products and decide whether to purchase the product
    #
    # The goal of the retailer is to maximize revenue by setting optimal price over time and decide whether to
    # inform customers about the true quality of product quality
    # 
    # Extension Online Setting: consider two groups of customers, and set price and information strategy for
    # customers in respective group. Four actions in totall
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
    # (1) price, continuous from 0 to 300
    # (2) information strategy, whether to notify customers about food quality
    #     
    #
    #####################################################################################################

    def __init__(self):

        # arrival rate for each group of customers
        self.PSarrivalRate = 35 # arrival rate of price-sensitive shoppers per day
        self.QSarrivalRate = 35 # arrival rate of quality-sensitive shoppers per day

        
        # price sensitivity and quality sensitivity for each group of customers
        # self.PSpriceSensitivity = 0.005
        # self.PSpriceScale = 5
        # self.PSqualitySensitivity = 30
        # self.PSqualityScale = 4

        # self.QSpriceSensitivity = 0.003
        # self.QSpriceScale = 3
        # self.QSqualitySensitivity = 1.5
        # self.QSqualityScale = 4

        self.PSpriceSensitivity = 0.005
        self.PSpriceScale = 5
        self.PSqualitySensitivity = 20
        self.PSqualityScale = 1

        self.QSpriceSensitivity = 0.003
        self.QSpriceScale = 3
        self.QSqualitySensitivity = 3
        self.QSqualityScale = 1


        self.numberOrder = 500 # assume we order the same quantity of products each time
        self.orderingCost = 500 * 150 # assume unit ordering cost is 150
        self.unitOrderingCost = 150.
        self.messageCost = 250.
        self.priceLow = -1.
        self.priceHigh = 1.
        self.qualityLow = 0.
        self.qualityHigh = 1.
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

        ## NOTE: I changed the action space to shape 4
        self.action_space = spaces.Box(low = self.priceLow, high = self.priceHigh, shape=(4,), dtype=np.float32)

        self.observation_space = spaces.Box(low=np.array([0., self.qualityLow]), high=np.array([self.numberOrder, self.qualityHigh]))
        
        self.seed()
        self.reset()
    

    def get_bayesian_alpha(self, hist_y, hist_x):
        assert (len(hist_y) > 0) and (len(hist_y) <= 22)
        data_y = np.array(hist_y)
        data_x = np.array(hist_x)

        init_alpha = self.priorQualityRate
        ratio_sigma2 = (self.qualitySigma / self.priorQualitySigma0) ** 2
        c_alpha = perform_SGD(data_x, data_y, init_alpha, ratio_sigma2, self.priorQualityRate, lr=1e-3)
        return c_alpha

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        # shift to range [1, 3]
        # NOTE: i used price instead of action here, and changed it to price in the following
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # price = 75 * ( action[0] + 3.)
        ## NOTE: add a new price action
        price1 = 150 * ( action[0] + 1.) # price for PS customers
        price2 = 150 * ( action[1] + 1.) # price for QS customers
        act1 = (action[2] + 1.) / 2. # information strategy for PS customers
        act2 = (action[3] + 1.) / 2. # information strategy for QS customers
        
        self._cur_episode_step += 1
        tmp_quality = math.exp(-self.qualityDeteriorateRate * self._cur_episode_step) + np.random.normal(0, self.qualitySigma)
        tmp_quality = np.clip(tmp_quality, 0., 1.0)
        self.state[1] = tmp_quality
        
        inventoryLevel = self.state[0]

        ms_ps_flag = False
       
        if np.random.uniform() < act1:
            self.hist_y_ps.append(float(tmp_quality))
            self.hist_x_ps.append(float(self._cur_episode_step))
            ms_ps_flag = True
        
        ms_qs_flag = False
        if np.random.uniform() < act2:
            self.hist_y_qs.append(float(tmp_quality))
            self.hist_x_qs.append(float(self._cur_episode_step))
            ms_qs_flag = True

        # sample number of shoppers from Poisson process, among those shoppers, some buy apples, others not
        # DIFFERENT
        PSnumberSP = np.random.poisson(self.PSarrivalRate)
        QSnumberSP = np.random.poisson(self.QSarrivalRate)

        
        alpha_ps = self.get_bayesian_alpha(self.hist_y_ps, self.hist_x_ps)
        quality_ps = math.exp(-alpha_ps * self._cur_episode_step) + np.random.normal(0, self.qualitySigma)
        quality_ps = np.clip(quality_ps, 0., 1.0)
        # probability of buying product for each price sensitive individual customer
        PSprobBuySP = math.exp(-(self.PSpriceSensitivity*price1)**self.PSpriceScale)*(1-math.exp(-(self.PSqualitySensitivity*quality_ps)**self.PSqualityScale))
        self.hist_alpha_ps.append(alpha_ps)
        
        alpha_qs = self.get_bayesian_alpha(self.hist_y_qs, self.hist_x_qs)
        quality_qs = math.exp(-alpha_qs * self._cur_episode_step + np.random.normal(0, self.qualitySigma))
        quality_qs = np.clip(quality_qs, 0., 1.0)
        # probability of buying product for each individual customer
        QSprobBuySP = math.exp(-(self.QSpriceSensitivity*price2)**self.QSpriceScale)*(1-math.exp(-(self.QSqualitySensitivity*quality_qs)**self.QSqualityScale))
        self.hist_alpha_qs.append(alpha_qs)
        # calculate number of buyers of each group of customers, among the shoppers
        PSnumberBuySP = 0
        QSnumberBuySP = 0
        
        for i in range(PSnumberSP):
            rn = np.random.uniform()
            if rn < PSprobBuySP:
                PSnumberBuySP += 1

        for i in range(QSnumberSP):
            
            rn = np.random.uniform()
            if rn < QSprobBuySP:
                QSnumberBuySP += 1

        numberBuySP = PSnumberBuySP + QSnumberBuySP
        
        # test whether demand exceeds inventory
        if numberBuySP  < inventoryLevel: 
            inventoryLevel = inventoryLevel - numberBuySP
            self.state[0] = inventoryLevel
            reward = (price1 - self.unitOrderingCost)* PSnumberBuySP + (price2 - self.unitOrderingCost)* QSnumberBuySP
        ## NOTE: when there is a inventory shortage, we assume the retailer first sell products to the group with higher prices
        else:
            if price1 > price2:
                if PSnumberBuySP < inventoryLevel:
                    reward = (price1 - self.unitOrderingCost)* PSnumberBuySP + (price2 - self.unitOrderingCost)* (inventoryLevel - PSnumberBuySP)
                else:
                    reward = (price1 - self.unitOrderingCost)* inventoryLevel
            else:
                if QSnumberBuySP < inventoryLevel:
                    reward = (price1 - self.unitOrderingCost)* (inventoryLevel - QSnumberBuySP) + (price2 - self.unitOrderingCost)* QSnumberBuySP
                else:
                    (price2 - self.unitOrderingCost)* inventoryLevel
            # reward = (price - self.unitOrderingCost) * inventoryLevel 
            inventoryLevel = 0
            self.state[0] = inventoryLevel

        done = inventoryLevel == 0   
        
        if self._cur_episode_step == self._max_episode_steps:
            done = True
            reward = reward - (inventoryLevel * self.unitOrderingCost) 

        if ms_ps_flag:
            reward = reward - self.messageCost * 1
        if ms_qs_flag:
            reward = reward - self.messageCost * 1
        
        
        
        new_obs = np.array([self.state[0] / float(self.numberOrder * 1.0), self.state[1]], dtype=np.float)

        return new_obs, reward / 100., done, {}

    def reset(self):
        self.state = np.array([self.numberOrder, self.qualityHigh])
        self._cur_episode_step = 0
        self.hist_y_ps = [self.state[1]]
        self.hist_x_ps = [0]
        self.hist_alpha_ps = []

        self.hist_y_qs = [self.state[1]]
        self.hist_x_qs = [0]
        self.hist_alpha_qs = []
        new_obs = np.array([self.state[0] / float(self.numberOrder * 1.0), self.state[1]], dtype=np.float)
        return new_obs

    def render(self, mode = 'human', close = False):
        pass
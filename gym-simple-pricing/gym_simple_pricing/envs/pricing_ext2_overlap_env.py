import sys
import random
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

class PricingEx2OverlapEnv(gym.Env):

    ## NOTE: please read this carefully
    #####################################################################################################
    # A monopoly retailer sells fresh produce to customers who come to retail stores, see the posted price, 
    # examine the quality of products and decide whether to purchase the product
    #
    # The goal of the retailer is to maximize revenue by setting optimal price over time.
    # 
    # Assumptions: 
    # (1) The retailer starts with N unit of products with life span time T.
    # (1.1) Another N unit of products arrive after a shift window ST, 
    # during the middle when the first batch of products are sold.
    # (2) Customers arrive with Poisson distribution at a rate of arrival_rate, which is constant over time.
    # (3) Customers decision making depends on price sensitivity and quality sensitivity, which are 
    # heterogeneous among customers. We assume Weibull distribution for both reservation price and reservation quality. 
    # (4) The quality of products deteriorate over time according to q = Qexp(-alpha * t).
    # (5) Each batch of products have a life span of time T steps. At the end of T, whatever left is wasted, 
    # no salvage value.
    # (6) In each time step, a customer buys at most one unit of product. 
    # (7) Terminal state if either of the following occurs
    #     - 2N units of products have been sold out 
    #     - By the end of time T + shift window.
    # 
    #
    # States:
    # (1) inventory at hand n: 2 states here, one for each batch of products
    # (2) quality of product: 2 states here, one for each batch of products
    #
    # Actions: 
    # (1) price, continuous from 0 to 3: 2 actions, one for each batch of products
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
        self.priceLow = -1.
        self.priceHigh = 1.
        self.qualityLow = 0.
        self.qualityHigh = 1.
        self.messageCost = 500
        # self.taxRate = 0.3 # tax benefit from donating
        
        self._max_episode_steps = 15
        ## NOTE: create two counter to track the current time in each batch of products' life span
        self._cur_episode_step1 = 0
        self._cur_episode_step2 = 0
        self.unitOrderingCost = 150.

        # self.qualityDeteriorateRate = 0.1
        # self.priorQualityRate = 0.12
        # self.priorQualitySigma0 = 0.0002
        # self.qualitySigma = 0.0005
        self.qualityDeteriorateRate = 0.1
        self.priorQualityRate = 0.15
        self.priorQualitySigma0 = 0.001
        self.qualitySigma = 0.004

        ## NOTE: add a shift window, after how many time steps, the second batch of products arrive
        ## NOTE: add done for each batch of products
        self.ST = 5
        self.done1 = False
        self.done2 = True
        
        self.action_space = spaces.Box(low = self.priceLow, high = self.priceHigh, shape=(4,), dtype=np.float32)
        #TODO: FIXME: not sure if it is better to use a normlized version of numberOrder
        ### quality need to start at high, inventory level need to start at numberOrder
        # self.observation_space = spaces.Tuple([spaces.Discrete(self.numberOrder + 1), 
        #                                        spaces.Box(low = self.qualityLow, high = self.qualityHigh, shape=(1,), dtype = np.float32)])
        self.observation_space = spaces.Box(low=np.array([0., 0., self.qualityLow,self.qualityLow]), high=np.array([self.numberOrder, self.numberOrder, self.qualityHigh, self.qualityHigh]))
        
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
        # shift to range [0, 300]
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        action1 = 150.0 * (np.array(action[0])+ 1.)
        action2 = 150.0 * (np.array(action[1])+ 1.)

        msg1 = (action[2] + 1.) / 2.
        msg2 = (action[3] + 1.) / 2.

        ## NOTE: track tiem step for the second batch of products.
        self.done2 = self._cur_episode_step1 < self.TS

        if not self.done1 and self.done2:
            self._cur_episode_step1 += 1
            tmp_quality1 = math.exp(-self.qualityDeteriorateRate * self._cur_episode_step1) + np.random.normal(0, self.qualitySigma)
            tmp_quality1 = np.clip(tmp_quality1, 0., 1.0)
            self.state[2] = tmp_quality1
            inventoryLevel1 = self.state[0]

            Message_flag1 = False
       
            if np.random.uniform() < msg1:
                self.hist_y.append(float(tmp_quality1))
                self.hist_x.append(float(self._cur_episode_step1))
                Message_flag1 = True

            bayes_alpha1 = self.get_bayesian_alpha()
            bayesian_quality1 = math.exp(-bayes_alpha1 * self._cur_episode_step1) + np.random.normal(0, self.qualitySigma)
            bayesian_quality1 = np.clip(bayesian_quality1, 0., 1.0)

            # probability of buying product for each individual customer
            probBuySP1 = math.exp(-(self.priceSensitivity*action1)**self.priceScale)*(1-math.exp(-(self.qualitySensitivity*bayesian_quality1)**self.qualityScale))
            self.hist_alpha1.append(float(bayes_alpha1))

            numberSP = np.random.poisson(self.arrivalRate)
            numberBuySP1 = 0

            for i in range(numberSP):
                rn = np.random.uniform()
                if rn < probBuySP1:
                    numberBuySP1 += 1

            if numberBuySP1  < inventoryLevel1: 
                inventoryLevel1 = inventoryLevel1 - numberBuySP1
                self.state[0] = inventoryLevel1
                reward1 = (action1 - self.unitOrderingCost) * numberBuySP1
            else:
                reward1 = (action1 - self.unitOrderingCost) * inventoryLevel1
                inventoryLevel1 = 0
                self.state[0] = inventoryLevel1
            
            self.done1 = inventoryLevel1 == 0  
            if self._cur_episode_step1 >= self._max_episode_steps:
                self.done1 = True
                reward1 = reward1 - (inventoryLevel1 * self.unitOrderingCost)

            if Message_flag1:
                reward1 = reward1 - self.messageCost * 1

            new_obs = np.array([self.state[0] / (float(self.numberOrder / 1.)) , self.state[1] / (float(self.numberOrder / 1.)), self.state[2], self.state[3]], dtype=np.float)

            return np.array(new_obs, dtype=np.float), reward1 / 100., self.done1, self.done2, {}

        elif (not self.self.done1) and (not self.done2):

            self._cur_episode_step1 += 1
            tmp_quality1 = math.exp(-self.qualityDeteriorateRate * self._cur_episode_step1) + np.random.normal(0, self.qualitySigma)
            tmp_quality1 = np.clip(tmp_quality1, 0., 1.0)
            self.state[2] = tmp_quality1
            inventoryLevel1 = self.state[0]

            Message_flag1 = False
       
            if np.random.uniform() < msg1:
                self.hist_y.append(float(tmp_quality1))
                self.hist_x.append(float(self._cur_episode_step1))
                Message_flag1 = True

            bayes_alpha1 = self.get_bayesian_alpha()
            bayesian_quality1 = math.exp(-bayes_alpha1 * self._cur_episode_step1) + np.random.normal(0, self.qualitySigma)
            bayesian_quality1 = np.clip(bayesian_quality1, 0., 1.0)

            # probability of buying product for each individual customer
            probBuySP1 = math.exp(-(self.priceSensitivity*action1)**self.priceScale)*(1-math.exp(-(self.qualitySensitivity*bayesian_quality1)**self.qualityScale))
            self.hist_alpha1.append(float(bayes_alpha1))


            self._cur_episode_step2 += 1
            tmp_quality2 = math.exp(-self.qualityDeteriorateRate * self._cur_episode_step2) + np.random.normal(0, self.qualitySigma)
            tmp_quality2 = np.clip(tmp_quality2, 0., 1.0)
            self.state[3] = tmp_quality2
            inventoryLevel2 = self.state[1]

            Message_flag2 = False
       
            if np.random.uniform() < msg2:
                self.hist_y2.append(float(tmp_quality2))
                self.hist_x2.append(float(self._cur_episode_step2))
                Message_flag2 = True

            bayes_alpha2 = self.get_bayesian_alpha()
            bayesian_quality2 = math.exp(-bayes_alpha2 * self._cur_episode_step2) + np.random.normal(0, self.qualitySigma)
            bayesian_quality2 = np.clip(bayesian_quality2, 0., 1.0)

            # probability of buying product for each individual customer
            probBuySP2 = math.exp(-(self.priceSensitivity*action2)**self.priceScale)*(1-math.exp(-(self.qualitySensitivity*bayesian_quality2)**self.qualityScale))
            self.hist_alpha2.append(float(bayes_alpha2))
        
            # sample number of shoppers from Poisson process, among those shoppers, some buy apples, others not
            numberSP = np.random.poisson(self.arrivalRate)
            # calculate number of buyers, among the shoppers
            numberBuySP1 = 0
            numberBuySP2 = 0
            
            for i in range(numberSP):
                # probBuySP = (1 - weibull_min.cdf(c = 0.05, x = action[0], scale = 0.1 / float(PS[i]))) * weibull_min.cdf(c = 0.1, x = quality, scale = 1 / float(QS[i]))  
                if probBuySP1 > probBuySP2:
                    rn = np.random.uniform()
                    if rn < probBuySP1:
                        numberBuySP1 += 1
                elif probBuySP1 < probBuySP2:
                    rn = np.random.uniform()
                    if rn < probBuySP2:
                        numberBuySP2 += 1
                else:
                    rn = np.random.uniform()
                    if rn < probBuySP1:
                        if random.choice([0,1]) == 0:
                            numberBuySP1 += 1
                        else:
                            numberBuySP2 += 1

            # test whether demand exceeds inventory
            if numberBuySP1  < inventoryLevel1: 
                inventoryLevel1 = inventoryLevel1 - numberBuySP1
                self.state[0] = inventoryLevel1
                reward1 = (action1 - self.unitOrderingCost) * numberBuySP1
            else:
                reward1 = (action1 - self.unitOrderingCost) * inventoryLevel1
                inventoryLevel1 = 0
                self.state[0] = inventoryLevel1
            
            if numberBuySP2  < inventoryLevel2: 
                inventoryLevel2 = inventoryLevel2 - numberBuySP2
                self.state[1] = inventoryLevel2
                reward2 = (action2 - self.unitOrderingCost) * numberBuySP2
            else:
                reward2 = (action2 - self.unitOrderingCost) * inventoryLevel2
                inventoryLevel2 = 0
                self.state[1] = inventoryLevel2
            
            
            self.done1 = inventoryLevel1 == 0  
            self.done2 = inventoryLevel2 == 0 
            
            if self._cur_episode_step1 >= self._max_episode_steps:
                self.done1 = True
                reward1 = reward1 - (inventoryLevel1 * self.unitOrderingCost)
            
            if self._cur_episode_step2 >= self._max_episode_steps:
                self.done2 = True
                reward2 = reward2 - (inventoryLevel2 * self.unitOrderingCost)

            if Message_flag1:
                reward1 = reward1 - self.messageCost * 1

            if Message_flag2:
                reward2 = reward2 - self.messageCost * 1

            new_obs = np.array([self.state[0] / (float(self.numberOrder / 1.)) , self.state[1] / (float(self.numberOrder / 1.)), self.state[2], self.state[3]], dtype=np.float)

            return np.array(new_obs, dtype=np.float), (reward1+reward2) / 100., self.done1, self.done2, {}
        
        elif (self.done1) and (not self.done2):
            self._cur_episode_step2 += 1
            tmp_quality2 = math.exp(-self.qualityDeteriorateRate * self._cur_episode_step2) + np.random.normal(0, self.qualitySigma)
            tmp_quality2 = np.clip(tmp_quality2, 0., 1.0)
            self.state[3] = tmp_quality2
            inventoryLevel2 = self.state[1]

            Message_flag2 = False
       
            if np.random.uniform() < msg2:
                self.hist_y2.append(float(tmp_quality2))
                self.hist_x2.append(float(self._cur_episode_step2))
                Message_flag2 = True

            bayes_alpha2 = self.get_bayesian_alpha()
            bayesian_quality2 = math.exp(-bayes_alpha2 * self._cur_episode_step2) + np.random.normal(0, self.qualitySigma)
            bayesian_quality2 = np.clip(bayesian_quality2, 0., 1.0)

            # probability of buying product for each individual customer
            probBuySP2 = math.exp(-(self.priceSensitivity*action2)**self.priceScale)*(1-math.exp(-(self.qualitySensitivity*bayesian_quality2)**self.qualityScale))
            self.hist_alpha2.append(float(bayes_alpha2))
        
            numberSP = np.random.poisson(self.arrivalRate)
            numberBuySP2 = 0

            for i in range(numberSP):
                rn = np.random.uniform()
                if rn < probBuySP2:
                    numberBuySP2 += 1

            if numberBuySP2  < inventoryLevel2: 
                inventoryLevel2 = inventoryLevel2 - numberBuySP2
                self.state[1] = inventoryLevel2
                reward2 = (action2 - self.unitOrderingCost) * numberBuySP2
            else:
                reward2 = (action2 - self.unitOrderingCost) * inventoryLevel2
                inventoryLevel2 = 0
                self.state[1] = inventoryLevel2
            
            self.done2 = inventoryLevel2 == 0  
            if self._cur_episode_step2 >= self._max_episode_steps:
                self.done2 = True
                reward2 = reward2 - (inventoryLevel2 * self.unitOrderingCost)

            if Message_flag2:
                reward2 = reward2 - self.messageCost * 1

            new_obs = np.array([self.state[0] / (float(self.numberOrder / 1.)) , self.state[1] / (float(self.numberOrder / 1.)), self.state[2], self.state[3]], dtype=np.float)

            return np.array(new_obs, dtype=np.float), reward2 / 100., self.done1, self.done2, {}


    def reset(self):
        self.state = np.array([self.numberOrder, self.numberOrder, self.qualityHigh,  self.qualityHigh])
        self._cur_episode_step1 = 0
        self._cur_episode_step2 = 0
        new_obs = np.array([self.state[0] / (float(self.numberOrder / 1.)) , self.state[1] / (float(self.numberOrder / 1.)), self.state[2], self.state[3]], dtype=np.float)
        self.hist_y = [self.state[2]]
        self.hist_y2 = [self.state[3]]
        self.hist_x = [0]
        self.hist_x2[0]
        self.hist_alpha1 = []
        self.hist_alpha2 = []
        return np.array(new_obs, dtype=np.float)

    def render(self, mode = 'human', close = False):
        pass

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

class PricingSimpleEnv(gym.Env):

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
        self.PSarrivalRate = 35 # arrival rate of price-sensitive shoppers per day
        self.QSarrivalRate = 35 # arrival rate of quality-sensitive shoppers per day

        self.PSpriceSensitivity = 0.005
        self.PSpriceScale = 5
        self.PSqualitySensitivity = 20
        self.PSqualityScale = 1

        self.QSpriceSensitivity = 0.003
        self.QSpriceScale = 3
        self.QSqualitySensitivity = 3
        self.QSqualityScale = 1

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
        
        self.action_space = spaces.Box(low = self.priceLow, high = self.priceHigh, shape=(8,), dtype=np.float32)
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

        price1_PS = 150.0 * (np.array(action[0])+ 1.)
        price2_PS = 150.0 * (np.array(action[1])+ 1.)
        price1_QS = 150.0 * (np.array(action[2])+ 1.)
        price2_QS = 150.0 * (np.array(action[3])+ 1.)

        msg1_PS = (action[4] + 1.) / 2.
        msg2_PS = (action[5] + 1.) / 2.
        msg1_QS = (action[6] + 1.) / 2.
        msg2_QS = (action[7] + 1.) / 2.

        ## NOTE: track tiem step for the second batch of products.
        self.done2 = self._cur_episode_step1 < self.TS

        if not self.done1 and self.done2:
            self._cur_episode_step1 += 1
            tmp_quality1 = math.exp(-self.qualityDeteriorateRate * self._cur_episode_step1) + np.random.normal(0, self.qualitySigma)
            tmp_quality1 = np.clip(tmp_quality1, 0., 1.0)
            self.state[2] = tmp_quality1
            inventoryLevel1 = self.state[0]

            Message_flag1_PS = False
       
            if np.random.uniform() < msg1_PS:
                self.hist_y_ps.append(float(tmp_quality1))
                self.hist_x_ps.append(float(self._cur_episode_step1))
                Message_flag1_PS = True

            Message_flag1_QS = False
       
            if np.random.uniform() < msg1_QS:
                self.hist_y_qs.append(float(tmp_quality1))
                self.hist_x_qs.append(float(self._cur_episode_step1))
                Message_flag1_QS = True

            bayes_alpha1_ps = self.get_bayesian_alpha()
            bayesian_quality1_ps = math.exp(-bayes_alpha1_ps * self._cur_episode_step1) + np.random.normal(0, self.qualitySigma)
            bayesian_quality1_ps = np.clip(bayesian_quality1_ps, 0., 1.0)

            # probability of buying product for each individual customer
            PSprobBuySP1 = math.exp(-(self.priceSensitivity*price1_PS)**self.priceScale)*(1-math.exp(-(self.qualitySensitivity*bayesian_quality1_ps)**self.qualityScale))
            self.hist_alpha1_ps.append(float(bayes_alpha1_ps))


            bayes_alpha1_qs = self.get_bayesian_alpha()
            bayesian_quality1_qs = math.exp(-bayes_alpha1_qs * self._cur_episode_step1) + np.random.normal(0, self.qualitySigma)
            bayesian_quality1_qs = np.clip(bayesian_quality1_qs, 0., 1.0)

            # probability of buying product for each individual customer
            QSprobBuySP1 = math.exp(-(self.priceSensitivity*price1_QS)**self.priceScale)*(1-math.exp(-(self.qualitySensitivity*bayesian_quality1_qs)**self.qualityScale))
            self.hist_alpha1_qs.append(float(bayes_alpha1_qs))

            PSnumberSP = np.random.poisson(self.PSarrivalRate)
            QSnumberSP = np.random.poisson(self.QSarrivalRate)
            PSnumberBuySP1 = 0
            QSnumberBuySP1 = 0

            for i in range(PSnumberSP):
                rn = np.random.uniform()
                if rn < PSprobBuySP1:
                    PSnumberBuySP1 += 1

            for i in range(QSnumberSP):
                
                rn = np.random.uniform()
                if rn < QSprobBuySP1:
                    QSnumberBuySP1 += 1

            numberBuySP1 = PSnumberBuySP1 + QSnumberBuySP1
        
            # test whether demand exceeds inventory
            if numberBuySP1  < inventoryLevel1: 
                inventoryLevel1 = inventoryLevel1 - numberBuySP1
                self.state[0] = inventoryLevel1
                reward1 = (price1_PS - self.unitOrderingCost)* PSnumberBuySP1 + (price1_QS - self.unitOrderingCost)* QSnumberBuySP1
            ## NOTE: when there is a inventory shortage, we assume the retailer first sell products to the group with higher prices
            else:
                if price1_PS > price1_QS:
                    if PSnumberBuySP1 < inventoryLevel1:
                        reward1 = (price1_PS - self.unitOrderingCost)* PSnumberBuySP1 + (price1_QS - self.unitOrderingCost)* (inventoryLevel1 - PSnumberBuySP1)
                    else:
                        reward1 = (price1_PS - self.unitOrderingCost)* inventoryLevel1
                else:
                    if QSnumberBuySP1 < inventoryLevel1:
                        reward1 = (price1_PS - self.unitOrderingCost)* (inventoryLevel1 - QSnumberBuySP1) + (price1_QS - self.unitOrderingCost)* QSnumberBuySP1
                    else:
                        reward1 = (price1_QS - self.unitOrderingCost)* inventoryLevel1
                # reward = (price - self.unitOrderingCost) * inventoryLevel 
                inventoryLevel1 = 0
                self.state[0] = inventoryLevel1
            
            self.done1 = inventoryLevel1 == 0  


            if self._cur_episode_step1 >= self._max_episode_steps:
                self.done1 = True
                reward1 = reward1 - (inventoryLevel1 * self.unitOrderingCost)

            if Message_flag1_PS:
                reward1 = reward1 - self.messageCost * 1
            if Message_flag1_QS:
                reward1 = reward1 - self.messageCost * 1

            new_obs = np.array([self.state[0] / (float(self.numberOrder / 1.)) , self.state[1] / (float(self.numberOrder / 1.)), self.state[2], self.state[3]], dtype=np.float)

            return np.array(new_obs, dtype=np.float), reward1 / 100., self.done1, self.done2, {}

        elif (not self.self.done1) and (not self.done2):

            self._cur_episode_step1 += 1
            tmp_quality1 = math.exp(-self.qualityDeteriorateRate * self._cur_episode_step1) + np.random.normal(0, self.qualitySigma)
            tmp_quality1 = np.clip(tmp_quality1, 0., 1.0)
            self.state[2] = tmp_quality1
            inventoryLevel1 = self.state[0]

            Message_flag1_PS = False
       
            if np.random.uniform() < msg1_PS:
                self.hist_y_ps.append(float(tmp_quality1))
                self.hist_x_ps.append(float(self._cur_episode_step1))
                Message_flag1_PS = True

            Message_flag1_QS = False
       
            if np.random.uniform() < msg1_QS:
                self.hist_y_qs.append(float(tmp_quality1))
                self.hist_x_qs.append(float(self._cur_episode_step1))
                Message_flag1_QS = True

            bayes_alpha1_ps = self.get_bayesian_alpha()
            bayesian_quality1_ps = math.exp(-bayes_alpha1_ps * self._cur_episode_step1) + np.random.normal(0, self.qualitySigma)
            bayesian_quality1_ps = np.clip(bayesian_quality1_ps, 0., 1.0)

            # probability of buying product for each individual customer
            PSprobBuySP1 = math.exp(-(self.priceSensitivity*price1_PS)**self.priceScale)*(1-math.exp(-(self.qualitySensitivity*bayesian_quality1_ps)**self.qualityScale))
            self.hist_alpha1_ps.append(float(bayes_alpha1_ps))

            bayes_alpha1_qs = self.get_bayesian_alpha()
            bayesian_quality1_qs = math.exp(-bayes_alpha1_qs * self._cur_episode_step1) + np.random.normal(0, self.qualitySigma)
            bayesian_quality1_qs = np.clip(bayesian_quality1_qs, 0., 1.0)

            # probability of buying product for each individual customer
            QSprobBuySP1 = math.exp(-(self.priceSensitivity*price1_QS)**self.priceScale)*(1-math.exp(-(self.qualitySensitivity*bayesian_quality1_qs)**self.qualityScale))
            self.hist_alpha1_qs.append(float(bayes_alpha1_qs))


            self._cur_episode_step2 += 1
            tmp_quality2 = math.exp(-self.qualityDeteriorateRate * self._cur_episode_step2) + np.random.normal(0, self.qualitySigma)
            tmp_quality2 = np.clip(tmp_quality2, 0., 1.0)
            self.state[3] = tmp_quality2
            inventoryLevel2 = self.state[1]

            Message_flag2_PS = False
       
            if np.random.uniform() < msg2_PS:
                self.hist_y2_ps.append(float(tmp_quality2))
                self.hist_x2_ps.append(float(self._cur_episode_step2))
                Message_flag2_PS = True

            Message_flag2_QS = False
       
            if np.random.uniform() < msg2_QS:
                self.hist_y2_qs.append(float(tmp_quality2))
                self.hist_x2_qs.append(float(self._cur_episode_step2))
                Message_flag2_QS = True

            bayes_alpha2_ps = self.get_bayesian_alpha()
            bayesian_quality2_ps = math.exp(-bayes_alpha2_ps * self._cur_episode_step2) + np.random.normal(0, self.qualitySigma)
            bayesian_quality2_ps = np.clip(bayesian_quality2_ps, 0., 1.0)

            # probability of buying product for each individual customer
            PSprobBuySP2 = math.exp(-(self.priceSensitivity*price2_PS)**self.priceScale)*(1-math.exp(-(self.qualitySensitivity*bayesian_quality2_ps)**self.qualityScale))
            self.hist_alpha2_ps.append(float(bayes_alpha2_ps))

            bayes_alpha2_qs = self.get_bayesian_alpha()
            bayesian_quality2_qs = math.exp(-bayes_alpha2_qs * self._cur_episode_step2) + np.random.normal(0, self.qualitySigma)
            bayesian_quality2_qs = np.clip(bayesian_quality2_qs, 0., 1.0)

            # probability of buying product for each individual customer
            QSprobBuySP2 = math.exp(-(self.priceSensitivity*price2_QS)**self.priceScale)*(1-math.exp(-(self.qualitySensitivity*bayesian_quality2_qs)**self.qualityScale))
            self.hist_alpha2_qs.append(float(bayes_alpha2_qs))
        
            # sample number of shoppers from Poisson process, among those shoppers, some buy apples, others not
            PSnumberSP = np.random.poisson(self.PSarrivalRate)
            QSnumberSP = np.random.poisson(self.QSarrivalRate)
            PSnumberBuySP1 = 0
            QSnumberBuySP1 = 0
            PSnumberBuySP2 = 0
            QSnumberBuySP2 = 0

            
            for i in range(PSnumberSP):
                # probBuySP = (1 - weibull_min.cdf(c = 0.05, x = action[0], scale = 0.1 / float(PS[i]))) * weibull_min.cdf(c = 0.1, x = quality, scale = 1 / float(QS[i]))  
                if PSprobBuySP1 > PSprobBuySP2:
                    rn = np.random.uniform()
                    if rn < PSprobBuySP1:
                        PSnumberBuySP1 += 1
                elif PSprobBuySP1 < PSprobBuySP2:
                    rn = np.random.uniform()
                    if rn < PSprobBuySP2:
                        PSnumberBuySP2 += 1
                else:
                    rn = np.random.uniform()
                    if rn < PSprobBuySP1:
                        if random.choice([0,1]) == 0:
                            PSnumberBuySP1 += 1
                        else:
                            PSnumberBuySP2 += 1

            for i in range(QSnumberSP):
                # probBuySP = (1 - weibull_min.cdf(c = 0.05, x = action[0], scale = 0.1 / float(PS[i]))) * weibull_min.cdf(c = 0.1, x = quality, scale = 1 / float(QS[i]))  
                if QSprobBuySP1 > QSprobBuySP2:
                    rn = np.random.uniform()
                    if rn < QSprobBuySP1:
                        QSnumberBuySP1 += 1
                elif QSprobBuySP1 < QSprobBuySP2:
                    rn = np.random.uniform()
                    if rn < QSprobBuySP2:
                        QSnumberBuySP2 += 1
                else:
                    rn = np.random.uniform()
                    if rn < QSprobBuySP1:
                        if random.choice([0,1]) == 0:
                            QSnumberBuySP1 += 1
                        else:
                            QSnumberBuySP2 += 1

            numberBuySP1 = PSnumberBuySP1 + QSnumberBuySP1
        
            # test whether demand exceeds inventory
            if numberBuySP1  < inventoryLevel1: 
                inventoryLevel1 = inventoryLevel1 - numberBuySP1
                self.state[0] = inventoryLevel1
                reward1 = (price1_PS - self.unitOrderingCost)* PSnumberBuySP1 + (price1_QS - self.unitOrderingCost)* QSnumberBuySP1
            ## NOTE: when there is a inventory shortage, we assume the retailer first sell products to the group with higher prices
            else:
                if price1_PS > price1_QS:
                    if PSnumberBuySP1 < inventoryLevel1:
                        reward1 = (price1_PS - self.unitOrderingCost)* PSnumberBuySP1 + (price1_QS - self.unitOrderingCost)* (inventoryLevel1 - PSnumberBuySP1)
                    else:
                        reward1 = (price1_PS - self.unitOrderingCost)* inventoryLevel1
                else:
                    if QSnumberBuySP1 < inventoryLevel1:
                        reward1 = (price1_PS - self.unitOrderingCost)* (inventoryLevel1 - QSnumberBuySP1) + (price1_QS - self.unitOrderingCost)* QSnumberBuySP1
                    else:
                        reward1 = (price1_QS - self.unitOrderingCost)* inventoryLevel1
                # reward = (price - self.unitOrderingCost) * inventoryLevel 
                inventoryLevel1 = 0
                self.state[0] = inventoryLevel1
            
            self.done1 = inventoryLevel1 == 0  


            if self._cur_episode_step1 >= self._max_episode_steps:
                self.done1 = True
                reward1 = reward1 - (inventoryLevel1 * self.unitOrderingCost)

            if Message_flag1_PS:
                reward1 = reward1 - self.messageCost * 1
            if Message_flag1_QS:
                reward1 = reward1 - self.messageCost * 1

            
            numberBuySP2 = PSnumberBuySP2 + QSnumberBuySP2
        
            # test whether demand exceeds inventory
            if numberBuySP2  < inventoryLevel2: 
                inventoryLevel2 = inventoryLevel2 - numberBuySP2
                self.state[1] = inventoryLevel2
                reward12= (price2_PS - self.unitOrderingCost)* PSnumberBuySP2 + (price2_QS - self.unitOrderingCost)* QSnumberBuySP2
            ## NOTE: when there is a inventory shortage, we assume the retailer first sell products to the group with higher prices
            else:
                if price2_PS > price2_QS:
                    if PSnumberBuySP2 < inventoryLevel2:
                        reward2 = (price2_PS - self.unitOrderingCost)* PSnumberBuySP2 + (price2_QS - self.unitOrderingCost)* (inventoryLevel2 - PSnumberBuySP2)
                    else:
                        reward2 = (price2_PS - self.unitOrderingCost)* inventoryLevel2
                else:
                    if QSnumberBuySP2 < inventoryLevel2:
                        reward2 = (price2_PS - self.unitOrderingCost)* (inventoryLevel2 - QSnumberBuySP2) + (price2_QS - self.unitOrderingCost)* QSnumberBuySP2
                    else:
                        reward2 = (price2_QS - self.unitOrderingCost)* inventoryLevel2
                # reward = (price - self.unitOrderingCost) * inventoryLevel 
                inventoryLevel2 = 0
                self.state[1] = inventoryLevel2
            
            self.done2 = inventoryLevel2 == 0  


            if self._cur_episode_step2 >= self._max_episode_steps:
                self.done2 = True
                reward2 = reward2 - (inventoryLevel2 * self.unitOrderingCost)

            if Message_flag1_PS:
                reward2 = reward2 - self.messageCost * 1
            if Message_flag1_QS:
                reward2 = reward2 - self.messageCost * 1

            new_obs = np.array([self.state[0] / (float(self.numberOrder / 1.)) , self.state[1] / (float(self.numberOrder / 1.)), self.state[2], self.state[3]], dtype=np.float)

            return np.array(new_obs, dtype=np.float), (reward1+reward2) / 100., self.done1, self.done2, {}
        
        elif (self.done1) and (not self.done2):
            self._cur_episode_step2 += 1
            tmp_quality2 = math.exp(-self.qualityDeteriorateRate * self._cur_episode_step2) + np.random.normal(0, self.qualitySigma)
            tmp_quality2 = np.clip(tmp_quality2, 0., 1.0)
            self.state[3] = tmp_quality2
            inventoryLevel2 = self.state[1]

            Message_flag2_PS = False
       
            if np.random.uniform() < msg2_PS:
                self.hist_y2_ps.append(float(tmp_quality2))
                self.hist_x2_ps.append(float(self._cur_episode_step2))
                Message_flag2_PS = True

            Message_flag2_QS = False
       
            if np.random.uniform() < msg2_QS:
                self.hist_y2_qs.append(float(tmp_quality2))
                self.hist_x2_qs.append(float(self._cur_episode_step2))
                Message_flag2_QS = True

            bayes_alpha2_ps = self.get_bayesian_alpha()
            bayesian_quality2_ps = math.exp(-bayes_alpha2_ps * self._cur_episode_step2) + np.random.normal(0, self.qualitySigma)
            bayesian_quality2_ps = np.clip(bayesian_quality2_ps, 0., 1.0)

            # probability of buying product for each individual customer
            PSprobBuySP2 = math.exp(-(self.priceSensitivity*price2_PS)**self.priceScale)*(1-math.exp(-(self.qualitySensitivity*bayesian_quality2_ps)**self.qualityScale))
            self.hist_alpha2_ps.append(float(bayes_alpha2_ps))


            bayes_alpha2_qs = self.get_bayesian_alpha()
            bayesian_quality2_qs = math.exp(-bayes_alpha2_qs * self._cur_episode_step2) + np.random.normal(0, self.qualitySigma)
            bayesian_quality2_qs = np.clip(bayesian_quality2_qs, 0., 1.0)

            # probability of buying product for each individual customer
            QSprobBuySP2 = math.exp(-(self.priceSensitivity*price2_QS)**self.priceScale)*(1-math.exp(-(self.qualitySensitivity*bayesian_quality2_qs)**self.qualityScale))
            self.hist_alpha2_qs.append(float(bayes_alpha2_qs))

            PSnumberSP = np.random.poisson(self.PSarrivalRate)
            QSnumberSP = np.random.poisson(self.QSarrivalRate)
            PSnumberBuySP2 = 0
            QSnumberBuySP2 = 0

            for i in range(PSnumberSP):
                rn = np.random.uniform()
                if rn < PSprobBuySP2:
                    PSnumberBuySP2 += 1

            for i in range(QSnumberSP):
                
                rn = np.random.uniform()
                if rn < QSprobBuySP2:
                    QSnumberBuySP2 += 1

            numberBuySP2 = PSnumberBuySP2 + QSnumberBuySP2
        
            # test whether demand exceeds inventory
            if numberBuySP2  < inventoryLevel2: 
                inventoryLevel2 = inventoryLevel2 - numberBuySP2
                self.state[1] = inventoryLevel2
                reward2 = (price2_PS - self.unitOrderingCost)* PSnumberBuySP2 + (price2_QS - self.unitOrderingCost)* QSnumberBuySP2
            ## NOTE: when there is a inventory shortage, we assume the retailer first sell products to the group with higher prices
            else:
                if price2_PS > price2_QS:
                    if PSnumberBuySP2 < inventoryLevel2:
                        reward2 = (price2_PS - self.unitOrderingCost)* PSnumberBuySP2 + (price2_QS - self.unitOrderingCost)* (inventoryLevel2 - PSnumberBuySP2)
                    else:
                        reward2 = (price2_PS - self.unitOrderingCost)* inventoryLevel2
                else:
                    if QSnumberBuySP2 < inventoryLevel2:
                        reward2 = (price2_PS - self.unitOrderingCost)* (inventoryLevel2 - QSnumberBuySP2) + (price2_QS - self.unitOrderingCost)* QSnumberBuySP2
                    else:
                        reward2 = (price2_QS - self.unitOrderingCost)* inventoryLevel2
                # reward = (price - self.unitOrderingCost) * inventoryLevel 
                inventoryLevel2 = 0
                self.state[1] = inventoryLevel2
            
            self.done2 = inventoryLevel2 == 0  


            if self._cur_episode_step2 >= self._max_episode_steps:
                self.done2 = True
                reward2 = reward2 - (inventoryLevel2 * self.unitOrderingCost)

            if Message_flag2_PS:
                reward2 = reward2 - self.messageCost * 1
            if Message_flag2_QS:
                reward2 = reward2 - self.messageCost * 1

            new_obs = np.array([self.state[0] / (float(self.numberOrder / 1.)) , self.state[1] / (float(self.numberOrder / 1.)), self.state[2], self.state[3]], dtype=np.float)

            return np.array(new_obs, dtype=np.float), reward2 / 100., self.done1, self.done2, {}


    def reset(self):
        self.state = np.array([self.numberOrder, self.numberOrder, self.qualityHigh,  self.qualityHigh])
        self._cur_episode_step1 = 0
        self._cur_episode_step2 = 0
        new_obs = np.array([self.state[0] / (float(self.numberOrder / 1.)) , self.state[1] / (float(self.numberOrder / 1.)), self.state[2], self.state[3]], dtype=np.float)
        self.hist_y_ps = [self.state[2]]
        self.hist_y_qs = [self.state[2]]
        self.hist_y2_ps = [self.state[3]]
        self.hist_y2_qs = [self.state[3]]
        self.hist_x_ps = [0]
        self.hist_x_qs = [0]
        self.hist_x2_ps = [0]
        self.hist_x2_qs = [0]
        self.hist_alpha1_ps = []
        self.hist_alpha2_ps = []
        self.hist_alpha1_qs = []
        self.hist_alpha2_qs = []
        return np.array(new_obs, dtype=np.float)

    def render(self, mode = 'human', close = False):
        pass
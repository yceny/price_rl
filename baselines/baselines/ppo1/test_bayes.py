import numpy as np
import pdb 

'''

'''


def dnlogp_grad(data_x, data_y, c_alpha, ratio_sigma2, mu_0):
    XY = (data_y - np.exp( - data_x * c_alpha)) * data_x *  np.exp( - data_x * c_alpha)
    prior = ratio_sigma2 * (c_alpha - mu_0)
    return 2. * (np.sum(XY) + prior)
    

def nlogp_loss(data_x, data_y, c_alpha, ratio_sigma2, mu_0):
    XY =  (data_y - np.exp( - data_x * c_alpha)) ** 2 
    prior = ratio_sigma2 *((c_alpha - mu_0) ** 2)
    return np.sum(XY) + prior

def perform_SGD(data_x, data_y, init_alpha, ratio_sigma2, mu_0, lr=1e-3):
    c_alpha = init_alpha
    loss = nlogp_loss(data_x, data_y, c_alpha, ratio_sigma2, mu_0)
    print(len(data_x), c_alpha, loss)
    for tt in range(100):
        c_alpha -= lr * dnlogp_grad(data_x, data_y, c_alpha, ratio_sigma2, mu_0)
        loss = nlogp_loss(data_x, data_y, c_alpha, ratio_sigma2, mu_0)
    print(len(data_x), c_alpha, loss)
    return c_alpha

def get_observed_data(T=21, alpha=0.10, sigma=0.02):
    data_x = []
    data_y = []
    for t in range(T):
        x = t 
        noise = np.random.normal(0, sigma)
        tmp = -alpha * t
        y = np.clip(np.exp(tmp) + noise, 0., 1.0)
        data_y.append(y)
        data_x.append(x)
    return np.array(data_x), np.array(data_y) 

sigma_0 = 0.0002
mu_0 = 0.12
sigma = 0.0005
TTT = 22
estimated_alpha = []
for TT in range(1, TTT):
    data_x, data_y = get_observed_data(TT, alpha=0.10, sigma=sigma)
    
    data_x = np.array(data_x)
    data_y = np.array(data_y)
    init_alpha = mu_0
    ratio_sigma2 = (sigma / sigma_0) ** 2
    c_alpha = perform_SGD(data_x, data_y, init_alpha, ratio_sigma2, mu_0, lr=3e-3)
    print("-----------------------------------------------------------")
    print("")
    # pdb.set_trace()
    # optimal_alpha = (- XY  +  ratio_sigma_2* mu_0) / (XX + ratio_sigma_2)
    # print(optimal_alpha)
    estimated_alpha.append(c_alpha)


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 
xx = np.arange(1, TTT)
plt.plot(xx, estimated_alpha, lw=5)
plt.plot(xx, [mu_0] * (TTT-1), lw=5, ls='--')
plt.plot(xx, [0.10] * (TTT-1), lw=5, ls='--')
plt.xticks(np.arange(1, TTT, 2), fontsize=20)
# plt.yticks([0.1, 0.11, 0.12, 0.13, 0.14, 0.15], fontsize=20)
plt.xlabel("Step T", fontsize=18)
plt.title(r'$\alpha^{*}$')
plt.savefig('bayes_alpha.pdf', bbox_inches='tight')









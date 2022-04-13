from baselines.common.cmd_util import make_mujoco_env
from baselines.common import tf_util as U
import os.path as osp 
import tensorflow as tf 
import numpy as np 
import time
import glob
import gym
import pdb
import os
import pandas as pd
import gym_simple_pricing
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 
import warnings

warnings.filterwarnings("ignore")


def get_spaces(env):
    if not isinstance(env.observation_space, gym.spaces.Box): 
        ob_space = gym.spaces.Box(env.observation_space.low, env.observation_space.high)
        if hasattr(env.action_space, 'shape'):
            ac_space = gym.spaces.Box(env.action_space.low, env.action_space.high)
        else:
            ac_space = gym.spaces.Discrete(env.action_space.n)
    else:
        ob_space =  env.observation_space
        ac_space = env.action_space
    
    return ob_space, ac_space

def construct(env_id, seed, log_dir):
    import mlp_policy, pposgd_simple
    sess = U.make_session(num_cpu=1)
    sess.__enter__()
    def policy_fn(name, ob_space, ac_space):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=32, num_hid_layers=2)
     
    
    env = make_mujoco_env(env_id, seed)
    ob_space, ac_space = get_spaces(env)

    pi = policy_fn("pi", ob_space, ac_space) # Construct network for new policy
    U.initialize()
    
    
    var_list = pi.get_variables()
    
    saver = tf.train.Saver(var_list=var_list)
    # load from models
    saver.restore(sess, log_dir+'/best/models-')
    
    return env, pi

if __name__ == '__main__':
    seed = 0
    # env_id = 'pricing-simple-v1'
    # env_id = 'pricing-simple-v3'
    # env_id = 'pricing-ext-v2'
    # env_id = 'pricing-ext-v2-v3'
    # env_id = 'pricing-ext-v1'
    # env_id = 'pricing-ext-v1-v3'
    # env_id = 'pricing-base-v2'
    # env_id = 'pricing-base-v3'
    # env_id = 'pricing-base-v9'
    # env_id = 'pricing-model1-v1'
    env_id = 'pricing-model2-v9'
    # env_id = 'pricing-model1-v9'
    # env_id = 'pricing-model2-v1'
    # env_id = 'pricing-model2-v2'
    # env_id = 'pricing-model1-v2'
    # env_id = 'pricing-model1-v3'
    # env_id = 'pricing-model3-v1'
    # env_id = 'pricing-base-v1'
    # env_id = 'pricing-simple-v3'
    # env_id = 'pricing-ext-v4'
    # env_id = 'pricing-ext-v5'
    # env_id = 'pricing-ext-v5-v3'
    # env_id = 'pricing-ext-v3'
    # env_id = 'pricing-ext-v9'
    # env_id = 'pricing-base-v8'
    # env_id = 'pricing-model1-v8'
    # env_id = 'pricing-model2-v8'

    pol_type = 'mlp'
    # cost = 1000
    cost=500

    PSpriceSensitivity = 0.005
    PSpriceScale = 5
    PSqualitySensitivity = 5
    PSqualityScale = 1
    
    QSpriceSensitivity = 0.003
    QSpriceScale = 3
    QSqualitySensitivity = 3
    QSqualityScale = 1

    #if 'ext-v2' in env_id:
    #    pdir = osp.join('TRY_logs', 'env=%s-c-%d'%(env_id, cost), 'seed=None_*')
    #else:
    pdir = osp.join('TRY_logs_mixed_prior_cost', 'env=%s'%(env_id), 'seed=None_*')
    
    print('###########################')
    print(pdir)
    print('###########################')
    #logdir = glob.glob(pdir)[0]

    print(sorted(glob.glob(pdir), key=os.path.getmtime, reverse = True))
    logdir = sorted(glob.glob(pdir), key=os.path.getmtime, reverse = True)[0]

    # logdir = osp.join('TRY_logs_mixed_prior', 'env=%s'%(env_id), 'seed=None_0.05_0.1_0.15_25_25_50')

    print('########## final directory ##############')
    print(logdir)
    print('########## final directory ##############')
    

    # pdb.set_trace()
    
    num_steps = int(30000)
    # num_steps = int(300)
    env, pi = construct(env_id,  1, logdir)
    # env.messageCost = cost
    obs_dim = env.observation_space.shape[0]
    acts_dim = env.action_space.shape[0]
    # print('action dimension is ', acts_dim)
    T = 12

    ob = env.reset()
    # print('start observation ', ob)
    traj_rewards = []
    traj_acts = []
    total_rewards = []
    # total_trajs = []
    total_trajs1 = np.zeros([num_steps, T])
    total_trajs2 = np.zeros([num_steps, T])
    total_trajs3 = np.zeros([num_steps, T])
    total_trajs4 = np.zeros([num_steps, T])
    traj_obs = [ob]
    # total_obs = []
    total_obs1 = np.zeros([num_steps, T+1])
    total_obs2 = np.zeros([num_steps, T+1])
    total_alpha1 = []
    total_alpha2 = []
    total_alpha3 = []
    total_alpha_ps = []
    total_alpha_qs = []
    episode_rews = 0
    # env_id += '_cost='+str(cost)
    total_all_rewards = []
    total_final_obs = []
    total_final_days = []
    psbuyer = []
    qsbuyer = []
    buyer = []
    traj_psbuyer = np.zeros([num_steps, T])
    traj_qsbuyer = np.zeros([num_steps, T])
    traj_buyer = np.zeros([num_steps, T])
    # print(logdir)
    quality = []
    for tt in range(1, num_steps):
        ac, _ = pi.act(False, ob)

        # brute force enumerate ac to get fixed pricing strategy
        
        # ac = 30
        ob, r, done, info = env.step(ac)
        episode_rews += r * 100
        traj_rewards.append(episode_rews)
        traj_acts.append(ac)
        traj_obs.append(ob)
        # quality.append(info['quality'])

        # if ('v3' in env_id) or ('ext-v4' in env_id):
            # print('# PS buyer ', info['PSbuyer'])
            # print('# QS buyer ', info['QSbuyer'])
        if 'v9' in env_id or 'v8' in env_id:
            buyer.append(info['buyer'])

        else:
            psbuyer.append(info['PSbuyer'])
            qsbuyer.append(info['QSbuyer'])


        if done :
            # print("Finished one trajectory with quality ", quality)
            # print("Finished one trajectory with reward=%f"%(traj_rewards[-1]))
            # print("length=%d"%len(traj_rewards))
            total_all_rewards.append(traj_rewards[-1])
            
            traj_acts_len = len(traj_acts)
            if traj_acts_len < T:
                for _ in range(T - traj_acts_len):
                    traj_acts.append([np.nan]*len(ac))

            
            #print('############ traj_acts length is ##############', len(traj_acts))
            arr_traj = np.array(traj_acts).reshape(-1, acts_dim)
            # print('############ traj_acts shape is ##############', arr_traj.shape)
            # arr_traj[:, 0] = np.clip((arr_traj[:, 0] + 3.) * 75, 150, 300)
            

            arr_traj[:, 0] = np.clip((arr_traj[:, 0] + 1.) * 150, 0, 300)
            #  arr_traj[:, 0] = np.clip((arr_traj[:, 0] + 1.) * 2.5, 0, 5)
            total_trajs1[tt,:] = arr_traj[:,0]

            if arr_traj.shape[1] > 1:
                arr_traj[:, 1] = (arr_traj[:, 1] + 1.)  / 2.
                total_trajs2[tt,:] = arr_traj[:, 1]
                
            if arr_traj.shape[1] > 2:
                arr_traj[:, 2] = (arr_traj[:, 2] + 1.)  / 2.
                total_trajs3[tt,:] = arr_traj[:, 2]

            if 'v9' in env_id or 'v8' in env_id:
                traj_buyer_len = len(buyer)
                if traj_buyer_len < T:
                    buyer += [np.nan] * (T - traj_buyer_len)

                traj_buyer[tt,:] = buyer

            else:
                
            # if ('v3' in env_id) or ('ext-v4' in env_id):
                traj_buyer_len = len(psbuyer)
                if traj_buyer_len < T:
                    psbuyer += [np.nan] * (T - traj_buyer_len)
                    qsbuyer += [np.nan] * (T - traj_buyer_len)

                # print('buyer trajectory ', psbuyer)
                traj_psbuyer[tt,:] = psbuyer
                traj_qsbuyer[tt,:] = qsbuyer
                

            # total_trajs.append(arr_traj)
            
            traj_obs_len = len(traj_obs)
            if traj_obs_len < T+1:
                for _ in range(T+1 - traj_obs_len):
                    traj_obs.append([np.nan]*len(ob))
            
            # print('traj_obs ', traj_obs)
            traj_obs = np.array(traj_obs).reshape(-1, obs_dim)
            traj_obs[:, 0] = traj_obs[:, 0] * 500
            # print('traj_obs after transformation ', traj_obs)

            # total_obs.append(traj_obs.tolist())
            total_obs1[tt,:] = traj_obs[:, 0]
            
            if traj_obs.shape[1] > 1:
                total_obs2[tt,:] = traj_obs[:, 1]
                


            total_rewards.append(np.array(traj_rewards))
            total_final_obs.append(np.array(traj_obs[-1]))
            total_final_days.append(len(traj_rewards))
            if hasattr(env.env, 'hist_alpha1'):
                # print('########## YES HIST_ALPHA ###################')
                if len(env.env.hist_alpha1) < T:
                    env.env.hist_alpha1 += [np.nan] * (T - len(env.env.hist_alpha1))
                total_alpha1.append(np.array(env.env.hist_alpha1))

            if hasattr(env.env, 'hist_alpha2'):
                # print('########## YES HIST_ALPHA ###################')
                if len(env.env.hist_alpha2) < T:
                    env.env.hist_alpha2 += [np.nan] * (T - len(env.env.hist_alpha2))
                total_alpha2.append(np.array(env.env.hist_alpha2))

            if hasattr(env.env, 'hist_alpha3'):
                # print('########## YES HIST_ALPHA ###################')
                if len(env.env.hist_alpha3) < T:
                    env.env.hist_alpha3 += [np.nan] * (T - len(env.env.hist_alpha3))
                total_alpha3.append(np.array(env.env.hist_alpha3))

            if hasattr(env.env, 'hist_alpha_qs'):
                if len(env.env.hist_alpha_qs) < T:
                    env.env.hist_alpha_qs += [np.nan] * (T - len(env.env.hist_alpha_qs))
                total_alpha_qs.append(np.array(env.env.hist_alpha_qs))
            
            if hasattr(env.env, 'hist_alpha_ps'):
               
                if len(env.env.hist_alpha_ps) < T:
                    env.env.hist_alpha_ps += [np.nan] * (T - len(env.env.hist_alpha_ps))
                total_alpha_ps.append(np.array(env.env.hist_alpha_ps))
            

            traj_rewards = []
            traj_acts = []
            psbuyer = []
            qsbuyer = []
            buyer = []
            # pdb.set_trace()
            episode_rews = 0
            ob = env.reset()
            traj_obs = [ob]
            quality = []
            # import pdb; pdb.set_trace()
    
    total_all_rewards = np.array(total_all_rewards)
    total_final_days = np.array(total_final_days)
    total_final_obs = np.array(total_final_obs)
    total_alpha1 = np.array(total_alpha1)
    total_alpha2 = np.array(total_alpha2)
    total_alpha3 = np.array(total_alpha3)
    # remove all 0 in total_trajs and total_obs
    total_trajs1 = total_trajs1[~np.all(total_trajs1 == 0, axis=1)]
    # print('total shape of alpha1', total_alpha1.shape)
    # print('total shape of alpha2', total_alpha2.shape)
    # print('total shape of alpha3', total_alpha3.shape)
    # pdb.set_trace()
    print('shape of total_trajs1 ', total_trajs1.shape)
    print('mean of price ', np.nanmean(total_trajs1, axis = 0))
    if acts_dim > 1:
        total_trajs2 = total_trajs2[~np.all(total_trajs2 == 0, axis=1)]
        print('shape of total_trajs2 ', total_trajs2.shape)
        print('mean of total_trajs2 ', np.nanmean(total_trajs2, axis = 0))
    if acts_dim > 2:
        total_trajs3 = total_trajs3[~np.all(total_trajs3 == 0, axis=1)]
        print('shape of total_trajs3 ', total_trajs3.shape)
        print('mean of total_trajs3 ', np.nanmean(total_trajs3, axis = 0))
    if acts_dim > 3:
        total_trajs4 = total_trajs4[~np.all(total_trajs4 == 0, axis=1)]
        print('shape of total_trajs4 ', total_trajs4.shape)
        print('mean of total_trajs4 ', np.nanmean(total_trajs4, axis = 0))
    
    total_obs1 = total_obs1[~np.all(total_obs1 == 0, axis=1)]
    print('shape of inventory ', total_obs1.shape)
    # print('total_obs1 ', total_obs1)
    print('mean of inventory ', np.nanmean(total_obs1, axis = 0))
    if obs_dim > 1:
        total_obs2 = total_obs2[~np.all(total_obs2 == 0, axis=1)]
        print('shape of quality ', total_obs2.shape)
        # print('total_obs2 ', total_obs2)
        print('mean of quality ', np.nanmean(total_obs2, axis = 0))

    if 'v9' in env_id or 'v8' in env_id:
        traj_buyer = traj_buyer[~np.all(traj_buyer == 0, axis = 1)]
        print('shape of np array of buyer trajectory ', traj_buyer.shape)
        print('mean of np array of buyer trajectory ', np.nanmean(traj_buyer, axis = 0))
        print("All buyers Total = %.3f"%(np.sum(np.nanmean(traj_buyer, axis = 0))))
    else:
    # if ('v3' in env_id) or ('ext-v9' in env_id):
        traj_psbuyer = traj_psbuyer[~np.all(traj_psbuyer == 0, axis = 1)]
        traj_qsbuyer = traj_qsbuyer[~np.all(traj_qsbuyer == 0, axis = 1)]
        print('shape of np array of buyer trajectory ', traj_psbuyer.shape)
        print('mean of np array of ps buyer trajectory ', np.nanmean(traj_psbuyer, axis = 0))
        print('mean of np array of qs buyer trajectory ', np.nanmean(traj_qsbuyer, axis = 0))
        print("All QS buyers Total = %.3f"%(np.sum(np.nanmean(traj_qsbuyer, axis = 0))))
        print("All PS buyers Total = %.3f"%(np.sum(np.nanmean(traj_psbuyer, axis = 0))))

    print('mean of np array of alpha1 trajectory ', np.nanmean(total_alpha1, axis = 0))
    print('mean of np array of alpha2 trajectory ', np.nanmean(total_alpha2, axis = 0))
    print('mean of np array of alpha3 trajectory ', np.nanmean(total_alpha3, axis = 0))


    print("All Rewards Mean = %.3f, std = %.3f"%(total_all_rewards.mean(), total_all_rewards.std()))
    print("All Inventory Mean = %.3f, std = %.3f"%(np.nanmean(total_final_obs,axis=0)[0], np.nanstd(total_final_obs,axis=0)[0]))
    print("All Period Mean = %.3f, std = %.3f"%(total_final_days.mean(), total_final_days.std()))
    
    
    pdb.set_trace()
    print("Hello World")
    '''
    Needed Index: ext-3 : 143
    ext-1: 8
    simple: 9
    '''
    all_idxs = [8]
    #simple : -1
    # ext1: 7 or 6
    #base: -3
    # selected_list = np.nanmean(np.array(acts_list), axis = 1)
    # selected_rewards = [total_rewards[idx] for idx in all_idxs]
    # selected_obs = np.nanmean(np.array(total_obs),axis = 1)
    xlim = T + 0.5
    # plots things 
    plt.clf()
    # plot the first dimension
    # for idy in range(acts_list[0].shape[1]):
    # print('action dimension is ', acts_dim)
    for idy in range(acts_dim):
        plt.clf()
        plt.rc('xtick',labelsize=20)
        plt.rc('ytick',labelsize=20)
        
        if idy == 0:
            plt.errorbar(np.arange(np.nanmean(total_trajs1, axis = 0).shape[0]), np.nanmean(total_trajs1, axis = 0), np.nanstd(total_trajs1, axis = 0), alpha=.9, ls='-', lw=4, marker='d',  markersize=10)
             
            plt.legend()

        if idy >= 1:
            # for idx, act in enumerate(selected_list):
            plt.errorbar(np.arange(np.nanmean(total_trajs2, axis = 0).shape[0]), np.nanmean(total_trajs2, axis = 0), np.nanstd(total_trajs2, axis = 0), alpha=.9, ls='-', lw=4, marker='d',  markersize=10)
            
            if idy == 2:
                # for idx, act in enumerate(selected_list):
                plt.errorbar(np.arange(np.nanmean(total_trajs3, axis = 0).shape[0]), np.nanmean(total_trajs3, axis = 0), np.nanstd(total_trajs3, axis = 0), alpha=.9, ls='-', lw=4, marker='d',  markersize=10,  c='g', label='Quality Sensitive')

            plt.legend()

            
        plt.xticks(np.arange(0, 16, 2), np.arange(1, 17, 2))
        plt.xlabel('Days', fontsize=18)
        plt.xlim([0, xlim])
        # plt.title('The %d dimension of Action'%(idy+1))
        if 'ext-v9' in env_id:
            plt.savefig('final_figures_mixed_prior_cost/{}_num_step={}_act={}_ps={}_{}_qs={}_{}_psc={}_{}_qsc={}_{}.pdf'.format(env_id, num_steps, idy, PSpriceSensitivity, QSpriceSensitivity, PSqualitySensitivity, QSqualitySensitivity, PSpriceScale, QSpriceScale, PSqualityScale, QSqualityScale), bbox_inches='tight')
        else:
            plt.savefig('final_figures_mixed_prior_cost/%s_num_step=%d_act=%d.pdf'%(env_id, num_steps, idy), bbox_inches='tight')
        plt.clf()
    
    
    # for idy in range(total_obs[0].shape[1]):
    for idy in range(obs_dim):
        plt.clf()
        plt.rc('xtick',labelsize=20)
        plt.rc('ytick',labelsize=20)
        # for idx, obs in enumerate(selected_obs):
        if idy == 0:
            plt.plot(np.arange(np.nanmean(total_obs1, axis = 0).shape[0]), np.nanmean(total_obs1, axis = 0), alpha=.9, ls='-', lw=4, marker='d', markersize=10)
        if idy == 1:
            plt.plot(np.arange(np.nanmean(total_obs2, axis = 0).shape[0]), np.nanmean(total_obs2, axis = 0), alpha=.9, ls='-', lw=4, marker='d', markersize=10)
        plt.xticks(np.arange(0, 16, 2), np.arange(1, 17, 2))
        plt.xlabel('Days', fontsize=18)
        plt.xlim([0, xlim])
        if idy == 0:
            plt.ylim([-10, 500])
        else:
            plt.ylim([0, 1])
        # plt.title('The %d dimension of Action'%(idy+1))
        if 'ext-v9' in env_id:
            plt.savefig('final_figures_mixed_prior_cost/{}_num_step={}_obs={}_ps={}_{}_qs={}_{}_psc={}_{}_qsc={}_{}.pdf'.format(env_id, num_steps, idy, PSpriceSensitivity, QSpriceSensitivity, PSqualitySensitivity, QSqualitySensitivity, PSpriceScale, QSpriceScale, PSqualityScale, QSqualityScale), bbox_inches='tight')
        else:
            plt.savefig('final_figures_mixed_prior_cost/%s_num_step=%d_obs=%d.pdf'%(env_id, num_steps, idy), bbox_inches='tight')
        plt.clf()

    plt.clf()
    plt.rc('xtick',labelsize=20)
    plt.rc('ytick',labelsize=20)
    if 'v9' in env_id or 'v8' in env_id:
        plt.plot(np.arange(np.nanmean(traj_buyer, axis = 0).shape[0]), np.nanmean(traj_buyer, axis = 0), alpha=.9, ls='-', lw=4, marker='d', markersize=10)
    else:
        plt.plot(np.arange(np.nanmean(traj_psbuyer, axis = 0).shape[0]), np.nanmean(traj_psbuyer, axis = 0), alpha=.9, ls='-', lw=4, marker='d', markersize=10, label = 'Price Sensitive')
        plt.plot(np.arange(np.nanmean(traj_qsbuyer, axis = 0).shape[0]), np.nanmean(traj_qsbuyer, axis = 0), alpha=.9, ls='-', lw=4, marker='d', markersize=10, label = 'Quality Sensitive')
    plt.legend()
    plt.xticks(np.arange(0, 16, 2), np.arange(1, 17, 2))
    plt.xlabel('Days', fontsize=18)
    plt.xlim([0, xlim])
    # plt.ylim([0, 35])
    plt.ylim([0, 40])
    # plt.title('The %d dimension of Action'%(idy+1))
    if 'ext-v9' in env_id:
        plt.savefig('final_figures_mixed_prior_cost/{}_num_step={}_buyer_ps={}_{}_qs={}_{}_psc={}_{}_qsc={}_{}.pdf'.format(env_id, num_steps, PSpriceSensitivity, QSpriceSensitivity, PSqualitySensitivity, QSqualitySensitivity, PSpriceScale, QSpriceScale, PSqualityScale, QSqualityScale), bbox_inches='tight')
    else:
        plt.savefig('final_figures_mixed_prior_cost/%s_num_step=%d_buyer.pdf'%(env_id, num_steps), bbox_inches='tight')
    plt.clf()
    
    # if hasattr(env.env, 'hist_alpha'): 
    if hasattr(env.env, 'hist_alpha1'): 
        # print('############## YES HIST_ALPHA ##################')
        # selected_alpha = [total_alpha[idx] for idx in all_idxs]
        plt.clf()
        plt.rc('xtick',labelsize=20)
        plt.rc('ytick',labelsize=20)
        # for idx, alphas in enumerate(selected_alpha):
        #     plt.plot(np.arange(alphas.shape[0]), alphas, alpha=.9, ls='-', lw=4, marker='d', markersize=10)
        
        plt.plot(np.arange(np.nanmean(np.array(total_alpha1), axis = 0).shape[0]), np.nanmean(np.array(total_alpha1), axis = 0), alpha=.9, ls='-', lw=4, marker='d', markersize=10)
        plt.plot(np.arange(np.nanmean(np.array(total_alpha2), axis = 0).shape[0]), np.nanmean(np.array(total_alpha2), axis = 0), alpha=.9, ls='-', lw=4, marker='d', markersize=10)
        plt.plot(np.arange(np.nanmean(np.array(total_alpha3), axis = 0).shape[0]), np.nanmean(np.array(total_alpha3), axis = 0), alpha=.9, ls='-', lw=4, marker='d', markersize=10)
        plt.xticks(np.arange(0, 16, 2), np.arange(1, 17, 2))
        plt.xlabel('Days', fontsize=18)
        plt.xlim([0, xlim])
        plt.ylim([0.05, 0.153])
        # plt.title('The %d dimension of Action'%(idy+1))
        if 'ext-v9' in env_id:
            plt.savefig('final_figures_mixed_prior_cost/{}_num_step={}_alpha_ps={}_{}_qs={}_{}_psc={}_{}_qsc={}_{}.pdf'.format(env_id, num_steps, PSpriceSensitivity, QSpriceSensitivity, PSqualitySensitivity, QSqualitySensitivity, PSpriceScale, QSpriceScale, PSqualityScale, QSqualityScale), bbox_inches='tight')
        else:
            plt.savefig('final_figures_mixed_prior_cost/%s_num_step=%d_alpha.pdf'%(env_id, num_steps), bbox_inches='tight')
        plt.clf()

    
    # if ('v3' in env_id) or ('ext-v9' in env_id):
    


    
    if hasattr(env.env, 'hist_alpha_qs'): 
    
        # selected_alpha_qs = [total_alpha_qs[idx] for idx in all_idxs]
        plt.clf()
        plt.rc('xtick',labelsize=20)
        plt.rc('ytick',labelsize=20)
        # for idx, alphas in enumerate(selected_alpha_qs):
        #     plt.plot(np.arange(alphas.shape[0]), alphas, alpha=.9, ls='-', lw=4, marker='d', markersize=10)
        plt.plot(np.arange(np.nanmean(np.array(total_alpha_qs), axis = 0).shape[0]), np.nanmean(np.array(total_alpha_qs), axis = 0), alpha=.9, ls='-', lw=4, marker='d', markersize=10)
        plt.xticks(np.arange(0, 16, 2), np.arange(1, 17, 2))
        plt.xlabel('Days', fontsize=18)
        plt.xlim([0, xlim])
        plt.ylim([0.10, 0.153])
        # plt.title('The %d dimension of Action'%(idy+1))
        if 'ext-v9' in env_id:
            plt.savefig('final_figures_mixed_prior_cost/{}_num_step={}_alpha_qs_ps={}_{}_qs={}_{}_psc={}_{}_qsc={}_{}.pdf'.format(env_id, num_steps, PSpriceSensitivity, QSpriceSensitivity, PSqualitySensitivity, QSqualitySensitivity, PSpriceScale, QSpriceScale, PSqualityScale, QSqualityScale), bbox_inches='tight')
        else:
            plt.savefig('final_figures_mixed_prior_cost/%s_num_step=%d_alpha_qs.pdf'%(env_id, num_steps), bbox_inches='tight')
        plt.clf()
    
    if hasattr(env.env, 'hist_alpha_ps'): 
        # selected_alpha_ps = [total_alpha_ps[idx] for idx in all_idxs]
        plt.clf()
        plt.rc('xtick',labelsize=20)
        plt.rc('ytick',labelsize=20)
        # for idx, alphas in enumerate(selected_alpha_ps):
        #    plt.plot(np.arange(alphas.shape[0]), alphas, alpha=.9, ls='-', lw=4, marker='d', markersize=10, c='r', label='Price Sensitive')
        plt.plot(np.arange(np.nanmean(np.array(total_alpha_ps), axis = 0).shape[0]), np.nanmean(np.array(total_alpha_ps), axis = 0), alpha=.9, ls='-', lw=4, marker='d', markersize=10)
        
        plt.xticks(np.arange(0, 16, 2), np.arange(1, 17, 2))
        plt.xlabel('Days', fontsize=18)
        plt.xlim([0, xlim])
        plt.ylim([0.10, 0.153])
        # plt.title('The %d dimension of Action'%(idy+1))
        if env_id == 'pricing-ext-v9':
            plt.savefig('final_figures_mixed_prior_cost/{}_num_step={}_alpha_ps_ps={}_{}_qs={}_{}_psc={}_{}_qsc={}_{}.pdf'.format(env_id, num_steps, PSpriceSensitivity, QSpriceSensitivity, PSqualitySensitivity, QSqualitySensitivity, PSpriceScale, QSpriceScale, PSqualityScale, QSqualityScale), bbox_inches='tight')
        else:
            plt.savefig('final_figures_mixed_prior_cost/%s_num_step=%d_alpha_ps.pdf'%(env_id, num_steps), bbox_inches='tight')
        plt.clf()

        plt.rc('xtick',labelsize=20)
        plt.rc('ytick',labelsize=20)

        plt.plot(np.arange(np.nanmean(np.array(total_alpha_ps), axis = 0).shape[0]), np.nanmean(np.array(total_alpha_ps), axis = 0), alpha=.9, ls='-', lw=4, marker='d', markersize=10)
        plt.plot(np.arange(np.nanmean(np.array(total_alpha_qs), axis = 0).shape[0]), np.nanmean(np.array(total_alpha_qs), axis = 0), alpha=.9, ls='-', lw=4, marker='d', markersize=10)
        # for idx, alphas in enumerate(selected_alpha_qs):
        #     plt.plot(np.arange(alphas.shape[0]), alphas, alpha=.9, ls='-', lw=4, marker='d', markersize=10, c='g', label='Quality Sensitive')
        plt.legend()
        
        plt.xticks(np.arange(0, 16, 2), np.arange(1, 17, 2))
        plt.xlabel('Days', fontsize=18)
        plt.xlim([0, xlim])
        plt.ylim([0.10, 0.153])
        # plt.title('The %d dimension of Action'%(idy+1))
        if 'ext-v9' in env_id:
            plt.savefig('final_figures_mixed_prior_cost/{}_num_step={}_alpha_qs_ps_ps={}_{}_qs={}_{}_psc={}_{}_qsc={}_{}.pdf'.format(env_id, num_steps, PSpriceSensitivity, QSpriceSensitivity, PSqualitySensitivity, QSqualitySensitivity, PSpriceScale, QSpriceScale, PSqualityScale, QSqualityScale), bbox_inches='tight')
        else:
            plt.savefig('final_figures_mixed_prior_cost/%s_num_step=%d_alpha_qs_ps.pdf'%(env_id, num_steps), bbox_inches='tight')

        plt.clf()

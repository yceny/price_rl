from baselines.common.cmd_util import make_mujoco_env
from baselines.common import tf_util as U
import os.path as osp 
import tensorflow as tf 
import numpy as np 
import test_envs
import time
import glob
import gym
import pdb
import os
import gym_simple_pricing
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 


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
    import mlp_policy, pposgd_simpler, tree_policy, simple_tree_policy
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
    env_id = 'pricing-ext-v2'
    pol_type = 'mlp'
    # cost = 1000
    cost=500

    if 'ext-v2' in env_id:
        pdir = osp.join('TRY_logs', 'env=%s-c-%d'%(env_id, cost), 'seed=%d_*'%(seed))
    else:
        pdir = osp.join('TRY_logs', 'env=%s'%(env_id), 'seed=%d_*'%(seed))

    print(pdir)
    logdir = glob.glob(pdir)[0]
    
    num_steps = int(3000)
    # num_steps = int(150)
    env, pi = construct(env_id,  1, logdir)
    # env.messageCost = cost
    obs_dim = env.observation_space.shape[0]
    acts_dim = env.action_space.shape[0]

    ob = env.reset()
    traj_rewards = []
    traj_acts = []
    total_rewards = []
    total_trajs = []
    traj_obs = [ob]
    total_obs = []
    total_alpha = []
    total_alpha_ps = []
    total_alpha_qs = []
    episode_rews = 0
    # env_id += '_cost='+str(cost)
    total_all_rewards = []
    total_final_obs = []
    total_final_days = []
    print(logdir)
    for tt in range(1, num_steps):
        ac, _ = pi.act(False, ob)
        
        # ac = 30
        ob, r, done, info = env.step(ac)
        episode_rews += r * 100
        traj_rewards.append(episode_rews)
        traj_acts.append(ac)
        traj_obs.append(ob)
        if done :#
            print("Finished one trajectory with reward=%f"%(traj_rewards[-1]))
            print("length=%d"%len(traj_rewards))
            total_all_rewards.append(traj_rewards[-1])
            
            arr_traj = np.array(traj_acts).reshape(-1, acts_dim)
            # arr_traj[:, 0] = np.clip((arr_traj[:, 0] + 3.) * 75, 150, 300)
            arr_traj[:, 0] = np.clip((arr_traj[:, 0] + 1.) * 150, 0, 300)
            if arr_traj.shape[1] > 1:
                arr_traj[:, 1] = (arr_traj[:, 1] + 1.)  / 2.
                
            if arr_traj.shape[1] > 2:
                arr_traj[:, 2] = (arr_traj[:, 2] + 1.)  / 2.
                
            
            total_trajs.append(arr_traj)
            
            traj_obs = np.array(traj_obs).reshape(-1, obs_dim)
            traj_obs[:, 0] = traj_obs[:, 0] * 500

            total_obs.append(traj_obs)
            total_rewards.append(np.array(traj_rewards))
            total_final_obs.append(np.array(traj_obs[-1]))
            total_final_days.append(len(traj_rewards))
            if hasattr(env.env, 'hist_alpha'):
                total_alpha.append(np.array(env.env.hist_alpha))

            if hasattr(env.env, 'hist_alpha_qs'):
                total_alpha_qs.append(np.array(env.env.hist_alpha_qs))
            
            if hasattr(env.env, 'hist_alpha_ps'):
                total_alpha_ps.append(np.array(env.env.hist_alpha_ps))
            

            traj_rewards = []
            traj_acts = []
            # pdb.set_trace()
            episode_rews = 0
            ob = env.reset()
            traj_obs = [ob]
            # import pdb; pdb.set_trace()
    
    total_all_rewards = np.array(total_all_rewards)
    total_final_days = np.array(total_final_days)
    total_final_obs = np.array(total_final_obs)
    acts_list = total_trajs

    print("All Rewards Mean = %.3f, std = %.3f"%(total_all_rewards.mean(), total_all_rewards.std()))
    print("All Inventory Mean = %.3f, std = %.3f"%(total_final_obs.mean(axis=0)[0], total_final_obs.std(axis=0)[0]))
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
    selected_list = [acts_list[idx] for idx in all_idxs]
    selected_rewards = [total_rewards[idx] for idx in all_idxs]
    selected_obs = [total_obs[idx] for idx in all_idxs]
    xlim = 15.5
    # plots things 
    plt.clf()
    # plot the first dimension
    for idy in range(acts_list[0].shape[1]):
        plt.clf()
        plt.rc('xtick',labelsize=20)
        plt.rc('ytick',labelsize=20)

        if idy >= 2:
            for idx, act in enumerate(selected_list):
                plt.plot(np.arange(act.shape[0]), act[:, 1], alpha=.9, ls='-', lw=4, marker='d',  markersize=10, c='r', label='Price Sensitive')
            
            if idy == 2:
                for idx, act in enumerate(selected_list):
                    plt.plot(np.arange(act.shape[0]), act[:, 2], alpha=.9, ls='-', lw=4, marker='d',  markersize=10,  c='g', label='Quality Sensitive')

                plt.legend()
        else:
            for idx, act in enumerate(selected_list):
                plt.plot(np.arange(act.shape[0]), act[:, idy], alpha=.9, ls='-', lw=4, marker='d',  markersize=10)
        
        if idy == 0:
            pass
            # plt.ylim([128, 210])
        else:
            pass
            # plt.ylim([0, 1])
            
        plt.xticks(np.arange(0, 16, 2), np.arange(1, 17, 2))
        plt.xlabel('Days', fontsize=18)
        plt.xlim([0, xlim])
        # plt.title('The %d dimension of Action'%(idy+1))
        plt.savefig('final_figures/%s_act=%d.pdf'%(env_id, idy), bbox_inches='tight')
        plt.clf()
    
    
    for idy in range(total_obs[0].shape[1]):
        plt.clf()
        plt.rc('xtick',labelsize=20)
        plt.rc('ytick',labelsize=20)
        for idx, obs in enumerate(selected_obs):
            plt.plot(np.arange(obs.shape[0]), obs[:, idy], alpha=.9, ls='-', lw=4, marker='d', markersize=10)
        plt.xticks(np.arange(0, 16, 2), np.arange(1, 17, 2))
        plt.xlabel('Days', fontsize=18)
        plt.xlim([0, xlim])
        if idy == 0:
            plt.ylim([-10, 500])
        else:
            plt.ylim([0, 1])
        # plt.title('The %d dimension of Action'%(idy+1))
        plt.savefig('final_figures/%s_obs=%d.pdf'%(env_id, idy), bbox_inches='tight')
        plt.clf()
    
    if hasattr(env.env, 'hist_alpha'): 
        selected_alpha = [total_alpha[idx] for idx in all_idxs]
        plt.clf()
        plt.rc('xtick',labelsize=20)
        plt.rc('ytick',labelsize=20)
        for idx, alphas in enumerate(selected_alpha):
            plt.plot(np.arange(alphas.shape[0]), alphas, alpha=.9, ls='-', lw=4, marker='d', markersize=10)
        plt.xticks(np.arange(0, 16, 2), np.arange(1, 17, 2))
        plt.xlabel('Days', fontsize=18)
        plt.xlim([0, xlim])
        plt.ylim([0.10, 0.153])
        # plt.title('The %d dimension of Action'%(idy+1))
        plt.savefig('final_figures/%s_alpha.pdf'%(env_id), bbox_inches='tight')
        plt.clf()

    
    if hasattr(env.env, 'hist_alpha_qs'): 
        selected_alpha_qs = [total_alpha_qs[idx] for idx in all_idxs]
        plt.clf()
        plt.rc('xtick',labelsize=20)
        plt.rc('ytick',labelsize=20)
        for idx, alphas in enumerate(selected_alpha_qs):
            plt.plot(np.arange(alphas.shape[0]), alphas, alpha=.9, ls='-', lw=4, marker='d', markersize=10)
        plt.xticks(np.arange(0, 16, 2), np.arange(1, 17, 2))
        plt.xlabel('Days', fontsize=18)
        plt.xlim([0, xlim])
        plt.ylim([0.10, 0.153])
        # plt.title('The %d dimension of Action'%(idy+1))
        plt.savefig('final_figures/%s_alpha_qs.pdf'%(env_id), bbox_inches='tight')
        plt.clf()
    
    if hasattr(env.env, 'hist_alpha_ps'): 
        selected_alpha_ps = [total_alpha_ps[idx] for idx in all_idxs]
        plt.clf()
        plt.rc('xtick',labelsize=20)
        plt.rc('ytick',labelsize=20)
        for idx, alphas in enumerate(selected_alpha_ps):
            plt.plot(np.arange(alphas.shape[0]), alphas, alpha=.9, ls='-', lw=4, marker='d', markersize=10, c='r', label='Price Sensitive')
        
        for idx, alphas in enumerate(selected_alpha_qs):
            plt.plot(np.arange(alphas.shape[0]), alphas, alpha=.9, ls='-', lw=4, marker='d', markersize=10, c='g', label='Quality Sensitive')
        plt.legend()
        
        plt.xticks(np.arange(0, 16, 2), np.arange(1, 17, 2))
        plt.xlabel('Days', fontsize=18)
        plt.xlim([0, xlim])
        plt.ylim([0.10, 0.153])
        # plt.title('The %d dimension of Action'%(idy+1))
        plt.savefig('final_figures/%s_alpha_ps.pdf'%(env_id), bbox_inches='tight')
        plt.clf()



    # for idx, rew in enumerate(selected_rewards):
    #     plt.plot(np.arange(rew.shape[0]), rew, alpha=0.5, ls='--', lw=2, marker='x')
    # plt.xticks(np.arange(0, 22, 3), np.arange(1, 23, 3))
    # plt.xlabel('Days')
    # # plt.title('Accumlated Rewards')
    # plt.savefig('final_figures/%s_rew.pdf'%(env_id))
    # plt.clf()

#!/usr/bin/env python3

from baselines.common.cmd_util import make_mujoco_env, mujoco_arg_parser
from baselines.common import tf_util as U
from baselines import logger
from datetime import datetime

import gym_simple_pricing
import pdb 

def train(env_id, num_timesteps, seed):
    import mlp_policy, pposgd_simple
    U.make_session(num_cpu=1).__enter__()
    def policy_fn(name, ob_space, ac_space):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=32, num_hid_layers=2)

    env = make_mujoco_env(env_id, seed)
    logger.log("========observation_space %s action_space %s"%(str(env.observation_space), str(env.action_space)))
    pposgd_simple.learn(env, policy_fn,
            max_timesteps=num_timesteps,
            timesteps_per_actorbatch=1024,
            clip_param=0.2, entcoeff=0.0,
            optim_epochs=10, optim_stepsize=3e-4, optim_batchsize=64,
            gamma=0.99, lam=0.95, schedule='linear')
    env.close()

def main():
    parser = mujoco_arg_parser()
    args = parser.parse_args()
    if 'ext-v2' in args.env:
        import gym
        cost = gym.make(args.env).messageCost
        logdir = 'TRY_logs/env=%s-c-%d/seed=%d_%s'%(args.env, 
            cost,
            args.seed, 
            datetime.now().strftime('%d_%H:%M:%S'))
    else:

        logdir = 'TRY_logs/env=%s/seed=%d_%s'%(args.env, 
            args.seed, 
            datetime.now().strftime('%d_%H:%M:%S'))
    logger.configure(logdir)

    train(args.env,
        num_timesteps=args.num_timesteps, 
        seed=args.seed)


if __name__ == '__main__':
    main()

import os
import argparse
import subprocess
from collections import defaultdict

import numpy as np
from stable_baselines.results_plotter import load_results, ts2xy

from utils import get_trained_models

parser = argparse.ArgumentParser()
parser.add_argument('--log-dir', help='Root log folder', default='trained_agents/', type=str)
parser.add_argument('--benchmark-dir', help='Benchmark log folder', default='logs/benchmark/', type=str)
parser.add_argument('-n', '--n-timesteps', help='number of timesteps', default=10000,
                    type=int)
parser.add_argument('--n-envs', help='number of environments', default=1,
                    type=int)
parser.add_argument('--verbose', help='Verbose mode (0: no output, 1: INFO)', default=1,
                    type=int)
parser.add_argument('--seed', help='Random generator seed', type=int, default=0)
parser.add_argument('--test-mode', action='store_true', default=False,
                    help='Do only one experiments (useful for testing)')
args = parser.parse_args()

trained_models = get_trained_models(args.log_dir)
models_results = {}
env_results = defaultdict(dict)
n_experiments = len(trained_models)

for idx, trained_model in enumerate(trained_models.keys()):
    algo, env_id = trained_models[trained_model]
    n_envs = args.n_envs
    n_timesteps = args.n_timesteps
    if algo in ['dqn', 'ddpg']:
        n_envs = 1
        n_timesteps *= args.n_envs
    reward_log = '{}/{}/'.format(args.benchmark_dir, trained_model)
    arguments = [
        '-n', str(n_timesteps),
        '--n-envs', str(n_envs),
        '-f', args.log_dir,
        '--algo', algo,
        '--env', env_id,
        '--no-render',
        '--seed', str(args.seed),
        '--reward-log', reward_log
    ]
    if args.verbose >= 1:
        print('{}/{}'.format(idx + 1, n_experiments))
        print("Evaluating {} on {}...".format(algo, env_id))


    skip_eval = False
    if os.path.isdir(reward_log):
        try:
            x, y = ts2xy(load_results(reward_log), 'timesteps')
            skip_eval = len(x) > 0
        except:
            pass

    if skip_eval:
        print("Skipping eval...")
    else:
        return_code = subprocess.call(['python', 'enjoy.py'] + arguments)
        x, y = ts2xy(load_results(reward_log), 'timesteps')

    if len(x) > 0:
        mean_reward = np.mean(y[-100:])
        std_reward = np.std(y[-100:])
        models_results[trained_model] = (mean_reward, std_reward)
        env_results[env_id][algo] = (mean_reward, std_reward)
        if args.verbose >= 1:
            print(x[-1], 'timesteps')
            print(len(y), "Episodes")
            print("Mean reward: {:.2f} +- {:.2f}".format(mean_reward, std_reward))
            print()
    else:
        print("Not enough timesteps")
        
    if args.test_mode:
        break

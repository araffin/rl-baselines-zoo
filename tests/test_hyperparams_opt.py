import os
import shutil
import subprocess

import pytest


def _assert_eq(left, right):
    assert left == right, '{} != {}'.format(left, right)


N_STEPS = 100
N_TRIALS = 2
N_JOBS = 1

ALGOS = ('ppo2', 'a2c', 'trpo', 'acktr')
# Not yet supported:
# ALGOS = ('acer', 'dqn')
ENV_IDS = ('CartPole-v1',)
LOG_FOLDER = 'logs/tests_optimize/'

experiments = {}

for algo in ALGOS:
    for env_id in ENV_IDS:
        experiments['{}-{}'.format(algo, env_id)] = (algo, env_id)

# Test for DDPG
experiments['ddpg-MountainCarContinuous-v0'] = ('ddpg', 'MountainCarContinuous-v0')
# Test for SAC
experiments['sac-Pendulum-v0'] = ('sac', 'Pendulum-v0')
# Test for TD3
experiments['td3-Pendulum-v0'] = ('td3', 'Pendulum-v0')

# Clean up
if os.path.isdir(LOG_FOLDER):
    shutil.rmtree(LOG_FOLDER)


@pytest.mark.parametrize("sampler", ['random', 'tpe'])
@pytest.mark.parametrize("pruner", ['none', 'halving', 'median'])
@pytest.mark.parametrize("experiment", experiments.keys())
def test_optimize(sampler, pruner, experiment):
    algo, env_id = experiments[experiment]
    args = [
        '-n', str(N_STEPS),
        '--algo', algo,
        '--env', env_id,
        '--log-folder', LOG_FOLDER,
        '--n-trials', str(N_TRIALS),
        '--n-jobs', str(N_JOBS),
        '--sampler', sampler,
        '--pruner', pruner,
        '-optimize'
    ]

    return_code = subprocess.call(['python', 'train.py'] + args)
    _assert_eq(return_code, 0)

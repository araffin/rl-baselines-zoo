import os
import glob
import subprocess

import pytest


def _assert_eq(left, right):
    assert left == right, '{} != {}'.format(left, right)


FOLDER = 'trained_agents/'
N_STEPS = 100

algos = os.listdir(FOLDER)
trained_models = {}
for algo in algos:
    for env_id in glob.glob('{}/{}/*.pkl'.format(FOLDER, algo)):
        # Retrieve env name
        env_id = env_id.split('/')[-1].split('.pkl')[0]
        trained_models['{}-{}'.format(algo, env_id)] = (algo, env_id)


@pytest.mark.parametrize("trained_model", trained_models.keys())
def test_enjoy(trained_model):
    algo, env_id = trained_models[trained_model]
    args = [
        '-n', str(N_STEPS),
        '-f', FOLDER,
        '--algo', algo,
        '--env', env_id,
        '--no-render'
    ]

    return_code = subprocess.call(['python', 'enjoy.py'] + args)
    _assert_eq(return_code, 0)

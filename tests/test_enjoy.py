import subprocess

import pytest

from utils import get_trained_models

def _assert_eq(left, right):
    assert left == right, '{} != {}'.format(left, right)


FOLDER = 'trained_agents/'
N_STEPS = 100

trained_models = get_trained_models(FOLDER)


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


def test_benchmark():
    args = [
        '-n', str(N_STEPS),
        '--benchmark-dir', 'logs/tests/benchmark/',
        '--test-mode'
    ]

    return_code = subprocess.call(['python', '-m', 'utils.benchmark'] + args)
    _assert_eq(return_code, 0)

import gym
import time
import os

from stable_baselines.deepq.policies import FeedForwardPolicy
from stable_baselines.common.policies import register_policy
from stable_baselines.bench import Monitor
from stable_baselines import PPO2, A2C, ACER, ACKTR, DQN
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize,\
    VecFrameStack, SubprocVecEnv
from stable_baselines.common.cmd_util import make_atari_env

ALGOS = {
    'a2c': A2C,
    'acer': ACER,
    'acktr': ACKTR,
    'dqn': DQN,
    'ppo2': PPO2
}


# ================== Custom Policies =================

class CustomDQNPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomDQNPolicy, self).__init__(*args, **kwargs,
                                              layers=[64],
                                              layer_norm=True,
                                              feature_extraction="mlp")


register_policy('CustomDQNPolicy', CustomDQNPolicy)


def make_env(env_id, rank=0, seed=0, log_dir=None):
    """
    Helper function to multiprocess training
    and log the progress.

    :param env_id: (str)
    :param rank: (int)
    :param seed: (int)
    :param log_dir: (str)
    """
    if log_dir is None:
        log_dir = "/tmp/gym/{}/".format(int(time.time()))
    os.makedirs(log_dir, exist_ok=True)

    def _init():
        env = gym.make(env_id)
        env.seed(seed + rank)
        env = Monitor(env, os.path.join(log_dir, str(rank)), allow_early_resets=True)
        return env

    return _init


def create_test_env(env_id, n_envs=1, is_atari=False,
                    stats_path=None, norm_reward=False, seed=0):
    """
    Create environment for testing a trained agent

    :param env_id: (str)
    :param n_envs: (int) number of processes
    :param is_atari: (bool)
    :param stats_path: (str) path to folder containing saved running averaged
    :param norm_reward: (bool) Whether to normalize rewards or not when using Vecnormalize
    :param seed: (int) Seed for random number generator
    """
    # Create the environment and wrap it if necessary
    if is_atari:
        print("Using Atari wrapper")
        env = make_atari_env(env_id, num_env=n_envs, seed=seed)
        # Frame-stacking with 4 frames
        env = VecFrameStack(env, n_stack=4)
    elif n_envs > 1:
        env = SubprocVecEnv([make_env(env_id, i, seed) for i in range(n_envs)])
    else:
        env = DummyVecEnv([lambda: gym.make(env_id)])

    # Load saved stats for normalizing input and rewards
    if stats_path is not None:
        print("Loading running average")
        env = VecNormalize(env, training=False, norm_reward=norm_reward)
        env.load_running_average(stats_path)
    return env

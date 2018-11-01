import gym
import time
import os

from stable_baselines.deepq.policies import FeedForwardPolicy
from stable_baselines.common.policies import register_policy
from stable_baselines.bench import Monitor
from stable_baselines import PPO2, A2C, ACER, ACKTR, DQN


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


def make_env(env_id, rank=0, seed=0):
    """
    Helper function to multiprocess training
    and log the progress.

    :param env_id: (str)
    :param rank: (int)
    :param seed: (int)
    """
    log_dir = "/tmp/gym/{}/".format(int(time.time()))
    os.makedirs(log_dir, exist_ok=True)

    def _init():
        env = gym.make(env_id)
        env.seed(seed + rank)
        env = Monitor(env, os.path.join(log_dir, str(rank)), allow_early_resets=True)
        return env

    return _init

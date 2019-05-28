import argparse
import difflib
import os
from collections import OrderedDict
from pprint import pprint
import warnings

# For pybullet envs
warnings.filterwarnings("ignore")
import gym
import pybullet_envs
import numpy as np
import yaml
try:
    import highway_env
except ImportError:
    highway_env = None
from mpi4py import MPI

from stable_baselines import HER, SAC
from stable_baselines.her import HERGoalEnvWrapper

from stable_baselines.common import set_global_seeds
from stable_baselines.common.base_class import _UnvecWrapper
from stable_baselines.common.vec_env import VecFrameStack, SubprocVecEnv, VecNormalize, DummyVecEnv
from stable_baselines.ddpg import AdaptiveParamNoiseSpec, NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines.ppo2.ppo2 import constfn

from utils import make_env, ALGOS, linear_schedule, get_latest_run_id

env = gym.make("FetchReach-v1")
set_global_seeds(1)
env.seed(1)
model = HER('MlpPolicy', env, SAC)

# env2 = gym.make("FetchReach-v1")
# env2.seed(1)
env2 = DummyVecEnv([lambda: env])
env2.envs[0].seed(1)
env2_wrapped = HERGoalEnvWrapper(_UnvecWrapper(env2))

env_wrapped = HERGoalEnvWrapper(env)
obs = env_wrapped.reset()
obs2 = env2_wrapped.reset()
print(obs)
print(obs2)

# action = env.action_space.sample()
print(model.predict(obs))
print(model.predict(obs2))

# env_ = model.get_env()
# print(env_.reset())
# print(env_.step(action))
# print(env_.step(action))

# env = gym.make("FetchReach-v1")
# set_global_seeds(1)
# env.seed(1)
# model = HER('MlpPolicy', env, SAC)
#
# env_ = model.get_env()
# print(env_.reset())
# print(env_.step(action)[0])

# env2 = gym.make("FetchReach-v1")
# env2.seed(1)
# env2 = DummyVecEnv([lambda: env2])
# set_global_seeds(1)
# model2 = HER('MlpPolicy', env2, SAC)
#
# env2_ = model.get_env()
# env2_.seed(1)
# print(env2_.reset())
# print(env2_.step(action))
# print(env2_.step(action))

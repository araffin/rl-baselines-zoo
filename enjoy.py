import argparse
import os

import gym
import numpy as np

from stable_baselines import PPO2, A2C, ACER, ACKTR
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize, VecFrameStack
from stable_baselines.common.cmd_util import make_atari_env

ALGOS = {
	'a2c': A2C,
	'acer': ACER,
	'acktr': ACKTR,
	'ppo2': PPO2
}

parser = argparse.ArgumentParser()
parser.add_argument('--env', help='environment ID', type=str, default='CartPole-v1')
parser.add_argument('-f', '--folder', help='Log folder', type=str, default='trained_agents')
parser.add_argument('--algo', help='RL Algorithm', default='ppo2',
					type=str, required=False, choices=list(ALGOS.keys()))
parser.add_argument('-n', '--n-timesteps', help='number of timesteps', default=1000,
					type=int)
parser.add_argument('--no-render', action='store_true', default=False,
                    help='Do not render the environment (useful for tests)')
args = parser.parse_args()


env_id = args.env
algo = args.algo
folder = args.folder
model_path = "{}/{}/{}.pkl".format(folder, algo, env_id)

assert os.path.isdir(folder + '/' + algo), "The {}/{}/ folder was not found".format(folder, algo)
assert os.path.isfile(model_path), "No model found for {} on {}, path: {}".format(algo, env_id, model_path)

is_atari = False
if 'NoFrameskip' in env_id:
	is_atari = True
	print("Using Atari wrapper")
	env = make_atari_env(env_id, num_env=1, seed=0)
	# Frame-stacking with 4 frames
	env = VecFrameStack(env, n_stack=4)
else:
	env = DummyVecEnv([lambda: gym.make(env_id)])

stats_path = "{}/{}/{}/".format(folder, algo, env_id)

if os.path.isdir(stats_path):
	print("Loading running average")
	env = VecNormalize(env, training=False)
	env.load_running_average(stats_path)

model = ALGOS[algo].load(model_path)

obs = env.reset()

running_reward = 0.0
ep_len = 0
for _ in range(args.n_timesteps):
	action, _ = model.predict(obs, deterministic=False)
	# Random Agent
	# action = [env.action_space.sample()]
	# Clip Action
	if isinstance(env.action_space, gym.spaces.Box):
		action = np.clip(action, env.action_space.low, env.action_space.high)
	obs, reward, done, infos = env.step(action)
	if not args.no_render:
		env.render()
	running_reward += reward[0]
	ep_len += 1

	# For atari the return reward is not the atari score
	# so we have to get it from the infos dict
	if infos is not None:
		episode_infos = infos[0].get('episode')
		if episode_infos is not None:
			print("Atari Episode Score: {:.2f}".format(episode_infos['r']))
			print("Atari Episode Length", episode_infos['l'])

	if done and not is_atari:
		# NOTE: for env using VecNormalize, the mean reward
		# is a normalized reward
		print("Episode Reward: {:.2f}".format(running_reward))
		print("Episode Length", ep_len)
		running_reward = 0.0
		ep_len = 0

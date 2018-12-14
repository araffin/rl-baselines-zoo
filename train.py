import argparse
import difflib
import os
from collections import OrderedDict
from pprint import pprint

import gym
import pybullet_envs
import numpy as np
import yaml
from stable_baselines.common import set_global_seeds
from stable_baselines.common.cmd_util import make_atari_env
from stable_baselines.common.vec_env import VecFrameStack, SubprocVecEnv, VecNormalize, DummyVecEnv
from stable_baselines.ddpg import AdaptiveParamNoiseSpec, NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines.ppo2.ppo2 import constfn

from utils import make_env, ALGOS, linear_schedule, get_latest_run_id

parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, nargs='+', default=["CartPole-v1"], help='environment ID(s)')
parser.add_argument('-tb', '--tensorboard-log', help='Tensorboard log dir', default='', type=str)
parser.add_argument('-i', '--trained-agent', help='Path to a pretrained agent to continue training',
                    default='', type=str)
parser.add_argument('--algo', help='RL Algorithm', default='ppo2',
                    type=str, required=False, choices=list(ALGOS.keys()))
parser.add_argument('-n', '--n-timesteps', help='Overwrite the number of timesteps', default=-1,
                    type=int)
parser.add_argument('--log-interval', help='Override log interval (default: -1, no change)', default=-1,
                    type=int)
parser.add_argument('-f', '--log-folder', help='Log folder', type=str, default='logs')
parser.add_argument('--seed', help='Random generator seed', type=int, default=0)
args = parser.parse_args()

env_ids = args.env

registered_envs = set(gym.envs.registry.env_specs.keys())

for env_id in env_ids:
    # If the environment is not found, suggest the closest match
    if env_id not in registered_envs:
        closest_match = difflib.get_close_matches(env_id, registered_envs, n=1)[0]
        raise ValueError('{} not found in gym registry, you maybe meant {}?'.format(env_id, closest_match))

set_global_seeds(args.seed)

if args.trained_agent != "":
    assert args.trained_agent.endswith('.pkl') and os.path.isfile(args.trained_agent), \
        "The trained_agent must be a valid path to a .pkl file"

for env_id in env_ids:
    tensorboard_log = None if args.tensorboard_log == '' else args.tensorboard_log + '/' + env_id

    is_atari = False
    if 'NoFrameskip' in env_id:
        is_atari = True

    print("=" * 10, env_id, "=" * 10)

    # Load hyperparameters from yaml file
    with open('hyperparams/{}.yml'.format(args.algo), 'r') as f:
        if is_atari:
            hyperparams = yaml.load(f)['atari']
        else:
            hyperparams = yaml.load(f)[env_id]

    # Sort hyperparams that will be saved
    saved_hyperparams = OrderedDict([(key, hyperparams[key]) for key in sorted(hyperparams.keys())])
    pprint(saved_hyperparams)

    n_envs = hyperparams.get('n_envs', 1)

    print("Using {} environments".format(n_envs))

    # Create learning rate schedules for ppo2 and sac
    if args.algo in ["ppo2", "sac"]:
        for key in ['learning_rate', 'cliprange']:
            if key not in hyperparams:
                continue
            if isinstance(hyperparams[key], str):
                schedule, initial_value = hyperparams[key].split('_')
                initial_value = float(initial_value)
                hyperparams[key] = linear_schedule(initial_value)
            elif isinstance(hyperparams[key], float):
                hyperparams[key] = constfn(hyperparams[key])
            else:
                raise ValueError('Invalid valid for {}: {}'.format(key, hyperparams[key]))

    # Should we overwrite the number of timesteps?
    if args.n_timesteps > 0:
        n_timesteps = args.n_timesteps
    else:
        n_timesteps = int(hyperparams['n_timesteps'])

    normalize = False
    normalize_kwargs = {}
    if 'normalize' in hyperparams.keys():
        normalize = hyperparams['normalize']
        if isinstance(normalize, str):
            normalize_kwargs = eval(normalize)
            normalize = True
        del hyperparams['normalize']

    # Delete keys so the dict can be pass to the model constructor
    if 'n_envs' in hyperparams.keys():
        del hyperparams['n_envs']
    del hyperparams['n_timesteps']

    # Create the environment and wrap it if necessary
    if is_atari:
        print("Using Atari wrapper")
        env = make_atari_env(env_id, num_env=n_envs, seed=args.seed)
        # Frame-stacking with 4 frames
        env = VecFrameStack(env, n_stack=4)
    elif args.algo in ['dqn', 'ddpg']:
        if hyperparams.get('normalize', False):
            print("WARNING: normalization not supported yet for DDPG/DQN")
        env = gym.make(env_id)
        env.seed(args.seed)
    else:
        if n_envs == 1:
            env = DummyVecEnv([make_env(env_id, 0, args.seed)])
        else:
            env = SubprocVecEnv([make_env(env_id, i, args.seed) for i in range(n_envs)])
        if normalize:
            print("Normalizing input and return")
            env = VecNormalize(env, **normalize_kwargs)

    # Optional Frame-stacking
    n_stack = 1
    if hyperparams.get('frame_stack', False):
        n_stack = hyperparams['frame_stack']
        env = VecFrameStack(env, n_stack)
        print("Stacking {} frames".format(n_stack))
        del hyperparams['frame_stack']

    # Parse noise string for DDPG
    if args.algo == 'ddpg' and hyperparams.get('noise_type') is not None:
        noise_type = hyperparams['noise_type'].strip()
        noise_std = hyperparams['noise_std']
        n_actions = env.action_space.shape[0]
        if 'adaptive-param' in noise_type:
            hyperparams['param_noise'] = AdaptiveParamNoiseSpec(initial_stddev=noise_std,
                                                                desired_action_stddev=noise_std)
        elif 'normal' in noise_type:
            hyperparams['action_noise'] = NormalActionNoise(mean=np.zeros(n_actions),
                                                            sigma=noise_std * np.ones(n_actions))
        elif 'ornstein-uhlenbeck' in noise_type:
            hyperparams['action_noise'] = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions),
                                                                       sigma=noise_std * np.ones(n_actions))
        else:
            raise RuntimeError('Unknown noise type "{}"'.format(noise_type))
        print("Applying {} noise with std {}".format(noise_type, noise_std))
        del hyperparams['noise_type']
        del hyperparams['noise_std']

    if args.trained_agent.endswith('.pkl') and os.path.isfile(args.trained_agent):
        # Continue training
        print("Loading pretrained agent")
        # Policy should not be changed
        del hyperparams['policy']

        model = ALGOS[args.algo].load(args.trained_agent, env=env,
                                      tensorboard_log=tensorboard_log, verbose=1, **hyperparams)

        exp_folder = args.trained_agent.split('.pkl')[0]
        if os.path.isdir(exp_folder):
            print("Loading saved running average")
            env.load_running_average(exp_folder)
    else:
        # Train an agent from scratch
        model = ALGOS[args.algo](env=env, tensorboard_log=tensorboard_log, verbose=1, **hyperparams)

    kwargs = {}
    if args.log_interval > -1:
        kwargs = {'log_interval': args.log_interval}

    model.learn(n_timesteps, **kwargs)

    # Save trained model
    log_path = "{}/{}/".format(args.log_folder, args.algo)
    save_path = os.path.join(log_path, "{}_{}".format(env_id, get_latest_run_id(log_path, env_id) + 1))
    params_path = "{}/{}".format(save_path, env_id)
    os.makedirs(params_path, exist_ok=True)
    print("Saving to {}".format(save_path))

    model.save("{}/{}".format(save_path, env_id))
    # Save hyperparams
    with open(os.path.join(params_path, 'config.yml'), 'w') as f:
        yaml.dump(saved_hyperparams, f)

    if normalize:
        # Unwrap
        if isinstance(env, VecFrameStack):
            env = env.venv
        # Important: save the running average, for testing the agent we need that normalization
        env.save_running_average(params_path)

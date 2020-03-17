import os
import time
import uuid
import difflib
import argparse
import importlib
import warnings
from pprint import pprint
from collections import OrderedDict

# numpy warnings because of tensorflow
warnings.filterwarnings("ignore", category=FutureWarning, module='tensorflow')
warnings.filterwarnings("ignore", category=UserWarning, module='gym')

import gym
import numpy as np
import yaml
# Optional dependencies
import utils.import_envs  # pytype: disable=import-error
try:
    import mpi4py
    from mpi4py import MPI
except ImportError:
    mpi4py = None

from stable_baselines.common import set_global_seeds
from stable_baselines.common.cmd_util import make_atari_env
from stable_baselines.common.vec_env import VecFrameStack, SubprocVecEnv, VecNormalize, DummyVecEnv
from stable_baselines.common.noise import AdaptiveParamNoiseSpec, NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines.common.schedules import constfn
from stable_baselines.common.callbacks import CheckpointCallback, EvalCallback

from utils import make_env, ALGOS, linear_schedule, get_latest_run_id, get_wrapper_class, find_saved_model
from utils.hyperparams_opt import hyperparam_optimization
from utils.callbacks import SaveVecNormalizeCallback
from utils.noise import LinearNormalActionNoise
from utils.utils import StoreDict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default="CartPole-v1", help='environment ID')
    parser.add_argument('-tb', '--tensorboard-log', help='Tensorboard log dir', default='', type=str)
    parser.add_argument('-i', '--trained-agent', help='Path to a pretrained agent to continue training',
                        default='', type=str)
    parser.add_argument('--algo', help='RL Algorithm', default='ppo2',
                        type=str, required=False, choices=list(ALGOS.keys()))
    parser.add_argument('-n', '--n-timesteps', help='Overwrite the number of timesteps', default=-1,
                        type=int)
    parser.add_argument('--log-interval', help='Override log interval (default: -1, no change)', default=-1,
                        type=int)
    parser.add_argument('--eval-freq', help='Evaluate the agent every n steps (if negative, no evaluation)',
                        default=10000, type=int)
    parser.add_argument('--eval-episodes', help='Number of episodes to use for evaluation',
                        default=5, type=int)
    parser.add_argument('--save-freq', help='Save the model every n steps (if negative, no checkpoint)',
                        default=-1, type=int)
    parser.add_argument('-f', '--log-folder', help='Log folder', type=str, default='logs')
    parser.add_argument('--seed', help='Random generator seed', type=int, default=0)
    parser.add_argument('--n-trials', help='Number of trials for optimizing hyperparameters', type=int, default=10)
    parser.add_argument('-optimize', '--optimize-hyperparameters', action='store_true', default=False,
                        help='Run hyperparameters search')
    parser.add_argument('--n-jobs', help='Number of parallel jobs when optimizing hyperparameters', type=int, default=1)
    parser.add_argument('--sampler', help='Sampler to use when optimizing hyperparameters', type=str,
                        default='tpe', choices=['random', 'tpe', 'skopt'])
    parser.add_argument('--pruner', help='Pruner to use when optimizing hyperparameters', type=str,
                        default='median', choices=['halving', 'median', 'none'])
    parser.add_argument('--verbose', help='Verbose mode (0: no output, 1: INFO)', default=1,
                        type=int)
    parser.add_argument('--gym-packages', type=str, nargs='+', default=[],
                        help='Additional external Gym environemnt package modules to import (e.g. gym_minigrid)')
    parser.add_argument('-params', '--hyperparams', type=str, nargs='+', action=StoreDict,
                        help='Overwrite hyperparameter (e.g. learning_rate:0.01 train_freq:10)')
    parser.add_argument('-uuid', '--uuid', action='store_true', default=False,
                        help='Ensure that the run has a unique ID')
    args = parser.parse_args()

    # Going through custom gym packages to let them register in the global registory
    for env_module in args.gym_packages:
        importlib.import_module(env_module)

    env_id = args.env
    registered_envs = set(gym.envs.registry.env_specs.keys())

    # If the environment is not found, suggest the closest match
    if env_id not in registered_envs:
        try:
            closest_match = difflib.get_close_matches(env_id, registered_envs, n=1)[0]
        except IndexError:
            closest_match = "'no close match found...'"
        raise ValueError('{} not found in gym registry, you maybe meant {}?'.format(env_id, closest_match))

    # Unique id to ensure there is no race condition for the folder creation
    uuid_str = f'_{uuid.uuid4()}' if args.uuid else ''
    if args.seed < 0:
        # Seed but with a random one
        args.seed = np.random.randint(2**32 - 1)

    set_global_seeds(args.seed)

    if args.trained_agent != "":
        valid_extension = args.trained_agent.endswith('.pkl') or args.trained_agent.endswith('.zip')
        assert valid_extension and os.path.isfile(args.trained_agent), \
            "The trained_agent must be a valid path to a .zip/.pkl file"

    rank = 0
    if mpi4py is not None and MPI.COMM_WORLD.Get_size() > 1:
        print("Using MPI for multiprocessing with {} workers".format(MPI.COMM_WORLD.Get_size()))
        rank = MPI.COMM_WORLD.Get_rank()
        print("Worker rank: {}".format(rank))

        args.seed += rank
        if rank != 0:
            args.verbose = 0
            args.tensorboard_log = ''

    tensorboard_log = None if args.tensorboard_log == '' else os.path.join(args.tensorboard_log, env_id)

    is_atari = False
    if 'NoFrameskip' in env_id:
        is_atari = True

    print("=" * 10, env_id, "=" * 10)
    print("Seed: {}".format(args.seed))

    # Load hyperparameters from yaml file
    with open('hyperparams/{}.yml'.format(args.algo), 'r') as f:
        hyperparams_dict = yaml.safe_load(f)
        if env_id in list(hyperparams_dict.keys()):
            hyperparams = hyperparams_dict[env_id]
        elif is_atari:
            hyperparams = hyperparams_dict['atari']
        else:
            raise ValueError("Hyperparameters not found for {}-{}".format(args.algo, env_id))

    if args.hyperparams is not None:
        # Overwrite hyperparams if needed
        hyperparams.update(args.hyperparams)

    # Sort hyperparams that will be saved
    saved_hyperparams = OrderedDict([(key, hyperparams[key]) for key in sorted(hyperparams.keys())])

    algo_ = args.algo
    # HER is only a wrapper around an algo
    if args.algo == 'her':
        algo_ = saved_hyperparams['model_class']
        assert algo_ in {'sac', 'ddpg', 'dqn', 'td3'}, "{} is not compatible with HER".format(algo_)
        # Retrieve the model class
        hyperparams['model_class'] = ALGOS[saved_hyperparams['model_class']]
        if hyperparams['model_class'] is None:
            raise ValueError('{} requires MPI to be installed'.format(algo_))

    if args.verbose > 0:
        pprint(saved_hyperparams)

    n_envs = hyperparams.get('n_envs', 1)

    if args.verbose > 0:
        print("Using {} environments".format(n_envs))

    # Create learning rate schedules for ppo2 and sac
    if algo_ in ["ppo2", "sac", "td3"]:
        for key in ['learning_rate', 'cliprange', 'cliprange_vf']:
            if key not in hyperparams:
                continue
            if isinstance(hyperparams[key], str):
                schedule, initial_value = hyperparams[key].split('_')
                initial_value = float(initial_value)
                hyperparams[key] = linear_schedule(initial_value)
            elif isinstance(hyperparams[key], (float, int)):
                # Negative value: ignore (ex: for clipping)
                if hyperparams[key] < 0:
                    continue
                hyperparams[key] = constfn(float(hyperparams[key]))
            else:
                raise ValueError('Invalid value for {}: {}'.format(key, hyperparams[key]))

    # Should we overwrite the number of timesteps?
    if args.n_timesteps > 0:
        if args.verbose:
            print("Overwriting n_timesteps with n={}".format(args.n_timesteps))
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

    # Convert to python object if needed
    if 'policy_kwargs' in hyperparams.keys() and isinstance(hyperparams['policy_kwargs'], str):
        hyperparams['policy_kwargs'] = eval(hyperparams['policy_kwargs'])

    # Delete keys so the dict can be pass to the model constructor
    if 'n_envs' in hyperparams.keys():
        del hyperparams['n_envs']
    del hyperparams['n_timesteps']

    # obtain a class object from a wrapper name string in hyperparams
    # and delete the entry
    env_wrapper = get_wrapper_class(hyperparams)
    if 'env_wrapper' in hyperparams.keys():
        del hyperparams['env_wrapper']

    log_path = "{}/{}/".format(args.log_folder, args.algo)
    save_path = os.path.join(log_path, "{}_{}{}".format(env_id, get_latest_run_id(log_path, env_id) + 1, uuid_str))
    params_path = "{}/{}".format(save_path, env_id)
    os.makedirs(params_path, exist_ok=True)

    callbacks = []
    if args.save_freq > 0:
        # Account for the number of parallel environments
        args.save_freq = max(args.save_freq // n_envs, 1)
        callbacks.append(CheckpointCallback(save_freq=args.save_freq,
                                            save_path=save_path, name_prefix='rl_model', verbose=1))

    def create_env(n_envs, eval_env=False):
        """
        Create the environment and wrap it if necessary
        :param n_envs: (int)
        :param eval_env: (bool) Whether is it an environment used for evaluation or not
        :return: (Union[gym.Env, VecEnv])
        :return: (gym.Env)
        """
        global hyperparams

        # Do not log eval env (issue with writing the same file)
        log_dir = None if eval_env else save_path

        if is_atari:
            if args.verbose > 0:
                print("Using Atari wrapper")
            env = make_atari_env(env_id, num_env=n_envs, seed=args.seed)
            # Frame-stacking with 4 frames
            env = VecFrameStack(env, n_stack=4)
        elif algo_ in ['dqn', 'ddpg']:
            if hyperparams.get('normalize', False):
                print("WARNING: normalization not supported yet for DDPG/DQN")
            env = gym.make(env_id)
            env.seed(args.seed)
            if env_wrapper is not None:
                env = env_wrapper(env)
        else:
            if n_envs == 1:
                env = DummyVecEnv([make_env(env_id, 0, args.seed, wrapper_class=env_wrapper, log_dir=log_dir)])
            else:
                # env = SubprocVecEnv([make_env(env_id, i, args.seed) for i in range(n_envs)])
                # On most env, SubprocVecEnv does not help and is quite memory hungry
                env = DummyVecEnv([make_env(env_id, i, args.seed, log_dir=log_dir,
                                            wrapper_class=env_wrapper) for i in range(n_envs)])
            if normalize:
                if args.verbose > 0:
                    if len(normalize_kwargs) > 0:
                        print("Normalization activated: {}".format(normalize_kwargs))
                    else:
                        print("Normalizing input and reward")
                env = VecNormalize(env, **normalize_kwargs)
        # Optional Frame-stacking
        if hyperparams.get('frame_stack', False):
            n_stack = hyperparams['frame_stack']
            env = VecFrameStack(env, n_stack)
            print("Stacking {} frames".format(n_stack))
            del hyperparams['frame_stack']
        return env


    env = create_env(n_envs)
    # Create test env if needed, do not normalize reward
    eval_env = None
    if args.eval_freq > 0:
        # Account for the number of parallel environments
        args.eval_freq = max(args.eval_freq // n_envs, 1)

        # Do not normalize the rewards of the eval env
        old_kwargs = None
        if normalize:
            if len(normalize_kwargs) > 0:
                old_kwargs = normalize_kwargs.copy()
                normalize_kwargs['norm_reward'] = False
            else:
                normalize_kwargs = {'norm_reward': False}

        if args.verbose > 0:
            print("Creating test environment")

        save_vec_normalize = SaveVecNormalizeCallback(save_freq=1, save_path=params_path)
        eval_callback = EvalCallback(create_env(1, eval_env=True), callback_on_new_best=save_vec_normalize,
                                     best_model_save_path=save_path, n_eval_episodes=args.eval_episodes,
                                     log_path=save_path, eval_freq=args.eval_freq)
        callbacks.append(eval_callback)

        # Restore original kwargs
        if old_kwargs is not None:
            normalize_kwargs = old_kwargs.copy()


    # Stop env processes to free memory
    if args.optimize_hyperparameters and n_envs > 1:
        env.close()

    # Parse noise string for DDPG and SAC
    if algo_ in ['ddpg', 'sac', 'td3'] and hyperparams.get('noise_type') is not None:
        noise_type = hyperparams['noise_type'].strip()
        noise_std = hyperparams['noise_std']
        n_actions = env.action_space.shape[0]
        if 'adaptive-param' in noise_type:
            assert algo_ == 'ddpg', 'Parameter is not supported by SAC'
            hyperparams['param_noise'] = AdaptiveParamNoiseSpec(initial_stddev=noise_std,
                                                                desired_action_stddev=noise_std)
        elif 'normal' in noise_type:
            if 'lin' in noise_type:
                hyperparams['action_noise'] = LinearNormalActionNoise(mean=np.zeros(n_actions),
                                                                      sigma=noise_std * np.ones(n_actions),
                                                                      final_sigma=hyperparams.get('noise_std_final', 0.0) * np.ones(n_actions),
                                                                      max_steps=n_timesteps)
            else:
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
        if 'noise_std_final' in hyperparams:
            del hyperparams['noise_std_final']

    if ALGOS[args.algo] is None:
        raise ValueError('{} requires MPI to be installed'.format(args.algo))

    if os.path.isfile(args.trained_agent):
        # Continue training
        print("Loading pretrained agent")
        # Policy should not be changed
        del hyperparams['policy']

        model = ALGOS[args.algo].load(args.trained_agent, env=env,
                                      tensorboard_log=tensorboard_log, verbose=args.verbose, **hyperparams)

        exp_folder = args.trained_agent[:-4]
        if normalize:
            print("Loading saved running average")
            stats_path = os.path.join(exp_folder, env_id)
            if os.path.exists(os.path.join(stats_path, 'vecnormalize.pkl')):
                env = VecNormalize.load(os.path.join(stats_path, 'vecnormalize.pkl'), env)
            else:
                # Legacy:
                env.load_running_average(exp_folder)

    elif args.optimize_hyperparameters:

        if args.verbose > 0:
            print("Optimizing hyperparameters")


        def create_model(*_args, **kwargs):
            """
            Helper to create a model with different hyperparameters
            """
            return ALGOS[args.algo](env=create_env(n_envs), tensorboard_log=tensorboard_log,
                                    verbose=0, **kwargs)


        data_frame = hyperparam_optimization(args.algo, create_model, create_env, n_trials=args.n_trials,
                                             n_timesteps=n_timesteps, hyperparams=hyperparams,
                                             n_jobs=args.n_jobs, seed=args.seed,
                                             sampler_method=args.sampler, pruner_method=args.pruner,
                                             verbose=args.verbose)

        report_name = "report_{}_{}-trials-{}-{}-{}_{}.csv".format(env_id, args.n_trials, n_timesteps,
                                                                args.sampler, args.pruner, int(time.time()))

        log_path = os.path.join(args.log_folder, args.algo, report_name)

        if args.verbose:
            print("Writing report to {}".format(log_path))

        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        data_frame.to_csv(log_path)
        exit()
    else:
        # Train an agent from scratch
        model = ALGOS[args.algo](env=env, tensorboard_log=tensorboard_log, verbose=args.verbose, **hyperparams)

    kwargs = {}
    if args.log_interval > -1:
        kwargs = {'log_interval': args.log_interval}

    if len(callbacks) > 0:
        kwargs['callback'] = callbacks

    # Save hyperparams
    with open(os.path.join(params_path, 'config.yml'), 'w') as f:
        yaml.dump(saved_hyperparams, f)

    print("Log path: {}".format(save_path))

    try:
        model.learn(n_timesteps, **kwargs)
    except KeyboardInterrupt:
        pass


    # Only save worker of rank 0 when using mpi
    if rank == 0:
        print("Saving to {}".format(save_path))

        model.save("{}/{}".format(save_path, env_id))

    if normalize:
        # Important: save the running average, for testing the agent we need that normalization
        model.get_vec_normalize_env().save(os.path.join(params_path, 'vecnormalize.pkl'))
        # Deprecated saving:
        # env.save_running_average(params_path)

import numpy as np


def hyperparam_optimization(algo, model_fn, n_trials=10, n_timesteps=5000, hyperparams=None):
    """
    :param algo: (str)
    :param model_fn: (func)
    :param n_trials: (int)
    :param n_timesteps: (int)
    :param hyperparams: (dict)
    :return: (pd.Dataframe)
    """
    # Avoid showing tf logging
    import optuna

    if hyperparams is None:
        hyperparams = {}

    study = optuna.create_study()
    sampler = HYPERPARAMS_SAMPLER[algo]

    def objective(trial):

        kwargs = hyperparams.copy()
        kwargs.update(sampler(trial))

        def callback(_locals, _globals):
            """
            Callback for monitoring learning progress.

            :param _locals: (dict)
            :param _globals: (dict)
            :return: (bool) If False: stop training
            """
            trial = _locals['self'].trial
            _locals['self'].is_pruned = False

            if not hasattr(_locals['self'], 'best_mean_reward'):
                _locals['self'].best_mean_reward = -np.inf

            if len(_locals['ep_info_buf']) < 100:
                return True
            mean_reward = np.mean([ep_info['r'] for ep_info in _locals['ep_info_buf']])

            if mean_reward > _locals['self'].best_mean_reward:
                _locals['self'].best_mean_reward = mean_reward

            # Prune trial if need
            trial.report(-1 * _locals['self'].best_mean_reward, _locals['self'].num_timesteps)
            if trial.should_prune(_locals['self'].num_timesteps):
                _locals['self'].is_pruned = True
                return False

            return True

        model = model_fn(**kwargs)
        model.trial = trial
        model.learn(n_timesteps, callback=callback)
        is_pruned = False
        best_cost = np.inf
        if hasattr(model, 'is_pruned'):
            is_pruned = model.is_pruned
            best_cost = -1 * model.best_mean_reward
        # Free memory
        model.env.close()
        del model.env
        del model

        if is_pruned:
            raise optuna.structs.TrialPruned()

        return best_cost

    study.optimize(objective, n_trials=n_trials, n_jobs=1)

    print('Number of finished trials: ', len(study.trials))

    print('Best trial:')
    trial = study.best_trial

    print('  Value: ', trial.value)

    print('  Params: ')
    for key, value in trial.params.items():
        print('    {}: {}'.format(key, value))

    return study.trials_dataframe()


def sample_ppo2_params(trial):
    """
    Sampler for PPO2 hyperparams.

    :param trial: (optuna.trial)
    :return: (dict)
    """
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
    n_steps = trial.suggest_categorical('n_steps', [16, 32, 64, 128, 256, 512, 1024, 2048])
    lam = trial.suggest_categorical('lamdba', [0.8, 0.9, 0.92, 0.95, 0.98, 0.99, 1.0])
    gamma = trial.suggest_categorical('gamma', [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])
    cliprange = trial.suggest_categorical('cliprange', [0.1, 0.2, 0.3, 0.4])
    ent_coef = trial.suggest_loguniform('ent_coef', 0.00000001, 0.1)
    # learning_rate = trial.suggest_loguniform('lr', 1e-5, 1)

    if n_steps < batch_size:
        nminibatches = 1
    else:
        nminibatches = int(n_steps / batch_size)

    return {
        'n_steps': n_steps,
        'nminibatches': nminibatches,
        'gamma': gamma,
        # 'learning_rate': learning_rate,
        'ent_coef': ent_coef,
        'cliprange': cliprange,
        'lam': lam
    }


def sample_a2c_params(trial):
    """
    Sampler for A2C hyperparams.

    :param trial: (optuna.trial)
    :return: (dict)
    """
    gamma = trial.suggest_categorical('gamma', [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])
    n_steps = trial.suggest_categorical('n_steps', [5, 16, 32, 64])
    lr_schedule = trial.suggest_categorical('lr_schedule', ['linear', 'constant'])
    learning_rate = trial.suggest_loguniform('lr', 1e-5, 1)
    ent_coef = trial.suggest_loguniform('ent_coef', 0.00000001, 0.1)

    return {
        'n_steps': n_steps,
        'gamma': gamma,
        'learning_rate': learning_rate,
        'lr_schedule': lr_schedule,
        'ent_coef': ent_coef
    }


def sample_sac_params(trial):
    """
    Sampler for SAC hyperparams.

    :param trial: (optuna.trial)
    :return: (dict)
    """
    gamma = trial.suggest_categorical('gamma', [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])
    learning_rate = trial.suggest_loguniform('lr', 1e-5, 1)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128, 256])
    buffer_size = trial.suggest_categorical('buffer_size', [int(1e4), int(1e5), int(1e6)])
    learning_starts = trial.suggest_categorical('learning_starts', [0, 1000, 10000, 20000])
    train_freq = trial.suggest_categorical('train_freq', [1, 100, 300])
    gradient_steps = trial.suggest_categorical('gradient_steps', [1, 100, 300])
    ent_coef = trial.suggest_categorical('ent_coef', ['auto', 0.5, 0.1, 0.05, 0.01, 0.0001])

    return {
        'gamma': gamma,
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'buffer_size': buffer_size,
        'learning_starts': learning_starts,
        'train_freq': train_freq,
        'gradient_steps': gradient_steps,
        'ent_coef': ent_coef
    }


HYPERPARAMS_SAMPLER = {
    'ppo2': sample_ppo2_params,
    'sac': sample_sac_params,
    'a2c': sample_a2c_params
}

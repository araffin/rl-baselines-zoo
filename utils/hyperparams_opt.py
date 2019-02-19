import numpy as np

from .utils import kill_env_processes


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

            trial.report(-1 * _locals['self'].best_mean_reward, _locals['self'].num_timesteps)
            if trial.should_prune(_locals['self'].num_timesteps):
                _locals['self'].is_pruned = True
                return False

            return True

        model = model_fn(**kwargs)
        model.trial = trial
        model.learn(n_timesteps, callback=callback)
        is_pruned = model.is_pruned
        best_cost = -1 * model.best_mean_reward
        # Free memory
        kill_env_processes(model.env)
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
    :param trial: (optuna.trial)
    :return: (dict)
    """
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
    n_steps = trial.suggest_categorical('n_steps', [16, 32, 64, 128, 256, 512, 1024, 2048])
    lam = trial.suggest_categorical('lamdba', [0.8, 0.9, 0.92, 0.95, 0.98, 0.99, 1.0])
    # learning_rate = trial.suggest_loguniform('lr', 1e-5, 1)

    if n_steps < batch_size:
        nminibatches = 1
    else:
        nminibatches = int(n_steps / batch_size)

    return {
        'n_steps': n_steps,
        'nminibatches': nminibatches,
        # 'learning_rate': learning_rate,
        'lam': lam
    }


HYPERPARAMS_SAMPLER = {
    'ppo2': sample_ppo2_params
}

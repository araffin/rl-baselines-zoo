import numpy as np
# TODO: remove when stable-baselines > 2.8.0 is out
try:
    # The base noise class was moved to the common folder
    from stable_baselines.common.noise import ActionNoise
except ImportError:
    from stable_baselines.ddpg.noise import ActionNoise


class LinearNormalActionNoise(ActionNoise):
    """
    A gaussian action noise with linear decay for the standard deviation.

    :param mean: (np.ndarray) the mean value of the noise
    :param sigma: (np.ndarray) the scale of the noise (std here)
    :param max_steps: (int)
    :param final_sigma: (np.ndarray)
    """
    def __init__(self, mean, sigma, max_steps, final_sigma=None):
        self._mu = mean
        self._sigma = sigma
        self._step = 0
        self._max_steps = max_steps
        if final_sigma is None:
            final_sigma = np.zeros_like(sigma)
        self._final_sigma = final_sigma

    def __call__(self):
        t = min(1.0, self._step / self._max_steps)
        sigma = (1 - t) * self._sigma + t * self._final_sigma
        self._step += 1
        return np.random.normal(self._mu, sigma)

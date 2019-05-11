import argparse

import matplotlib.pyplot as plt
import numpy as np
import seaborn
from matplotlib.ticker import FuncFormatter
from stable_baselines.results_plotter import load_results, ts2xy


def millions(x, pos):
    """
    Formatter for matplotlib
    The two args are the value and tick position

    :param x: (float)
    :param pos: (int) tick position (not used here
    :return: (str)
    """
    return '{:.1f}M'.format(x * 1e-6)


def moving_average(values, window):
    """
    Smooth values by doing a moving average

    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')


def smooth(xy, window=50):
    x, y = xy
    if y.shape[0] < window:
        return x, y

    original_y = y.copy()
    y = moving_average(y, window)

    if len(y) == 0:
        return x, original_y

    # Truncate x
    x = x[len(x) - len(y):]
    return x, y


# Init seaborn
seaborn.set()

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--log-dirs', help='Log folder(s)', nargs='+', required=True, type=str)
parser.add_argument('--title', help='Plot title', default='Learning Curve', type=str)
parser.add_argument('--smooth', action='store_true', default=False,
                    help='Smooth Learning Curve')
args = parser.parse_args()

results = []
algos = []

for folder in args.log_dirs:
    timesteps = load_results(folder)
    results.append(timesteps)
    if folder.endswith('/'):
        folder = folder[:-1]
    algos.append(folder.split('/')[-1])

min_timesteps = np.inf

# 'walltime_hrs', 'episodes'
for plot_type in ['timesteps']:
    xy_list = []
    for result in results:
        x, y = ts2xy(result, plot_type)
        if args.smooth:
            x, y = smooth((x, y), window=50)
        n_timesteps = x[-1]
        if n_timesteps < min_timesteps:
            min_timesteps = n_timesteps
        xy_list.append((x, y))

    fig = plt.figure(args.title)
    for i, (x, y) in enumerate(xy_list):
        print(algos[i])
        plt.plot(x[:min_timesteps], y[:min_timesteps], label=algos[i], linewidth=2)
    plt.title(args.title)
    plt.legend()
    if plot_type == 'timesteps':
        if min_timesteps > 1e6:
            formatter = FuncFormatter(millions)
            plt.xlabel('Number of Timesteps')
            fig.axes[0].xaxis.set_major_formatter(formatter)

plt.show()

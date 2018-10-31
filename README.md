[![Build Status](https://travis-ci.com/araffin/rl-baselines-zoo.svg?branch=master)](https://travis-ci.com/araffin/rl-baselines-zoo)

# RL Baselines Zoo: a Collection of Trained RL Agents

A collection of trained RL agents, with tuned hyperparameters, using [Stable Baselines](https://github.com/hill-a/stable-baselines).

We are **looking for contributors** to complete the collection!

## Enjoy a Trained Agent


If the trained agent exists, then you can see it in action using:
```
python enjoy.py --algo algo_name --env env_id
```

For example, enjoy A2C on Breakout during 5000 timesteps:
```
python enjoy.py --algo a2c --env BreakoutNoFrameskip-v4 --folder trained_agents/ -n 5000
```

## Train an Agent

The hyperparameters for each environment are defined in `hyperparameters/algo_name.yml`.

If the environment exists in this file, then you can train an agent using:
```
python train.py --algo algo_name --env env_id
```

For example:
```
python train.py --algo ppo2 --env CartPole-v1
```

## Colab Notebooks

You can train agents online using [colab notebook](https://colab.research.google.com/drive/1wUgHJJLvZDBEVYm99pMXkBuNxROxAtGS).

## Installation

```
apt-get install swig cmake libopenmpi-dev zlib1g-dev
pip install stable-baselines==2.1.1 box2d box2d-kengz pyyaml
```

Please see [Stable Baselines README](https://github.com/hill-a/stable-baselines) for alternatives.

Build docker image (CPU):
```
docker build . -f docker/Dockerfile.cpu -t rl-baselines-zoo-cpu
```

Pull built docker image:
```
docker pull araffin/rl-baselines-zoo-cpu
```

## Contributing

If you trained an agent that is not present in the rl zoo, please submit a Pull Request (containing the hyperparameters and the score too).

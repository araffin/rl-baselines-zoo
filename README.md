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
python train.py --algo ppo2 --env CartPole-v1 --tensorboard-log /tmp/stable-baselines/
```

Train for multiple environments (with one call) and with tensorboard logging:
```
python train.py --algo a2c --env MountainCar-v0 CartPole-v1 --tensorboard-log /tmp/stable-baselines/
```

Continue training (here, load pretrained agent for Breakout and continue training for 5000 steps):
```
python train.py --algo a2c --env BreakoutNoFrameskip-v4 -i trained_agents/a2c/BreakoutNoFrameskip-v4.pkl -n 5000
```

## Current Collection

### Atari Games

7 atari games from OpenAI benchmark (NoFrameskip-v4 versions).

|  RL Algo |  BeamRider         | Breakout           | Enduro             |  Pong | Qbert | Seaquest           | SpaceInvaders      |
|----------|--------------------|--------------------|--------------------|-------|-------|--------------------|--------------------|
| A2C      |                    | :heavy_check_mark: |                    |       | :heavy_check_mark: | :heavy_check_mark: |                    |
| ACER     | :heavy_check_mark: |                    |                    |:heavy_check_mark: |       |                    | :heavy_check_mark: |
| ACKTR    |                    |                    |                    |       |       |                    |                    |
| PPO2     |                    |                    | :heavy_check_mark: |       |       |                    |                    |
| DQN     |                    |                    |   |       |       |                    |                    |


### Classic Control Environments

|  RL Algo |  CartPole-v1 | MountainCar-v0 | Acrobot-v1 |  Pendulum-v0 | MountainCarContinuous-v0 |
|----------|--------------|----------------|------------|--------------|--------------------------|
| A2C      | :heavy_check_mark: | :heavy_check_mark:  | :heavy_check_mark: | missing      | missing                  |
| ACER     | :heavy_check_mark: | :heavy_check_mark:  | :heavy_check_mark: | N/A          | N/A                      |
| ACKTR    | :heavy_check_mark: | :heavy_check_mark:  | :heavy_check_mark: | N/A          | N/A                      |
| PPO2     | :heavy_check_mark: | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |:heavy_check_mark:  |
| DQN     | :heavy_check_mark: | :heavy_check_mark:  |  | N/A | N/A  |
| DDPG     |  N/A |  N/A  | N/A| :heavy_check_mark: | :heavy_check_mark:  |


### Box2D Environments

|  RL Algo |  BipedalWalker-v2 | LunarLander-v2 | LunarLanderContinuous-v2 |  BipedalWalkerHardcore-v2 | CarRacing-v0 |
|----------|--------------|----------------|------------|--------------|--------------------------|
| A2C      | missing | :heavy_check_mark:  | missing | missing      | missing                  |
| ACER     | N/A | :heavy_check_mark:      | N/A | N/A          | N/A                      |
| ACKTR    | N/A | :heavy_check_mark:      | N/A | N/A          | N/A                      |
| PPO2     | :heavy_check_mark: | :heavy_check_mark:  | :heavy_check_mark: | missing | missing  |
| DQN     | N/A | :heavy_check_mark: | N/A | N/A | N/A  |
| DDPG     |  | N/A | :heavy_check_mark: |  |   |


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

# Spinning-Up: Spaceship-Landing, Robot-Walking and Atari Games with Reinforcement Learning

## Abstract
Following the suggested educational path of OpenAI’s Spinning-Up series, in this project well-known deep reinforcement learning agents are implemented from scratch using the Tensorflow-framework.
They are tested on classical OpenAI-Gym-environments and benchmarked against their respective stable-baselines-3 implementations. Among others, in this project, reinforcement learning is successfully
used to land a spaceship on the moon (Lunar-Lander environment), make a robot walk (Bipedal Walker environment) and play classical computer games from raw pixel-input (Atari Breakout environment).


## Demo
(GIFs may take a few seconds to load)

![after_250000_steps](https://github.com/TomGoesGitHub/Spinning-Up-in-Reinforcement-Learning/assets/81027049/36e9dd13-662a-4db3-b177-540e93c8b223)

![after_1800000_steps](https://github.com/TomGoesGitHub/Spinning-Up-in-Reinforcement-Learning/assets/81027049/5c9e5768-b46d-431e-bd62-1b95351f9ca7)

![after_4550000_steps](https://github.com/TomGoesGitHub/Spinning-Up-in-Reinforcement-Learning/assets/81027049/c37b1506-271a-4837-a142-edfcfd2055dc)

## Architectures and Agents
Deep Reinforcement Learning (DRL) encompasses a spectrum of approaches to learn within the framework of Markov decicion processes. Three fundamental paradigms within DRL—namely value-based methods,
policy-learning-based strategies, and actor-critic architectures—have emerged as cornerstones in addressing the challenges of decision-making under uncertainty. All approaches use deep neural networks
as function approximators internally.

In this project the most popular agents are implemented for each of those 3 approaches: DQN as classical value-based approach, REINFORCE as vanilla policy gradient method and A2C and PPO as popular actor-critics.

## Implementation-Highlights
- OOP: All agents share one API and inherite from one shared mother class, which specifies the core behavior.
- All Agents make use of vectorized environments, which speed up the data-collection process significantly via parallelzation.
- Function approximation is strictly seperated from action selection by implicitly implementing policies as own classes (e.g. Epsilon Greedy, Boltzmann, Gaussian). This makes the library more modular...
- Within the Tensorflow-2 framework the Gradient-Tape-API is used for gradient comupation. (Explicitly the Keras-API was not used since Keras is not naturally suited for RL but rather for supervised learning, in my opinion)
- Utils: Callbacks / Schedules / Visualization

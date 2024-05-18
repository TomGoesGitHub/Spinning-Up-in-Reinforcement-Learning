import sys
import os

import gymnasium as gym
import numpy as np
import tensorflow as tf

sys.path.append(os.path.join(*3*[os.pardir]))
from learning.callbacks import PerformanceEvaluationCallback, RenderEnvironmentCallback, SaveModelCallback
from learning.agents import DQNAgent
from learning.policies import EpsilonGreedyPolicy
from learning.utils.common import create_mlp
from learning.utils.schedules import LinearSchedule


def main():
    env = gym.vector.make('LunarLander-v2', num_envs=2)

    epsilon_shedule = LinearSchedule(initial_value=1, c=-0.9/5e4, bounds=(0.1, 1))
    policy = EpsilonGreedyPolicy(epsilon_shedule)

    q_network = create_mlp(input_shape=env.single_observation_space.shape,
                               output_shape=env.single_action_space.n,
                               network_architecture=[256, 256],
                               activation='tanh',
                               output_activation='linear')

    callbacks = [
        PerformanceEvaluationCallback(env=gym.make('LunarLander-v2', render_mode=None),
                                      freq=50000, n_evals=5, filepath='logging.csv'),
        RenderEnvironmentCallback(env=gym.make('LunarLander-v2', render_mode='rgb_array'),
                                  directory='animated_rollouts', freq=int(1e5)),
        SaveModelCallback(dir='checkpoints')
    ] 

    agent = DQNAgent(env, policy, q_network, batch_size=128, freq_target_update=250,
                     optimizer=tf.keras.optimizers.Adam(learning_rate=6.3e-4), buffer_size=50000,
                     callbacks=callbacks)

    agent.train(n_epochs=5, steps_per_epoch=int(1e5))

if __name__ == '__main__':
    main()
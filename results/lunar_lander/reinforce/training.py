import sys
import os

import gymnasium as gym
import numpy as np
import tensorflow as tf

sys.path.append(os.path.join(*3*[os.pardir]))
from learning.callbacks import PerformanceEvaluationCallback, RenderEnvironmentCallback, SaveModelCallback
from learning.agents.reinforce import RichSuttonReinforceAgent as ReinforceAgent
from learning.policies import BoltzmannPolicy
from learning.utils.common import create_mlp
from learning.utils.schedules import LinearSchedule


def main():
    env = gym.vector.make('LunarLander-v2', num_envs=12, render_mode=None)

    temperature_schedule = LinearSchedule(initial_value=1., c=-0.9/3e5, bounds=(0.1, 1))
    policy = BoltzmannPolicy(temperature_schedule)
    policy_network = create_mlp(input_shape=env.single_observation_space.shape,
                                                output_shape=env.single_action_space.n,
                                                output_activation='linear')

    callbacks = [
        PerformanceEvaluationCallback(env=gym.make('LunarLander-v2'), freq=20000,
                                      n_evals=5, filepath='logging.csv'),
        # RenderEnvironmentCallback(env=gym.make('LunarLander-v2', render_mode='rgb_array'),
        #                           directory='animated_rollouts', freq=int(5e4)),
        # SaveModelCallback(dir='checkpoints')
    ] 

    agent = ReinforceAgent(env, policy, policy_network, train_freq=2000, gamma=0.99, callbacks=callbacks)
    agent.train(n_epochs=5, steps_per_epoch=int(1e6))
    agent.save_weights('reinforce_on_lunar_lander.weights.h5')


if __name__ == '__main__':
    main()
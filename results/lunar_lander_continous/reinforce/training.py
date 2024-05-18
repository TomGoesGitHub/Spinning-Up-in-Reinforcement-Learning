import sys
import os

import gymnasium as gym
import numpy as np
import tensorflow as tf

sys.path.append(os.path.join(*3*[os.pardir]))
from learning.callbacks import PerformanceEvaluationCallback, RenderEnvironmentCallback, SaveModelCallback
from learning.agents.reinforce import RichSuttonReinforceAgent as ReinforceAgent
from learning.policies import GaussianPolicy
from learning.utils.common import create_mlp

def main():
    env = gym.vector.make('LunarLander-v2', num_envs=12, render_mode=None, continuous=True)
    
    constant_scale = 0.05 * (env.single_action_space.high - env.single_action_space.low) # todo/note: learnable scale would require ReinforceAgent to include entropy-bonus
    policy = GaussianPolicy(environment=env, scale=constant_scale, act_deterministic_if_not_training=True)
    policy_network = create_mlp(input_shape=env.single_observation_space.shape,
                                output_shape=policy.expected_input_shape[-1],
                                output_activation='sigmoid')

    callbacks = [
        PerformanceEvaluationCallback(env=gym.make('LunarLander-v2', continuous=True), freq=20000,
                                      n_evals=5, filepath='logging.csv'),
        # RenderEnvironmentCallback(env=gym.make('LunarLander-v2', render_mode='rgb_array'),
        #                           directory='animated_rollouts', freq=int(5e4)),
        SaveModelCallback(dir='checkpoints')
    ] 

    agent = ReinforceAgent(env, policy, policy_network, train_freq=4000, gamma=0.99, callbacks=callbacks)
    agent.train(n_epochs=5, steps_per_epoch=int(1e6))
    agent.save_weights('reinforce_on_lunar_lander.weights.h5')


if __name__ == '__main__':
    main()
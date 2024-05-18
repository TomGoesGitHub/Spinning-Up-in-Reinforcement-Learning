import sys
import os

import gymnasium as gym
import numpy as np

sys.path.append(os.path.join(*3*[os.pardir]))
from learning.callbacks import PerformanceEvaluationCallback, RenderEnvironmentCallback, SaveModelCallback
from learning.agents.actor_critic import PpoTf2
from learning.policies import GaussianPolicy
from learning.utils.common import create_mlp

    
def main():
    env = gym.vector.make('BipedalWalker-v3', num_envs=4, render_mode=None)

    constant_scale = 0.2 * (env.single_action_space.high - env.single_action_space.low)
    policy = GaussianPolicy(environment=env, scale=constant_scale, act_deterministic_if_not_training=True)

    critic_network = create_mlp(input_shape=env.single_observation_space.shape,
                                                output_shape=1)
    actor_network = create_mlp(input_shape=env.single_observation_space.shape,
                                                output_shape=policy.expected_input_shape[-1],
                                                output_activation='linear')

    agent = PpoTf2(env, policy, actor_network, critic_network,
                epsilon_clip=0.1, n_steps=8*2048, minibatch_size=64, entropy_coef=0.001,
                epochs_actorupdate=10, gamma=0.99, lambda_gae=0.95)
    agent.load_weights(os.path.join('checkpoints', 'after_epoch_001.weights.h5'))

    print('Done!')

if __name__ == '__main__':
    main()
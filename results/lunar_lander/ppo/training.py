import sys
import os

import gymnasium as gym
import numpy as np

sys.path.append(os.path.join(*3*[os.pardir]))
from learning.callbacks import PerformanceEvaluationCallback, RenderEnvironmentCallback, SaveModelCallback
from learning.agents.actor_critic import PpoTf2
from learning.policies import BoltzmannPolicy
from learning.utils.common import create_mlp
from learning.utils.schedules import LinearSchedule


def main():
    env = gym.vector.make('LunarLander-v2', num_envs=4, render_mode=None)

    temperature_schedule = LinearSchedule(initial_value=1., c=-0.9/3e5, bounds=(0.1, 1))
    policy = BoltzmannPolicy(temperature_schedule)
    critic_network = create_mlp(input_shape=env.single_observation_space.shape,
                                                  output_shape=1)
    actor_network = create_mlp(input_shape=env.single_observation_space.shape,
                                                output_shape=env.single_action_space.n,
                                                output_activation='linear')

    callbacks = [
        PerformanceEvaluationCallback(env=gym.make('LunarLander-v2'), freq=4*5000,
                                      n_evals=5, filepath='logging.csv'),
        RenderEnvironmentCallback(env=gym.make('LunarLander-v2', render_mode='rgb_array'),
                                  directory='animated_rollouts', freq=int(5e4)),
        SaveModelCallback(dir='checkpoints')
    ] 

    agent = PpoTf2(env, policy, actor_network, critic_network,
                epsilon_clip=0.2, n_steps=1024, minibatch_size=64, entropy_coef=0.01,
                epochs_actorupdate=4, gamma=0.999, lambda_gae=0.98, callbacks=callbacks)
    agent.train(n_epochs=3, steps_per_epoch=int(1e5))
    agent.save_weights('ppo_on_lunar_lander.weights.h5')


if __name__ == '__main__':
    main()
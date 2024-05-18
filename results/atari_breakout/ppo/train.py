import sys
import os

import gymnasium as gym
import numpy as np

sys.path.append(os.path.join(*3*[os.pardir]))
from learning.callbacks import PerformanceEvaluationCallback, RenderEnvironmentCallback, SaveModelCallback
from learning.agents.actor_critic import PpoTf2
from learning.policies import BoltzmannPolicy
from learning.utils.common import create_cnn
from learning.utils.schedules import LinearSchedule


def main():
    
    def make_env(env_name, **kwargs):
        '''Returns a callable. The callable does not require any arguments and creates the desired
        Atari-Breakout-Environment Additional Wrappers are added to induce desired behavior.'''
        def _make():
            env = gym.make(env_name, **kwargs)
            env = gym.wrappers.AtariPreprocessing(env, noop_max=1, terminal_on_life_loss=True, scale_obs=True,grayscale_newaxis=True)
            env = gym.wrappers.FrameStack(env, num_stack=2)
            return env
        return _make

    env = gym.vector.AsyncVectorEnv(env_fns=[make_env('ALE/Breakout-v5', render_mode=None, frameskip=1)
                                             for _ in range(8)])

    temperature_schedule = LinearSchedule(initial_value=1., c=-0.9/3e6, bounds=(0.1, 1))
    policy = BoltzmannPolicy(temperature_schedule)

    critic_network = create_cnn(input_shape=env.single_observation_space.shape,
                                output_shape=1, network_architecture=[8, 16, 32, 32, 32])
    actor_network = create_cnn(input_shape=env.single_observation_space.shape,
                               output_shape=env.single_action_space.n,
                               network_architecture=[8, 16, 32, 32, 32],
                               output_activation='linear')

    callbacks = [
        PerformanceEvaluationCallback(env=make_env('ALE/Breakout-v5', render_mode=None, frameskip=1)(),
                                      freq=50000, n_evals=5, filepath='logging.csv'), # todo: 8*5000
        RenderEnvironmentCallback(env=make_env('ALE/Breakout-v5', render_mode='rgb_array', frameskip=1)(),
                                  directory='animated_rollouts', freq=int(1e5), max_rollout_len=1000, n=20), #todo
        SaveModelCallback(dir='checkpoints')
    ] 

    agent = PpoTf2(env, policy, actor_network, critic_network,
                epsilon_clip=0.2, n_steps=4096, minibatch_size=64, entropy_coef=0.001,
                epochs_actorupdate=10, gamma=0.99, lambda_gae=0.7, callbacks=callbacks)
    agent.load_weights(os.path.join('checkpoints', 'after_epoch_002.weights.h5'), by_name=True)

    agent.train(n_epochs=5, steps_per_epoch=int(1e6))
    agent.save_weights('ppo_on_bipedal_walker.weights.h5')


if __name__ == '__main__':
    main()
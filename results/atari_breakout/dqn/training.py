import sys
import os

import gymnasium as gym
import numpy as np
import tensorflow as tf
import stable_baselines3

sys.path.append(os.path.join(*3*[os.pardir]))
from learning.callbacks import PerformanceEvaluationCallback, RenderEnvironmentCallback, SaveModelCallback
from learning.agents import DQNAgent
from learning.policies import EpsilonGreedyPolicy
from learning.utils.common import get_atari_dqn_cnn
from learning.utils.schedules import LinearSchedule


def make_env(env_name, **kwargs):
    '''Returns a callable. The callable does not require any arguments and creates the desired
    Atari-Breakout-Environment Additional Wrappers are added to induce desired behavior.'''
    def _make():
        env = gym.make(env_name, frameskip=1, repeat_action_probability=0, **kwargs)
        env = gym.wrappers.AtariPreprocessing(env, noop_max=1, terminal_on_life_loss=True,
                                              scale_obs=True, grayscale_newaxis=True)
        env = gym.wrappers.FrameStack(env, num_stack=4)
        env = stable_baselines3.common.atari_wrappers.FireResetEnv(env)
        return env
    return _make

def main():
    env = gym.vector.AsyncVectorEnv(env_fns=[make_env('ALE/Breakout-v5', render_mode=None)])

    epsilon_shedule = LinearSchedule(initial_value=1, c=-0.9/1e6, bounds=(0.1, 1))
    policy = EpsilonGreedyPolicy(epsilon_shedule)

    q_network = get_atari_dqn_cnn(input_shape=env.single_observation_space.shape,
                                  output_shape=env.single_action_space.n)
    

    callbacks = [
        PerformanceEvaluationCallback(env=make_env('ALE/Breakout-v5', render_mode=None)(),
                                      freq=50000, n_evals=5, filepath='logging.csv'),
        RenderEnvironmentCallback(env=make_env('ALE/Breakout-v5', render_mode='rgb_array')(),
                                  directory='animated_rollouts', freq=int(50000), max_rollout_len=1000),
        SaveModelCallback(dir='checkpoints')
    ] 

    optimizer=tf.keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0)
    agent = DQNAgent(env, policy, q_network, batch_size=32, freq_target_update=10000,
                     optimizer=optimizer, buffer_size=int(1e5), callbacks=callbacks)

    agent.train(n_epochs=10, steps_per_epoch=int(5e5))

if __name__ == '__main__':
    main()
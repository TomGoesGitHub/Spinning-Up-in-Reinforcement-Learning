import gymnasium as gym

import os 
import sys

from training import make_env
sys.path.append(os.path.join(*3*[os.pardir]))
from learning.callbacks import PerformanceEvaluationCallback, RenderEnvironmentCallback, SaveModelCallback
from learning.agents.actor_critic import PpoTf2
from learning.policies import EpsilonGreedyPolicy
from learning.utils.common import get_atari_dqn_cnn
from learning.utils.schedules import LinearSchedule
from learning.utils.visualization.rollout_animation import RolloutAnimation

if __name__ == '__main__':
    env = make_env('ALE/Breakout-v5', render_mode='rgb_array')()
    # env = gym.make('ALE/Breakout-v5', render_mode='rgb_array')

    epsilon_shedule = LinearSchedule(initial_value=1, c=-0.9/1e6, bounds=(0.1, 1))
    policy = EpsilonGreedyPolicy(epsilon_shedule)

    q_network = get_atari_dqn_cnn(input_shape=env.observation_space.shape,
                                    output_shape=env.action_space.n)
    old_weights = q_network.get_weights()
    q_network.load_weights(os.path.join('checkpoints', 'after_epoch_001_return=11.weights.h5'), by_name=True)
    new_weights = q_network.get_weights()

    agent = PPOTf2(env, policy, q_network)

    animation = RolloutAnimation(env)
    animation.animate('tmp.gif')
    print('Done!')
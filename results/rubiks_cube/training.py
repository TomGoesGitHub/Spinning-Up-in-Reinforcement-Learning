import sys
import os

import tensorflow as tf
import gymnasium as gym

sys.path.append(os.path.join(*2*[os.pardir]))
from environments.rubiks_cube import RubiksCube
from environments.rubiks_cube.callbacks import DistortionDepthCallback
from learning.agents import DQNAgent
from learning.agents.actor_critic import PpoTf2
from learning.policies import EpsilonGreedyPolicy, BoltzmannPolicy
from learning.utils.schedules import ConstantSchedule, LinearSchedule
from learning.utils.common import create_mlp
from learning.callbacks import PerformanceEvaluationCallback, SaveModelCallback


# def main():
#     env = RubiksCube(distortion_depth=1)
#     vector_env = gym.vector.AsyncVectorEnv([lambda : env])

#     epsilon_shedule = ConstantSchedule(initial_value=0.1)
#     policy = EpsilonGreedyPolicy(epsilon_shedule)

#     q_network = create_mlp(input_shape=env.observation_space.shape, output_shape=env.action_space.n,
#                            network_architecture=[64,64], activation='relu', output_activation='sigmoid')
    
#     callbacks = [PerformanceEvaluationCallback(env, freq=1000, filepath='logging.csv'),
#                  DistortionDepthCallback(env, freq=10000),
#                  SaveModelCallback(dir='checkpoints')]

#     agent = DQNAgent(vector_env, policy, q_network, optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
#                      gamma=0.99, batch_size=1024, freq_target_update=250, buffer_size=5000000, callbacks=callbacks)
    
#     agent.train(n_epochs=5, steps_per_epoch=int(1e6))


def main():
    env = RubiksCube(distortion_depth=1)
    vector_env = gym.vector.AsyncVectorEnv([lambda : env])

    temperature_schedule = LinearSchedule(initial_value=0.7, c=-0.65/3e5, bounds=(0.05,0.7))
    policy = BoltzmannPolicy(temperature_schedule)

    critic_network = create_mlp(input_shape=env.observation_space.shape, output_shape=1,
                           network_architecture=[64,64,64], activation='tanh', output_activation='relu')
    actor_network = create_mlp(input_shape=env.observation_space.shape, output_shape=env.action_space.n,
                           network_architecture=[64,64,64], activation='tanh', output_activation='linear')
    
    callbacks = [DistortionDepthCallback(env, freq=5000, filepath='logging.csv')]

    agent = PpoTf2(vector_env, policy, actor_network, critic_network,
                optimizer_actor=tf.keras.optimizers.Adam(learning_rate=3e-4),# clipnorm=0.5),
                optimizer_critic=tf.keras.optimizers.Adam(learning_rate=3e-4),# clipnorm=0.5),
                epsilon_clip=0.1, n_steps=2048, minibatch_size=64, entropy_coef=0,
                epochs_actorupdate=4, gamma=0.99, lambda_gae=0.98,
                callbacks=callbacks)
    
    agent.train(n_epochs=5, steps_per_epoch=int(1e6))


if __name__ == '__main__':
    main()
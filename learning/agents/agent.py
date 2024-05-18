from ..callbacks import CallbackList
from ..utils.common import MdpDataset
import tensorflow as tf
import numpy as np
import gymnasium as gym

class Agent(tf.keras.Model):
    '''Abstract reinforcement-learning agent class.'''
    def __init__(self, environment, policy, gamma=0.99, callbacks=[]):
        super().__init__()
        # create a vector env # todo: how does this work with custom envs?
        if not isinstance(environment.unwrapped, gym.vector.AsyncVectorEnv):
            environment = gym.vector.AsyncVectorEnv(2*[lambda : environment])
            # note: the whole frame work assumes vector-environments (one extra batch dim)
        self.environment = environment 
        
        self.gamma = gamma
        self.policy = policy
        self.callback = CallbackList(callbacks)
        self.callback.init_callback(agent=self)
        self.memory = MdpDataset()

    def act(self, observation, training=False):
        raise NotImplementedError
    
    def call(self, observation, training=False):
        self.act(observation, training)
    
    def _trigger_train_step(self, step_index):
        raise NotImplementedError
    
    def _train_step(self, step_index):
        raise NotImplementedError
    
    def _handle_memory(self, triggered_training):
        raise NotImplementedError
    
    def _step_environment(self, action):
        self.callback._on_step_start()
        observation_next, reward, terminated, truncated, info = self.environment.step(action)
        self.callback._on_step_end()
        return observation_next, reward, terminated, truncated, info

    def _reset_environment(self):
        self.callback._on_rollout_end()
        observation, info = self.environment.reset()
        self.callback._on_rollout_start()
        return observation, info

    def _step(self, observation):
        action = self.act(tf.constant(observation), training=True)
        action = action.numpy()
        observation_next, reward, terminated, truncated, info = self._step_environment(action)
        
        datapoint = (observation, action, observation_next, reward, terminated, truncated)
        self.memory.remember(datapoint)
        
        # done = np.logical_or(terminated, truncated)
        return observation_next
    
    def train(self, n_epochs, steps_per_epoch):
        self.callback._on_training_start(locals=locals(), globals=globals())
        for epoch in range(n_epochs):
            self.callback._on_epoch_start()
            observation, _ = self.environment.reset()
            for step_index in range(1, steps_per_epoch):
                observation = self._step(observation)
                # todo: I think vector envs do auto-reset
                # if done:
                #     observation, _ = self._reset_environment()
                triggered_training = self._trigger_train_step(step_index)
                if triggered_training:
                    self._train_step(step_index)
                self._handle_memory(triggered_training)
            self.callback._on_epoch_end()
        self.callback._on_training_end()
from ...utils.common import MdpDataset
from ..agent import Agent
import tensorflow as tf
import numpy as np

class ActorCritic(Agent):
    def __init__(self, environment, policy, actor_network, critic_network, gamma, callbacks):
        super().__init__(environment, policy, gamma, callbacks)
        self.memory = MdpDataset()
        self.build(actor_network, critic_network)

    def build(self, actor_network, critic_network):
        self.actor_network = actor_network
        self.critic_network = critic_network     
        input_shape = [None, *self.environment.single_observation_space.shape]
        super().build(input_shape)

    @tf.function
    def act(self, observation, training=False): # note: observation is a tensor
        actor_network_output = self.actor_network(observation) 
        action = self.policy.act(actor_network_output, training)
        # batch_dim = action.shape[0]
        # action = tf.squeeze(action)
        # action = tf.reshape(action, [batch_dim, ...])
        return action
    
    # def critic(self, observation):
    #     return self.critic_network(observation) # todo: integrate into subclasses

    def _critic_loss(self):
        raise NotImplementedError # needs to be implemented for keras-style only
    
    def _actor_loss(self):
        raise NotImplementedError # needs to be implemented for keras-style only
    

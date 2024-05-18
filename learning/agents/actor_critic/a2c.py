import tensorflow as tf
import numpy as np

from .actor_critic import ActorCritic
from ...utils.common import generalized_advantage_estimate

class A2cKeras(ActorCritic):
    def __init__(self, environment, policy, actor_network, critic_network, 
                 gamma=0.99, t_max=5, lambda_gae=0, callbacks=[]):
        super().__init__(environment, policy, actor_network, critic_network, gamma, callbacks)
        self.lambda_gae = lambda_gae
        self.t_max = t_max

    def _handle_memory(self, triggered_training):
        # clear memory after every training step
        if self.triggered_training:
            self.memory.clear()
    
    def critic_loss(self, td_target, predicted_values):
        td_error = tf.keras.losses.mean_squared_error(td_target, predicted_values)
        return td_error

    def actor_loss(self, actions, actor_output):
        probas = self.policy.proba(actor_output, actions)
        negative_log_probas = - tf.math.log(probas)
        return negative_log_probas
    
    def _trigger_train_step(self, step):
        done = self.memory.truncations[-1] or self.memory.terminals[-1]
        return bool(done) or (len(self.memory) >= self.t_max)

    def _train_step(self):
        observations, actions, observations_next, rewards, terminals, _ = self.memory.to_tensor()

        values = tf.squeeze(self.critic_network(observations))
        values_next = tf.squeeze(self.critic_network(observations_next)) * (1-terminals)
        td_targets = rewards + self.gamma * values_next
        td_errors = td_targets - values
        advantages = generalized_advantage_estimate(td_errors, self.lambda_gae, self.gamma)

        self.critic_network.train_on_batch(x=observations_next, y=td_targets)
        self.actor_network.train_on_batch(x=observations, y=actions, sample_weight=advantages)


class A2CAgent(ActorCritic):
    def __init__(self, environment, policy, actor_network, critic_network,
                 optimizer_actor=tf.keras.optimizers.Adam(),
                 optimizer_critic=tf.keras.optimizers.Adam(),
                 gamma=0.99, t_max=5, lambda_gae=0, normalize_adv=False,
                 entropy_coef=0., semi_gradient=False, callbacks=[]):
        super().__init__(environment, policy, actor_network, critic_network, gamma, callbacks)
        self.t_max=t_max
        self.lambda_gae = lambda_gae  # lambda parameter for GAE
        self.normalize_adv = normalize_adv
        self.entropy_coef = entropy_coef
        self.semi_gradient = semi_gradient
        self.optimizer_actor = optimizer_actor
        self.optimizer_critic = optimizer_critic
        
    def _handle_memory(self, triggered_training):
        # clear memory after every training step
        if triggered_training:
            self.memory.clear()

    def _critic_loss(self, observations, observations_next, rewards, terminals):
        values = tf.squeeze(self.critic_network(observations))
        values_next = tf.squeeze(self.critic_network(observations_next)) * (1-terminals)
        if self.semi_gradient:
            values_next = tf.stop_gradient(values_next)
        td_targets = rewards + self.gamma * values_next
        td_errors = td_targets - values
        critic_loss = tf.reduce_mean(tf.square(td_errors))
        return critic_loss

    def compute_advantages(self, observations, observations_next, rewards, terminals, truncations):
        values = tf.squeeze(self.critic_network(observations))
        values_next = tf.squeeze(self.critic_network(observations_next)) * (1-terminals)
        deltas = rewards + self.gamma * values_next - values # bellman errors
        dones = tf.clip_by_value(terminals+truncations, 0, 1)
        
        advantages = generalized_advantage_estimate(deltas, self.gamma, self.lambda_gae, dones)
        if self.normalize_adv:
            advantages = (advantages - tf.math.reduce_mean(advantages, axis=-1)) \
                          / (tf.math.reduce_std(advantages, axis=-1) + 1e-10)
        return advantages

    def _actor_loss(self, observations, actions, observations_next, rewards, terminals, truncations):       
        advantages = self.compute_advantages(observations, observations_next, rewards, terminals, truncations)

        # regular loss
        policy_inputs = self.actor_network(observations)
        action_probs = self.policy.proba(policy_inputs, actions)
        a2c_loss = -tf.reduce_mean(tf.math.log(action_probs + 1e-8) * advantages)
        
        # entropy bonus
        entropies = self.policy.distribution(policy_inputs).entropy()
        entropy_loss = - self.entropy_coef * tf.reduce_mean(entropies)
        
        actor_loss = a2c_loss + entropy_loss
        return actor_loss

    @tf.function
    def update_critic(self, observations, observations_next, rewards, terminals):
        with tf.GradientTape() as tape:
            critic_loss = self._critic_loss(observations, observations_next, rewards, terminals)
        critic_gradients = tape.gradient(target=critic_loss, sources=self.critic_network.trainable_variables)
        self.optimizer_critic.apply_gradients(zip(critic_gradients, self.critic_network.trainable_variables))

    @tf.function
    def update_actor(self, observations, actions, observations_next, rewards, terminals, truncations):
        with tf.GradientTape() as tape:
            actor_loss = self._actor_loss(observations, actions, observations_next,
                                          rewards, terminals, truncations)
        actor_gradients = tape.gradient(target=actor_loss, sources=self.actor_network.trainable_variables)
        self.optimizer_actor.apply_gradients(zip(actor_gradients, self.actor_network.trainable_variables))

    def _train_step(self, step):
        # unpack data
        observations, actions, observations_next, rewards, terminals, truncations = self.memory.get_all()

        # note: gradient computation is excluded in own method to make use of tf.function speed-up
        self.update_critic(observations, observations_next, rewards, terminals)
        self.update_actor(observations, actions, observations_next, rewards, terminals, truncations)
        

    def _trigger_train_step(self, step):
        # done = self.memory.truncations[-1] or self.memory.terminals[-1]
        # return bool(done) or (len(self.memory) >= self.t_max)
        return step % self.t_max == 0
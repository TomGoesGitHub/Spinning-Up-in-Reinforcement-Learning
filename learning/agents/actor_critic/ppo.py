from .actor_critic import ActorCritic
from ...utils.common import generalized_advantage_estimate
import tensorflow as tf
import tensorflow_probability as tfp
import copy

class PpoTf2(ActorCritic):
    def __init__(self, environment, policy, actor_network, critic_network,
                optimizer_actor=tf.keras.optimizers.Adam(learning_rate=3e-4),# clipnorm=0.5),
                optimizer_critic=tf.keras.optimizers.Adam(learning_rate=3e-4),# clipnorm=0.5),
                epsilon_clip=0.2, n_steps=2048, minibatch_size=64, entropy_coef=0.01,
                epochs_actorupdate=4, gamma=0.99, lambda_gae=0.98,
                callbacks=[]):
        super().__init__(environment, policy, actor_network, critic_network, gamma, callbacks)
                
        # data-collection
        self.n_steps = n_steps
        self.lambda_gae = lambda_gae
        
        # ppo specific stabilization
        self.epsilon_clip = tf.constant(epsilon_clip, dtype=tf.float32)
        self.entropy_coef = tf.constant(entropy_coef, dtype=tf.float32)
        self.minibatch_size = minibatch_size # for actor learning
        self.epochs_actorupdate = epochs_actorupdate # for actor learning
        
        # gradient learning rules
        self.optimizer_actor = optimizer_actor
        self.optimizer_critic = optimizer_critic
    
    def _trigger_train_step(self, step):
        return len(self.memory) > self.n_steps
    
    def _handle_memory(self, triggered_training):
        # clear memory after every training step
        if triggered_training:
            self.memory.clear()
    
    @tf.function
    def _compute_critic_gradient(self, observations, targets): # observations_next, rewards, terminals):
        with tf.GradientTape() as tape_critic:
            values = tf.squeeze(self.critic_network(observations))
            # values_next = tf.squeeze(self.critic_network(observations_next)) * (1-terminals)
            # targets = rewards + self.gamma * values_next
            deltas =  targets - values # bellman errors
            critic_loss = tf.reduce_mean(tf.square(deltas), axis=-1) # MSE
        grad_critic = tape_critic.gradient(target=critic_loss, sources=self.critic_network.trainable_variables)
        self.optimizer_critic.apply_gradients(zip(grad_critic, self.critic_network.trainable_variables))

    @tf.function
    def _compute_actor_gradient(self, observations, actions, probas_old, advantages, step_idx):
        with tf.GradientTape() as tape_actor:
            policy_input=self.actor_network(observations)

            # surrogate objective
            probas_new = self.policy.proba(policy_input, actions) # using current (updated) policy
            r = probas_new / (probas_old + 1e-10)
            clipped = tf.clip_by_value(r, 1-self.epsilon_clip, 1+self.epsilon_clip)
            ppo_loss = - tf.reduce_mean(tf.reduce_min([r*advantages, clipped*advantages], axis=0), axis=-1)

            # entropy bonus
            entropies = self.policy.distribution(policy_input).entropy()
            entropy_loss = - self.entropy_coef * tf.reduce_mean(entropies)
     
            loss = ppo_loss + entropy_loss

        grad_actor = tape_actor.gradient(loss, self.actor_network.trainable_variables)
        self.optimizer_actor.apply_gradients(zip(grad_actor, self.actor_network.trainable_variables))    
    
    @tf.function
    def compute_advantages(self, observations, observations_next, rewards, terminals, truncations):
        values = tf.squeeze(self.critic_network(observations))
        values_next = tf.squeeze(self.critic_network(observations_next)) * (1-terminals)
        deltas = rewards + self.gamma * values_next - values # bellman errors
        dones = tf.clip_by_value(terminals+truncations, 0, 1)
        
        advantages = generalized_advantage_estimate(deltas, self.gamma, self.lambda_gae, dones)
        targets = advantages + values # see David Silver Lecture 4, 'Telescoping TD(lambda)'-slide
        
        advantages = (advantages - tf.math.reduce_mean(advantages, axis=-1)) / (tf.math.reduce_std(advantages, axis=-1) + 1e-10)
        return advantages, targets

    #@tf.function
    def create_minibatches_critic(self, critic_data):
        as_ds = tuple(tf.data.Dataset.from_tensor_slices(x) for x in critic_data)
        zipped = tf.data.Dataset.zip(as_ds)
        c = zipped.cardinality()
        minibatches = zipped.shuffle(buffer_size=c).batch(self.minibatch_size, drop_remainder=True)
        return minibatches
    
    #@tf.function
    def create_minibatches_actor(self, actor_data):
        as_ds = tuple(tf.data.Dataset.from_tensor_slices(x) for x in actor_data)
        zipped = tf.data.Dataset.zip(as_ds)
        c = zipped.cardinality()
        minibatches = zipped.shuffle(buffer_size=c).batch(self.minibatch_size, drop_remainder=True)
        return minibatches
    # todo: can both minibatch-methods be combined in a single one? (actor_data and critic_data have different structure)
    
    def _train_step(self, step_idx):
        '''Minibatches as in paper, clean'''
        #* gather data
        observations, actions, observations_next, rewards, terminals, truncations = self.memory.get_all()
        advantages, mc_returns = self.compute_advantages(observations, observations_next, rewards, terminals, truncations)
        probas_old = self.policy.proba(policy_input=self.actor_network(observations), actions=actions)

        #* critic update
        critic_data = [observations, mc_returns] # observations_next, rewards, terminals]
        minibatches = self.create_minibatches_critic(critic_data)
        for _ in range(self.epochs_actorupdate):
            for observations_mb, mc_returns_mb in minibatches: #observations_next_mb, rewards_mb, terminals_mb in minibatches:
                self._compute_critic_gradient(observations_mb, mc_returns_mb) #observations_next_mb, rewards_mb, terminals_mb)
        
        #* actor update
        actor_data = [observations, actions, probas_old, advantages]
        minibatches = self.create_minibatches_actor(actor_data)
        for _ in range(self.epochs_actorupdate):
            for observations_mb, actions_mb, probas_old_mb, advantages_mb in minibatches:
                self._compute_actor_gradient(observations_mb, actions_mb, probas_old_mb, advantages_mb, step_idx)
    
    # def _train_step(self, step_idx):
    #     '''Minibatches as in paper'''
    #     # unpack from memory
    #     observations, actions, observations_next, rewards, terminals, _ = self.memory.get_all()

    #     # using the old policy
    #     probas_old = self.policy.proba(policy_input=self.actor_network(observations),
    #                                    actions=actions)        
        
    #     # create minibatches
    #     as_ds = tuple(tf.data.Dataset.from_tensor_slices(x)
    #                   for x in [observations, actions, observations_next, \
    #                             rewards, terminals, probas_old])
    #     zipped = tf.data.Dataset.zip(as_ds)
    #     minibatches = zipped.batch(self.minibatch_size, drop_remainder=True) #todo
        
    #     for _ in range(self.epochs_actorupdate):
    #         for observations_mb, actions_mb, observations_next_mb, rewards_mb, terminals_mb, \
    #          probas_old_mb in minibatches:
    #             # critic update
    #             grad_critic = self._compute_critic_gradient(observations_mb, observations_next_mb, rewards_mb, terminals_mb)
    #             self.optimizer_critic.apply_gradients(zip(grad_critic, self.critic_network.trainable_variables))
        
    #     for _ in range(self.epochs_actorupdate):
    #         for observations_mb, actions_mb, observations_next_mb, rewards_mb, terminals_mb, \
    #          probas_old_mb in minibatches:
    #             # actor update
    #             values_mb = tf.squeeze(self.critic_network(observations_mb))
    #             values_next_mb = tf.squeeze(self.critic_network(observations_next_mb)) * (1-terminals_mb)
    #             deltas_mb = rewards_mb + self.gamma * values_next_mb - values_mb # bellman errors
    #             advantages_mb = generalized_advantage_estimate(deltas_mb, self.gamma, self.lambda_gae, terminals_mb)
    #             grad_actor = self._compute_actor_gradient(observations_mb, actions_mb, probas_old_mb, advantages_mb)
    #             self.optimizer_actor.apply_gradients(zip(grad_actor, self.actor_network.trainable_variables))
        


    # def _train_step(self, step_idx):
    #     '''SB3 inspired'''
    #     # unpack from memory
    #     observations, actions, observations_next, rewards, terminals, _ = self.memory.get_all()

    #     # using the old policy
    #     probas_old = self.policy.proba(policy_input=self.actor_network(observations),
    #                                    actions=actions)        
        
    #     # create minibatches
    #     as_ds = tuple(tf.data.Dataset.from_tensor_slices(x)
    #                   for x in [observations, actions, observations_next, \
    #                             rewards, terminals, probas_old])
    #     zipped = tf.data.Dataset.zip(as_ds)
    #     minibatches = zipped.batch(self.minibatch_size, drop_remainder=True) #todo
        
    #     for _ in range(self.epochs_actorupdate):
    #         for observations_mb, actions_mb, observations_next_mb, rewards_mb, terminals_mb, \
    #          probas_old_mb in minibatches:
    #             # actor update
    #             values_mb = tf.squeeze(self.critic_network(observations_mb))
    #             values_next_mb = tf.squeeze(self.critic_network(observations_next_mb)) * (1-terminals_mb)
    #             deltas_mb = rewards_mb + self.gamma * values_next_mb - values_mb # bellman errors
    #             advantages_mb = generalized_advantage_estimate(deltas_mb, self.gamma, self.lambda_gae, terminals_mb)
    #             grad_actor = self._compute_actor_gradient(observations_mb, actions_mb, probas_old_mb, advantages_mb)
    #             self.optimizer_actor.apply_gradients(zip(grad_actor, self.actor_network.trainable_variables))

    #             # critic update
    #             grad_critic = self._compute_critic_gradient(observations_mb, observations_next_mb, rewards_mb, terminals_mb)
    #             self.optimizer_critic.apply_gradients(zip(grad_critic, self.critic_network.trainable_variables))

    # def _train_step(self, step_idx):
    #     '''Fit'''
    #     # unpack from memory
    #     observations, actions, observations_next, rewards, terminals, _ = self.memory.get_all()
       
    #     # critic update
    #     for _ in range(160): # fit
    #         grad_critic = self._compute_critic_gradient(observations, observations_next, rewards, terminals)
    #         self.optimizer_critic.apply_gradients(zip(grad_critic, self.critic_network.trainable_variables))

    #     # using the old policy
    #     values = tf.squeeze(self.critic_network(observations))
    #     values_next = tf.squeeze(self.critic_network(observations_next)) * (1-terminals)
    #     deltas = rewards + self.gamma * values_next - values # bellman errors
    #     advantages = generalized_advantage_estimate(deltas, self.gamma, self.lambda_gae, terminals)
    #     probas_old = self.policy.proba(policy_input=self.actor_network(observations),
    #                                    actions=actions)        
        
    #     # create minibatches
    #     as_ds = tuple(tf.data.Dataset.from_tensor_slices(x)
    #                   for x in [observations, actions, observations_next, \
    #                             rewards, terminals, advantages, probas_old])
    #     zipped = tf.data.Dataset.zip(as_ds)
    #     minibatches = zipped.shuffle(self.n_steps).batch(self.minibatch_size, drop_remainder=False) #todo
        
    #     # actor update
    #     for _ in range(self.epochs_actorupdate):
    #         for observations_mb, actions_mb, observations_next_mb, rewards_mb, terminals_mb, \
    #          advantages_mb, probas_old_mb in minibatches:
    #             grad_actor = self._compute_actor_gradient(observations_mb, actions_mb, probas_old_mb, advantages_mb)
    #             self.optimizer_actor.apply_gradients(zip(grad_actor, self.actor_network.trainable_variables))



        # # critic update
        # for observations_mb, actions_mb, observations_next_mb, rewards_mb, terminals_mb, \
        # advantages_mb, probas_old_mb in minibatches:
            
        #     with tf.GradientTape() as tape_critic:
        #         values = tf.squeeze(self.critic_network(observations_mb))
        #         values_next = tf.squeeze(self.critic_network(observations_next_mb)) * (1-terminals_mb)
        #         values_next = tf.stop_gradient(values_next)
        #         deltas = rewards_mb + self.gamma * values_next - values # bellman errors
        #         critic_loss = tf.reduce_mean(tf.square(deltas), axis=-1) # MSE
        #     grad_critic = tape_critic.gradient(target=critic_loss, sources=self.critic_network.trainable_variables)
        #     self.optimizer_critic.apply_gradients(zip(grad_critic, self.critic_network.trainable_variables))


    # def _train_step(self): # todo: i am not sure if the critic update is included in the loop and is applied K times (K=n_epochs_actororupdate)
    #     # unpack from memory
    #     observations, actions, observations_next, rewards, terminals, _ = self.memory.to_tensor()
        
    #     # compute based on old critic and old policy
    #     values_old = tf.squeeze(self.critic(observations))
    #     values_next_old = tf.squeeze(self.critic(observations_next)) * (1-terminals)
    #     deltas_old = rewards + self.gamma * values_next_old - values_old
    #     advantages_old = generalized_advantage_estimate(deltas_old, self.gamma, self.lambda_gae)
    #     probas_old = self.policy.proba(policy_input=self.actor_network(observations),
    #                                    actions=actions)                   
        
    #     minibatches = [tf.data.Dataset.from_tensor_slices(x).batch(self.minibatch_size)
    #                     for x in [observations, actions, rewards, observations_next, terminals,
    #                               advantages_old, probas_old]
    #                   ] 
        
    #     for _ in range(self.epochs_actorupdate):            
    #         for observations_mb, actions_mb, rewards_mb, observations_next_mb, terminals_mb, \
    #             advantages_old_mb, probas_old_mb in zip(*minibatches):
    #             # critic update
    #             with tf.GradientTape() as tape_critic:
    #                 values_mb = tf.squeeze(self.critic_network(observations_mb))
    #                 values_next_mb = tf.squeeze(self.critic_network(observations_next_mb)) * (1-terminals_mb)
    #                 deltas = rewards_mb + self.gamma * values_next_mb - values_mb # bellman errors
    #                 critic_loss = tf.math.reduce_mean(tf.math.square(deltas))
    #                 critic_loss = tf.reduce_mean(tf.square(deltas), axis=-1) # MSE
    #             grad_critic = tape_critic.gradient(target=critic_loss, sources=self.critic_network.trainable_variables)
    #             self.optimizer_critic.apply_gradients(zip(grad_critic, self.critic_network.trainable_variables))

    #             # actor update 
    #             with tf.GradientTape() as tape_actor:
    #                 probas_new_mb = self.policy.proba(policy_input=self.actor_network(observations_mb),
    #                                                   actions=actions_mb) # using current (updated) policy
    #                 r = probas_new_mb / probas_old_mb
    #                 clipped = tf.clip_by_value(r, 1-self.epsilon_clip, 1+self.epsilon_clip)
    #                 loss = tf.reduce_mean(tf.reduce_min([r, clipped], axis=0)*advantages_old_mb)   
    #             grad_actor = tape_actor.gradient(loss, self.actor_network.trainable_variables)
    #             self.optimizer_actor.apply_gradients(zip(grad_actor, self.actor_network.trainable_variables))



# class PpoKeras(ActorCritic):
#     def __init__(self, environment, callbacks=[]):
#         super().__init__(environment, callbacks)

#     def _critic_loss(self, td_target, values):
#         td_error = tf.keras.losses.mean_squared_error(td_target, values)
#         return td_error

#     def _actor_loss(self, probas_old_and_actions, actor_network_outputs):
#         probas_old, actions = probas_old_and_actions
#         probas_new = self.policy.proba(observation, action) # using current (updated) policy
#         ratio = probas_new / probas_old
#         clipped = tf.clip_by_value(ratio, 1-self.epsilon_clip, 1+self.epsilon_clip)
#         loss = tf.reduce_mean(min(ratio, clipped))
#         return loss
    
#     def _train_step(self):
#         observations, _, observations_next, rewards, terminals, _ = self.memory.to_tensor()

#         values = tf.squeeze(self.critic_network(observations))
#         values_next = tf.squeeze(self.critic_network(observations_next)) * (1-terminals)
#         td_targets = rewards + self.gamma * values
#         td_errors = td_targets - values_next
#         advantages = generalized_advantage_estimate(td_errors, self.lambda_gae, self.gamma)

#         self.critic_network.train_on_batch(x=observations_next, y=td_targets)
        
#         for _ in range(self.epochs_actorupdate):
#             for minibatch in tf.data.batch():
#                 self.actor_network.train_on_batch(x=observations,
#                                                   y=tf.ones(shape=observations.shape[0]),
#                                                   sample_weight=advantages)
import tensorflow as tf
from ..agents import Agent
from ..utils.common import MdpDataset


class DQNAgent(Agent):
    '''paper: Mnih 2013 - Playing Atari with Deep Reinforcement Learning.'''
    def __init__(self, environment, policy, q_network, optimizer=tf.keras.optimizers.Adam(),
                 gamma=0.99, batch_size=32, freq_target_update=250, freq_train=4, buffer_size=50000, callbacks=[]):
        super().__init__(environment, policy, gamma, callbacks)
        
        self.online_network = q_network
        self.optimizer = optimizer
        
        self.target_network = tf.keras.models.clone_model(q_network) # sister-network (to be updated with lower frequency)
        self.freq_target_update = freq_target_update

        self.freq_train = freq_train
        self.memory = MdpDataset()
        self.buffer_size = buffer_size
        self.batch_size = batch_size
    
    @tf.function
    def act(self, state, training=False):
        policy_input = self.online_network(state)
        action = self.policy.act(policy_input, training)
        return action
    
    def _trigger_train_step(self, step_index):
        return (step_index % self.freq_train == 0)
    
    def _handle_memory(self, triggered_training):
        # make sure the memory size does not exceeds its maximum
        while len(self.memory) >= self.buffer_size:
            self.memory.pop(0)

    @tf.function
    def _compute_gradient(self, observations, actions, observations_next, rewards, terminals):
        q_next = self.target_network(observations_next)
        target = rewards + self.gamma*tf.reduce_max(q_next, axis=1)*(1-terminals)
        
        with tf.GradientTape() as tape:
            q = self.online_network(observations)
            q = tf.gather(q, indices=tf.cast(actions, dtype=tf.int64), axis=-1, batch_dims=1)
            # loss = tf.reduce_mean((target-q)**2)
            loss = tf.keras.losses.Huber()(y_true=target, y_pred=q) # aka gradient clipping

        gradient = tape.gradient(target=loss, sources=self.online_network.trainable_variables)
        self.optimizer.apply_gradients(zip(gradient, self.online_network.trainable_variables))
    
    def _train_step(self, step_index):
        if len(self.memory) < self.batch_size:
            return
        
        if step_index % self.freq_target_update == 0:
            self.target_network.set_weights(self.online_network.get_weights())

        observations, actions, observations_next, rewards, terminals, _ = self.memory.sample(self.batch_size)
        self._compute_gradient(observations, actions, observations_next, rewards, terminals) 

    def save_weights(self, *args, **kwargs):
        self.online_network.save_weights(*args, **kwargs)
        # note: we explicitly do no not save the weights of the target-network which
        #       would be the tf.keras-default





# class DQNAgent(Agent):
#     '''paper: Mnih 2013 - Playing Atari with Deep Reinforcement Learning.'''
#     def __init__(self, environment, policy, q_network, optimizer=tf.keras.optimizers.Adam(),
#                  gamma=0.99, batch_size=32, freq_target_update=250, buffer_size=50000, callbacks=[]):
#         super().__init__(environment, policy, gamma, callbacks)
        
#         self.online_network = q_network
#         self.optimizer = optimizer
#         self.target_network = tf.keras.models.clone_model(q_network) # sister-network (to be updated with lower frequency)
#         self.freq_target_update = freq_target_update

#         self.memory = MdpDataset()
#         self.buffer_size = buffer_size
#         self.batch_size = batch_size
    
#     @tf.function
#     def act(self, state, training=False):
#         q_values = self.online_network(state)
#         action = self.policy.act(q_values, training)
#         return action
    
#     def _trigger_train_step(self, step_index):
#         return True # learn at every time step
    
#     def _handle_memory(self, triggered_training):
#         # make sure the memory size does not exceeds its maximum
#         while len(self.memory) >= self.buffer_size:
#             self.memory.pop(0)

#     @tf.function
#     def _compute_gradient(self, observations, actions, observations_next, rewards, terminals):
#         max_q = tf.reduce_max(self.target_network(observations_next), axis=-1)
#         target = rewards + self.gamma * (1 - terminals) * max_q
#         with tf.GradientTape() as tape:
#             q_online = tf.gather(self.online_network(observations),
#                                  indices=tf.cast(actions, dtype=tf.int32),
#                                  axis=-1, batch_dims=1) # get Q(s,a)
#             loss = tf.reduce_mean( (target - q_online) ** 2 )
#         grad = tape.gradient(loss, self.online_network.trainable_variables)
#         self.optimizer.apply_gradients(zip(grad, self.online_network.trainable_variables))
    
#     def _train_step(self, step_index):
#         if step_index < self.batch_size:
#             return # no training, unless a whole batch can be taken from the memory-buffer (in order to avoid retracing the tensorflow-graph)
        
#         observations, actions, observations_next, rewards, terminals, _ = self.memory.sample(self.batch_size)

#         self._compute_gradient(observations, actions, observations_next, rewards, terminals)
        
#         # copy the weights of the the online network into the target network
#         if step_index % self.freq_target_update == 0:
#             self.target_network.set_weights(self.online_network.get_weights())

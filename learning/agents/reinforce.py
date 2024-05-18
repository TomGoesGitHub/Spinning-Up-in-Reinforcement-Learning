import tensorflow as tf
import numpy as np

from .agent import Agent
from ..utils.common import calculate_returns, MdpDataset

class RichSuttonReinforceAgent(Agent):
    '''
    To be more precice this is an implementation of the GPOMDP-agorithm with baseline,
    which is an improved version (less variance in gradient estimation) of classical REINFORCE.
    '''
    def __init__(self, environment, policy, policy_network, train_freq, gamma=0.99, baseline=True, callbacks=[]):
        super().__init__(environment, policy, gamma, callbacks)
        self.policy_network = policy_network
        self.train_freq = train_freq
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=3e-4)
        self.baseline = baseline
        self.memory = MdpDataset()

    def act(self, observation, training=False):
        policy_input = self.policy_network(observation)
        action = self.policy.act(policy_input, training)
        return action
    
    def _trigger_train_step(self, step_index):
        return step_index % self.train_freq == 0
    
    def _handle_memory(self, triggered_training):
        if triggered_training:
            self.memory.clear() # on-policy (clear memory after each update)

    @tf.function
    def update_network(self, observations, actions, rewards, terminals, truncations):
        # manage trjactory endings
        dones = tf.clip_by_value(terminals + truncations, 0, 1) 
        previous_dones = tf.concat([[1], dones[:-1]], axis=0)
        n_trajectories = tf.reduce_sum(terminals)
        
        discounts = tf.scan(lambda a,x: 1.*x + self.gamma*a*(1-x), elems=previous_dones)

        is_complete = tf.scan(lambda a,x: tf.case(pred_fn_pairs=[(x[0]==1, lambda: True),   # if terminal
                                                                 (x[1]==1, lambda: False)], # elif truncated
                                                  default = lambda: a),                     # else
                              elems=(terminals, truncations),
                              initializer=False,
                              reverse=True)

        returns_to_go = tf.scan(lambda a,x: x[0] + self.gamma*a*(1-x[1]),
                                elems=(rewards, dones),
                                initializer=0.,
                                reverse=True)
        
        # ignore non-terminated trajectories (by setting values to zero)
        returns_to_go = tf.where(is_complete, returns_to_go, 0)
        # note: this step is required because we use a vector-environment-framework (where
        #       trajectories potentially have different lengths) but still want to use
        #       the tf.function speed-up, which requires a a fixed tensor-shape (variable
        #       trajectory-lengths would cause retracing.)

        with tf.GradientTape() as tape:
            # action probabilities
            policy_inputs = self.policy_network(observations)
            probas = self.policy.proba(policy_inputs, actions)
            log_probas = tf.math.log(probas)
            log_probas = tf.where(is_complete, log_probas, 0)
            
            objective = 1/n_trajectories * tf.reduce_sum(discounts * returns_to_go * log_probas)
            loss = (-1) * objective
        
        gradients = tape.gradient(target=loss, sources=self.policy_network.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.policy_network.trainable_variables))

    def _train_step(self, step_index):
        observations, actions, _, rewards, terminals, truncations = self.memory.get_all()        
        self.update_network(observations, actions, rewards, terminals, truncations)

class JanPetersReinforceAgent(RichSuttonReinforceAgent): # todo: inheritance
    @tf.function
    def compute_rollout_data(self, observations, actions, rewards, terminals, truncations):
        '''
        This method precomputes important terms, which are required for the gradient computation.
        Therefore, more expressive quantities are calculated from the standard MDP-interaction-data.
        
        Note: this method is required because we use a vector-environment-framework (where
              trajectories potentially have different lengths) but still want to use
              the tf.function speed-up, which requires a a fixed tensor-shape (variable
              trajectory-lengths would cause retracing.)
        '''
        # manage trjactory starts and ends
        dones = tf.clip_by_value(terminals + truncations, 0, 1) 
        rollout_starts_by_done = tf.concat([[1], dones[:-1]], axis=0)

        # returns
        returns_to_go = tf.scan(lambda a,x: x[0] + self.gamma*a*(1-x[1]),
                                elems=(rewards, dones),
                                initializer=0.,
                                reverse=True)
        
        rollout_returns = tf.scan(lambda a,x: x[0]*(1-x[1]) + a*x[1],
                                  elems=(returns_to_go, rollout_starts_by_done),
                                  initializer=0.)

        # action probabilities
        policy_inputs = self.policy_network(observations)
        probas = self.policy.proba(policy_inputs, actions)
        log_probas = tf.math.log(probas)
        cumsum_log_probas = tf.scan(lambda a,x: x[0] + a * (1-x[1]),
                                    elems=(log_probas, rollout_starts_by_done),
                                    initializer=0.)
        
        return rollout_returns, cumsum_log_probas
    
    @tf.function
    def compute_single_logp_gradient(self, persistent_tape, target):
        return persistent_tape.gradient(target, sources=self.policy_network.trainable_variables)

    def _train_step(self, step_index):
        observations, actions, _, rewards, terminals, truncations = self.memory.get_all()

        # log probability gradients
        with tf.GradientTape(persistent=True) as tape:
            rollout_returns, cumsum_log_probas = self.compute_rollout_data(observations, actions, rewards,
                                                                         terminals, truncations)
        
            # rollout_returns = [x for x in tf.boolean_mask(rollout_returns, terminals)]
            # rollout_log_probas = [x for x in tf.boolean_mask(cumsum_log_probas, terminals)]
            rollout_returns = list(tf.boolean_mask(rollout_returns, terminals))
            rollout_log_probas = list(tf.boolean_mask(cumsum_log_probas, terminals))

        rollout_logp_gradients = [self.compute_single_logp_gradient(x) for x in rollout_log_probas]
        # rollout_logp_gradients = [tape.gradient(x, sources=self.policy_network.trainable_variables) for x in rollout_log_probas]
        # note: the computed gradients are a list of lists of tensors (a nested structre)

        # baseline (operating on nested structures)
        squared = [tf.nest.map_structure(lambda x: x**2, structure=g) for g in rollout_logp_gradients]
        weighted = [tf.nest.map_structure(lambda x: w*x**2, structure=g)
                    for w,g in zip(rollout_returns, rollout_logp_gradients)]
        numerator = [tf.reduce_sum(x, axis=0) for x in zip(*weighted)]
        denominator = [tf.reduce_sum(x, axis=0) for x in zip(*squared)]
        baseline = [n/d for n,d in zip(numerator, denominator)]

        # objective gradient (operating on nested structures)
        obj_gradient_terms = [g * (r - baseline) for g,r in zip(rollout_logp_gradients, rollout_returns)]
        obj_gradient = [tf.reduce_sum(x) for x in zip(*obj_gradient_terms)]
        loss_gradient = [(-1)*x for x in obj_gradient]
        self.optimizer.apply_gradients(zip(loss_gradient, self.policy_network.trainable_variables))
        

        

# class ReinforceAgent(Agent):
#     '''
#     To be more precice this is an implementation of the GPOMDP-agorithm with baseline,
#     which is an improved version (less variance in gradient estimation) of classical REINFORCE.
#     '''
#     def __init__(self, environment, policy, policy_network, train_freq, gamma=0.99, baseline=True, callbacks=[]):
#         super().__init__(environment, policy, gamma, callbacks)
#         self.policy_network = policy_network
#         self.train_freq = train_freq
#         self.optimizer = tf.keras.optimizers.Adam(learning_rate=3e-4)
#         self.baseline = baseline
#         self.memory = MdpDataset()

#     def act(self, observation, training=False):
#         policy_input = self.policy_network(observation)
#         action = self.policy.act(policy_input, training)
#         return action
    
#     def _trigger_train_step(self, step_index):
#         return step_index % self.train_freq == 0
    
#     def _handle_memory(self, triggered_training):
#         if triggered_training:
#             self.memory.clear() # on-policy (clear memory after each update)
    
#     def calculate_rollout_log_probability(self, log_probas, dones):
#         rollout_log_probas = tf.scan(lambda a,x: x[0] + a * (1-x[1]),
#                                     elems=(log_probas, dones),
#                                     initializer=0.,
#                                     reverse=True)
#         return rollout_log_probas

#     @tf.function
#     def update_network(self, observations, actions, rewards, dones):
#         # returns-to-go
#         previous_dones = tf.concat([[1], dones[:-1]], axis=0)
#         returns_to_go = tf.scan(lambda a,x: x[0] + self.gamma*a*(1-x[1]),
#                                 elems=(rewards, dones),
#                                 initializer=0.,
#                                 reverse=True)
        
#         discounts = tf.scan(lambda a,x: 1.*x + self.gamma*a*(1-x), elems=previous_dones)

#         with tf.GradientTape(persistent=True) as tape:
#             # action probabilities
#             policy_inputs = self.policy_network(observations)
#             probas = self.policy.proba(policy_inputs, actions)
#             log_probas = tf.math.log(probas)
            
#             # # baseline
#             # if self.baseline:
#             #     numerator = log_probas**2 * discounts * returns_to_go
#             #     denominator = log_probas**2 
#             #     baseline = numerator / denominator
#             # else:
#             #     baseline = 0
#             # baseline = tf.stop_gradient(baseline)
            
#             objective = tf.reduce_mean(discounts * returns_to_go * log_probas)
#             loss = (-1) * objective
        
#         gradients = tape.gradient(target=loss, sources=self.policy_network.trainable_variables)
#         self.optimizer.apply_gradients(zip(gradients, self.policy_network.trainable_variables))

    
#     def _train_step(self, step_index):
#         observations, actions, _, rewards, terminals, truncations = self.memory.get_all()       
#         dones = tf.clip_by_value(terminals + truncations, 0, 1) 
        
#         # drop non-terminated trajectories
#         is_complete = tf.scan(lambda a,x: 1. if x[0] else (0. if x[1] else a),
#                               elems=(terminals, truncations),
#                               initializer=0.,
#                               reverse=True)
#         observations, actions, rewards, dones = [tf.boolean_mask(x, is_complete)
#                                                  for x in (observations, actions, rewards, dones)]
        
#         self.update_network(observations, actions, rewards, dones)      

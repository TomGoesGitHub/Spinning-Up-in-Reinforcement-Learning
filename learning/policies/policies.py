import tensorflow as tf
import tensorflow_probability as tfp
import gymnasium as gym
import numpy as np
import scipy

class Policy:
    def __init__(self):
        self.network = None

    def initialize_policy(self, network):
        self.network = network

    def act(self, policy_input, training=False):
        raise NotImplementedError
    
    def probability(policy_input, action):
        raise NotImplementedError

    def reset(self):
        pass # potentially reset schedules 

class BoltzmannPolicy(Policy):
    def __init__(self, temperature_schedule, act_deterministic_if_not_training=False):       
        self.temperature_schedule = temperature_schedule
        self.act_deterministic_if_not_training = act_deterministic_if_not_training
    
    def distribution(self, policy_input):
        logits = policy_input
        dist = tfp.distributions.Categorical(logits)
        return dist
    
    def act(self, policy_input, training=False):
        logits = policy_input
        temperature = self.temperature_schedule(step=training)
        logits_temperated = logits / temperature
        if not training and self.act_deterministic_if_not_training:
            # act greedy (deterministic)
            action = tf.argmax(logits_temperated, axis=-1, output_type=tf.int32) 
        else:
            # sample from histogram
            distribution = tfp.distributions.Categorical(logits_temperated)
            action = distribution.sample()
        return action

    def proba(self, policy_input, actions=None):
        logits = policy_input
        temperature = self.temperature_schedule()
        logits_temperated = logits / temperature
        probas = tf.keras.activations.softmax(logits_temperated)
        if actions is None:
            return probas
        else:
            idx = tf.dtypes.cast(actions, tf.int32)
            return tf.gather(probas, indices=idx, axis=-1, batch_dims=1)

    def reset(self):
        self.temperature_schedule.reset()

class EpsilonGreedyPolicy(Policy):
    def __init__(self, epsilon_schedule):
        self.epsilon_schedule = epsilon_schedule

    def act(self, policy_input, training=False):
        probas = policy_input
        epsilon = self.epsilon_schedule(step=True)
    
        if not training or (tf.random.uniform(shape=[], minval=0, maxval=1) >= epsilon):
            # greedy action
            action = tf.argmax(probas, axis=-1, output_type=tf.int32) 
        else:
            # random action
            action = tf.random.uniform(shape=[len(probas)], minval=0,
                                       maxval=probas.shape[-1], dtype=tf.int32) 
        return action
            
    def proba(self, policy_input, actions):
        #* 1
        # actions = tf.dtypes.cast(actions, tf.int32)
        # categorical_probas = policy_input
        # return tf.gather(categorical_probas, indices=actions, batch_dims=1, axis=-1)
        
        #* 2
        categorical_probas = policy_input
        n_actions = categorical_probas.shape[-1]
        
        greedy_actions = tf.math.argmax(categorical_probas, axis=-1)
        greedy_actions = tf.dtypes.cast(greedy_actions, tf.float32)
        is_max = tf.math.equal(greedy_actions, actions)
        is_max = tf.dtypes.cast(is_max, tf.float32)

        epsilon = self.epsilon_schedule()
        p_greedy = (1-epsilon) + (epsilon / n_actions) # probabilty for greedy action
        p_random = epsilon / n_actions # probablility for random action
        probas = is_max * p_greedy + (1-is_max)* p_random
        return tf.dtypes.cast(probas, tf.float32)
    
    def reset(self):
        self.epsilon_schedule.reset()


class GaussianPolicy(Policy):
    def __init__(self, environment, scale=None, act_deterministic_if_not_training=False):
        action_space = environment.unwrapped.single_action_space
        action_space_flat = tf.squeeze(action_space.shape)
        self.bounds = (action_space.low, action_space.high)
        
        self._scale_is_learnable = (scale is None)
        if self._scale_is_learnable:
            self.expected_input_shape = [None, 2*action_space_flat]
        else:
            self.scale = tf.ensure_shape(scale, shape=action_space_flat)
            self.expected_input_shape = [None, action_space_flat]
        
        self.act_deterministic_if_not_training = act_deterministic_if_not_training
    
    def distribution(self, policy_input):
        tf.ensure_shape(policy_input, self.expected_input_shape)
        if self._scale_is_learnable:
            loc, scale_diag = tf.split(policy_input, num_or_size_splits=2, axis=-1)
        else:
            loc = policy_input
            scale_diag = self.scale
        dist = tfp.distributions.MultivariateNormalDiag(loc, scale_diag)
        return dist

    def act(self, policy_input, training=False):
        dist = self.distribution(policy_input)
        if not training and self.act_deterministic_if_not_training:
            action = dist.loc # choose mean action (deterministically)
        else:
            action = dist.sample() # sample from gaussian
        
        # clipping
        clip_value_min, clip_value_max = self.bounds
        action = tf.clip_by_value(action, clip_value_min, clip_value_max)
        # note: When policy gradient methods are applied, out-of-bound actions need to be clipped
        #       before execution, while policies are usually optimized as if the actions are not
        #       clipped.
        #       (compare: Fujita - Clipped Action Policy Gradient, https://arxiv.org/pdf/1802.07564.pdf)
        
        return action
    
    def proba(self, policy_input, actions):
        dist = self.distribution(policy_input)
        proba = dist.prob(actions)
        return proba

class BetaDistributionPolicy(Policy):
    def __init__(self, environment, act_deterministic_if_not_training=False):
        self.environment = environment
        self._action_space_flat = tf.squeeze(self.environment.action_space.shape)
        self.expected_input_shape = [None, 2*self._action_space_flat]
        self.act_deterministic_if_not_training = act_deterministic_if_not_training
        
    def distribution(self, policy_input):
        tf.ensure_shape(policy_input, self.expected_input_shape)
        alpha, beta = tf.split(policy_input + 10e-6, num_or_size_splits=2, axis=-1)
        independent_betas = tfp.distributions.Beta(alpha, beta)
        dist = tfp.distributions.Independent(independent_betas, reinterpreted_batch_ndims=1)
        #dist = tfp.distributions.Beta(alpha, beta)
        return dist

    def act(self, policy_input, training=False):
        dist = self.distribution(policy_input)
        if not training and self.act_deterministic_if_not_training: 
            unscaled_action = dist.mean() # choose mean action (deterministically)
        else:
            unscaled_action = dist.sample() # sample from gaussian
        
        # note: the beta distribution is defined on the interval [0,1] and therefore
        #       needs to be scaled
        lower_bound = tf.constant(self.environment.action_space.low)
        upper_bound = tf.constant(self.environment.action_space.high)
        action = (upper_bound - lower_bound) * unscaled_action + lower_bound
        action = tf.clip_by_value(action, lower_bound+1e-6, upper_bound-1e-6)
        tf.debugging.check_numerics(action, message='')
        return action
    
    def proba(self, policy_input, actions):
        # rescale action into the support of the beta distribution
        lower_bound = tf.constant(self.environment.action_space.low)
        upper_bound = tf.constant(self.environment.action_space.high)
        rescaled_actions = (actions - lower_bound) / (upper_bound - lower_bound)

        # evaluate distribution
        dist = self.distribution(policy_input)
        proba = dist.prob(rescaled_actions)
        return proba

# class BoltzmannPolicy(Policy):
#     def __init__(self, temperature_schedule):       
#         self.temperature_schedule = temperature_schedule
    
#     def act(self, policy_input, training=False):
#         logits = policy_input
#         temperature = self.temperature_schedule()
#         logits_temperated = logits / temperature
#         # if training: 
#         #     distribution = tfp.distributions.Categorical(logits_temperated)
#         #     action = distribution.sample() 
#         # else:
#         #     action = tf.math.argmax(logits_temperated, axis=-1)
#         distribution = tfp.distributions.Categorical(logits_temperated)
#         action = distribution.sample() 
#         return tf.squeeze(action)

#     def proba(self, policy_input, actions):
#         logits = policy_input
#         temperature = self.temperature_schedule()
#         logits_temperated = logits / temperature
#         distribution = tfp.distributions.Categorical(logits_temperated)
#         probas = distribution.prob(actions)
#         return probas
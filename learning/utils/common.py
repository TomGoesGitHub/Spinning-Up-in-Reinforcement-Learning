from collections import namedtuple
import random

import tensorflow as tf
import numpy as np
import gymnasium as gym


class MdpDataset:
    def __init__(self):
        self.clear()
        self._n_elems_per_datapoint = 6 # observations, actions, observations_next, rewards, terminals, truncations
    
    def __len__(self):
        return len(self._memory)

    def remember(self, datapoint):
        # datapoint = <observations, actions, observations_next,
        #               rewards, terminals, truncations>
        self._memory.append(datapoint)
    
    def clear(self):
        self._memory = []
    
    def pop(self, idx):
        self._memory.pop(idx)
    
    def _find_elems_by_idx(self, idxes):
        # data-structure: observations, actions, observations_next, rewards, terminals, truncations
        data = [], [], [], [], [], [] 
        
        for i in idxes:
            datapoint = self._memory[i]
            for data_list, datapoint_value in zip(data, datapoint):
                data_list.append(datapoint_value)
        
        return [np.array(data_list, dtype=np.float32) for data_list in data]

    def sample(self, batch_size):
        # sample time index
        idxes = [random.randint(0, len(self) - 1) for _ in range(batch_size)]
        data = self._find_elems_by_idx(idxes)

        # sample from the first environment from vector environments
        data = [d[:,0, ...] for d in data]
        return data
    
    def get_all(self):
        data = self._memory
        # shape: t * n_elems_per_datapoint * array[n_env * single_shape]

        # at the final time step the data-gathering process was interupted, therefore manually add a truncation
        data[-1][5][:, ...] = True

        data = [np.array(data_list, dtype=np.float32) for data_list in zip(*data)]
        # shape: n_elems_per_datapoint * t * array[n_env * single_shape]

        # change tensor-axes s.t. 1st dim: environment, 2nd dim: timestamp
        data = [tf.einsum('ij...->ji...', tf.constant(x)) for x in data]
        # shape: n_elems_per_datapoint * tensor[n_env* t * single_shape]
        
        # combine env-dim with timestamps-dim
        data = [tf.reshape(x, shape=[-1, *x.shape[2:]]) for x in data]
        # shape: n_elems_per_datapoint * tensor[(n_env*t)*single_shape]

        return data



def create_mlp(input_shape, output_shape, network_architecture=[64,64],
               activation='tanh', output_activation='linear'):
    # input
    inputs = x = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Flatten()(x)
    
    # hidden
    for units in network_architecture:
        x = tf.keras.layers.Dense(units,activation)(x)
    
    # output
    output_shape_flat = tf.reduce_prod(output_shape)
    if output_shape_flat != output_shape:
        x = tf.keras.layers.Dense(units=output_shape_flat, activation=output_activation)(x)
        outputs = tf.keras.layers.Reshape(target_shape=output_shape)(x)
    else: 
        outputs = tf.keras.layers.Dense(units=output_shape, activation=output_activation)(x)

    multilayer_perceptron = tf.keras.Model(inputs, outputs)
    multilayer_perceptron.summary()
    return multilayer_perceptron

def create_cnn(input_shape, output_shape, network_architecture,
               activation='tanh', output_activation='linear'):
    # input
    inputs = x = tf.keras.layers.Input(shape=input_shape)
    
    # hidden (convolutional)
    for filters in network_architecture:
        x = tf.keras.layers.Conv2D(filters, kernel_size=2, strides=2, padding='same')(x)
    
    # output
    x = tf.keras.layers.Flatten()(x)
    outputs = tf.keras.layers.Dense(units=output_shape, activation=output_activation)(x)
    
    cnn = tf.keras.Model(inputs, outputs)
    cnn.summary()
    return cnn

def get_atari_dqn_cnn(input_shape, output_shape):
    '''Architecture from Playing Mnih-Atari with Deep Reinforcement Learning'''

    initializer = tf.keras.initializers.VarianceScaling(scale=2) # as suggested here: https://www.tensorflow.org/api_docs/python/tf/keras/initializers/VarianceScaling

    inputs = x = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(filters=16, kernel_size=8, strides=4, activation='relu', kernel_initializer=initializer)(x)
    x = tf.keras.layers.Conv2D(filters=32, kernel_size=4, strides=2, activation='relu', kernel_initializer=initializer)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(units=256, activation='relu', kernel_initializer=initializer)(x)
    outputs = tf.keras.layers.Dense(units=output_shape, activation='linear', kernel_initializer=initializer)(x)
    
    cnn = tf.keras.Model(inputs, outputs, name='q_network')
    cnn.summary()
    return cnn
    
def reduce_gradients(gradients, fn=(lambda x: tf.reduce_sum(x, axis=0)), weights=None):
    '''
    In the tensorflow framework, gradients are nested structures and no gradients.
    This function allows operations on those nested structures.
    '''
    # todo: I think this is what tf.nest.map_structure does
    if weights is None:
        weights = tf.ones(shape=[len(gradients)])
    assert len(weights) == len(gradients)
    
    n_trainable_variables = len(gradients[0])
    reduced_gradient = [] # a nested structure of tensors
    for i in range(n_trainable_variables):
        to_be_reduced = [w*g[i] for g,w in zip(gradients, weights)]
        reduced = fn(to_be_reduced) # a tensor
        reduced_gradient.append(reduced)
    return reduced_gradient

@tf.function
def generalized_advantage_estimate(bellman_error, gamma, lamda, terminals=None):
    '''
    paper: https://arxiv.org/abs/1506.02438
    blog: https://towardsdatascience.com/generalized-advantage-estimate-maths-and-code-b5d5bd3ce737
    implementation: https://pylessons.com/LunarLander-v2-PPO (gives same results)'''
    if terminals is None:
        terminals = tf.zeros_like(bellman_error) # expecting a whole trajectory 
    advantages = tf.scan(lambda a, x: lamda * gamma * (1-x[1]) * a + x[0] ,
                        elems=[bellman_error, terminals],
                        initializer=tf.constant(0.0, dtype=tf.float32),
                        reverse=True)
    return advantages

def create_minibatches(datasets, batch_size):
    if type(datasets) is not list: datasets=list[datasets]
    batched_datasets = [tf.data.Dataset(datasets).batch(batch_size)]
    return batched_datasets


# def rollout(env, agent,  n_steps=np.inf):
#     Dataset = namedtuple(typename='Dataset',
#                          field_names=['observations', 'actions', 'next_observations',
#                                      'rewards', 'terminateds', 'truncateds', 'infos', 'dones'],
#                          defaults=[[],[],[],[],[],[],[],[]])
#     dataset = Dataset()

#     env.reset()
#     steps, done = 0, False
#     observation, _ = env.reset()
#     while not done and steps < n_steps:
#         action_flat = agent.act(tf.constant([observation]))

#         action =  gym.spaces.utils.unflatten(env.action_space, action_flat) #action_flat.numpy() # 
#         if env.render_mode is not None:
#             env.render()
#         next_observation, reward, terminated, truncated, info = env.step(action)
#         done = terminated or truncated
        
#         datapoint = (observation, action, next_observation, reward, terminated, truncated, info, done)
#         for i, item in enumerate(datapoint):
#             dataset[i].append(item)
#         steps += 1
#         observation = next_observation
#     return dataset


def rollout(env, agent,  n_steps=np.inf):
    # todo: assert that env is indeed a single environment (and not a vector env)
    dataset = MdpDataset()

    env.reset()
    steps, done = 0, False
    observation, _ = env.reset()
    while not done and steps < n_steps:
        action_flat = agent.act(tf.constant([observation]))
        action = action_flat.numpy().reshape([-1, *env.action_space.shape]) # unflatten
        action = action[0] # unbatch
        #action =  gym.spaces.utils.unflatten(env.action_space, action_flat) # todo: clean up
        if env.render_mode is not None:
            env.render()
        next_observation, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    
        datapoint = (np.array(observation), np.array(action), np.array(next_observation),
                     np.array(reward), np.array(terminated), np.array(truncated))
        # shape: n_elems_per_datapoint * arrray(single_shape)

        # add the VectorEnvironment-dimension manually
        datapoint = [elem[np.newaxis, ...] for elem in datapoint]
        # shape: n_elems_per_datapoint * array[n_env * single_shape]
        
        dataset.remember(datapoint)
        steps += 1
        observation = next_observation
    return dataset

def calculate_returns(rewards, dones=None, discount=1., ):
    discount = tf.convert_to_tensor(discount, dtype=tf.float32)
    if dones is None:
        dones = tf.zeros_like(rewards)
    
    returns = tf.scan(lambda a,x: x[0] + discount * a * (1-x[1]),
                      elems=(rewards, dones),
                      initializer=0.,
                      reverse=True)
    return returns
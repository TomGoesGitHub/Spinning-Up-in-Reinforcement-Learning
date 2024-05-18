import numpy as np
# from tqdm import tqdm
from .base import RubiksCube
import matplotlib.pyplot as plt

# def generate_dataset(n_rollouts):
#     cube = RubiksCube()
#     # dataset (to be filled)
#     states = np.empty(shape=[0, *cube.observation_space.shape], dtype=cube.observation_space.dtype)
#     actions = np.empty(shape=[0, *cube.action_space.shape], dtype=cube.action_space.dtype)
#     rewards = np.empty(shape=[0, 1], dtype=np.float32)
#     meta_data = np.empty(shape=[0, 2], dtype=int)
    
#     for i in tqdm(range(n_rollouts)):
#         cube._reset_to_solved()
#         n_steps = np.random.randint(low=20, high=30)
        
#         # forward: solved cube to unsolved cube
#         all_actions_forward, all_states_forward = [], []
#         for _ in range(n_steps):
#             state = cube.observe()
#             action = cube.action_space.sample() # random action
#             cube.step(action)
#             all_actions_forward.append(action)
#             all_states_forward.append(state)
        
#         # backward: unsolved cube to solved cube
#         # note: there is a shift in time index due to the reversing
#         all_states_backward = all_states_forward[-1:0:-1]
#         all_actions_backward = all_actions_forward[-2::-1]
        
#         X = np.vstack([X, all_states_backward])
#         y = np.vstack([y, all_actions_backward])
#         rewards = np.vstack([rewards, np.put(np.zeros(shape=[n_steps]), ind=-1, v=1)])
#         meta_data = np.vstack([meta_data,
#                                np.column_stack([np.full(shape=[n_steps-1], fill_value=i),
#                                                 np.arange(n_steps-1)]) ])
    
#     return states, actions, rewards, meta_data

def visualize_rollout(agent, n_steps=30):
    env = RubiksCube()
    env._reset_to_unsolved()

    fig, axes = plt.subplots(nrows=3, ncols=10, figsize=(12,6), tight_layout=True)
    for t in range(n_steps):
        ax = axes.ravel()[t]
        ax.set_title(f'{t=}')
        env.render(ax)
        state = env.observe()
        action = agent.act(state)[0][0]
    plt.show()

def visualize_rollout2(states):
    env = RubiksCube()
    fig, axes = plt.subplots(nrows=3, ncols=10, figsize=(12,6), tight_layout=True) # todo
    
    for ax in axes.ravel():
        ax.set_axis_off()
    
    for t, state in enumerate(states):
        ax = axes.ravel()[t]
        ax.set_title(f'{t=}')
        env.state = state
        env.render(ax)
    plt.show()

        
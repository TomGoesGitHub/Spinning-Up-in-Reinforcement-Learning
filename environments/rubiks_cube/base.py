import gymnasium as gym
import numpy as np
from scipy.spatial.transform import Rotation
from sklearn.preprocessing import OneHotEncoder

import itertools
import warnings

import matplotlib.pyplot as plt
from matplotlib import patches
from mpl_toolkits.mplot3d import Axes3D, art3d


class CubeElem:
    edge_len = 1
    def __init__(self, xyz, rpy, coloring):
        # state
        self.xyz = xyz
        self.rotation = Rotation.from_euler(seq='xyz', angles=rpy)

        # initial state
        self.initial_xyz = xyz
        self.initial_rotation = Rotation.from_euler(seq='xyz', angles=rpy)
        self.initial_state = self.state
        
        # visualization
        self.coloring = coloring
    
    def rotate(self, axis, angle=+90, degrees=True):
        rotvec = axis * angle
        applied_rotation = Rotation.from_rotvec(rotvec, degrees)
        self.rotation = applied_rotation * self.rotation
        self.xyz = applied_rotation.apply(self.xyz)

    def reset(self):
        self.state = self.initial_state
    
    def face_color(self, normal_vec):
        normal_vec_in_initial_config = self.rotation.inv().apply(normal_vec)
        normal_vec_in_initial_config = np.round(normal_vec_in_initial_config)
        color = self.coloring[tuple(normal_vec_in_initial_config)]
        return color
    
    @property
    def state(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            state = np.concatenate([np.round(self.xyz, decimals=6),
                                    np.mod(self.rotation.as_euler(seq='xyz'), 2*np.pi)],
                                    axis=0)
        return state

    @state.setter
    def state(self, state):
        self.xyz = state[:3]
        self.rotation = Rotation.from_euler(seq='xyz', angles=state[3:])


class CubeFace:
    def __init__(self, normal_vec, solved_color, rot_vec, offset_2d):
        self.normal_vec = np.array(normal_vec)
        self.solved_color = solved_color
        
        # 2d rendering
        self.rot_vec = np.array(rot_vec)
        self.offset_2d = np.array(offset_2d)
    
    def project_2d(self, xyz):
        rotation = Rotation.from_rotvec(self.rot_vec, degrees=True)
        return rotation.apply(xyz)[1:] - CubeElem.edge_len/2 + self.offset_2d


class RubiksCube(gym.Env):
    def __init__(self, seed=42, distortion_depth=25):
        # Geometry
        self.edge_len = CubeElem.edge_len * 3
        self.dist_to_surface = self.edge_len/2 - CubeElem.edge_len/2
        # note: due to the symmetry the surface can be described with only one distance

        self.faces = [
            CubeFace([1,0,0],  'blue',   [0,0,0],    [0,0]),                 # front
            CubeFace([-1,0,0], 'green',  [0,-180,0], [0, -2*self.edge_len]), # back
            CubeFace([0,1,0],  'orange', [0,0,-90],  [self.edge_len, 0]),    # right
            CubeFace([0,-1,0], 'red',    [0,0,90],   [-self.edge_len, 0]),   # left
            CubeFace([0,0,1],  'white',  [0,90,0],   [0, self.edge_len]),    # top
            CubeFace([0,0,-1], 'yellow', [0,-90,0],  [0, -self.edge_len]),   # bottom
        ]

        all_xyz = self._initial_positioning()
        initial_rpy = np.array([0,0,0]) # no rotation in inital configuration
        self.elems = [CubeElem(xyz, initial_rpy, coloring=self._colors_from_xyz(xyz))
                      for xyz in all_xyz]
        
        all_possible_rots = list(itertools.product(*3*[np.arange(0, 2*np.pi, np.pi/2)]))
        self.onehot_encoder_xyz = OneHotEncoder(sparse_output=False).fit(X=all_xyz)
        self.onehot_encoder_rot = OneHotEncoder(sparse_output=False).fit(X=all_possible_rots)
        
        
        # Simulation
        self.seed = seed
        self.t = 0
        self.distortion_depth = distortion_depth
        self.action_space = gym.spaces.Discrete(n=12, seed=self.seed)
        
        # self._observation_space_unflattened = gym.spaces.Dict(
        #     spaces={
        #         'time_state': gym.spaces.Box(low=0, high=np.inf, seed=self.seed),
        #         # scalar, remaining timesteps till truncation
                
        #         'cube_state': gym.spaces.MultiBinary(n=(27,3*3+3*4), seed=self.seed),
        #         # 27 elems, xyz with 3 options, rpy-eulers with 4 options 
        #     }
        # )

                
        self._observation_space_unflattened = gym.spaces.Tuple(
            spaces=(# time state (scalar, remaining timesteps till truncation)
                    gym.spaces.Box(low=0, high=np.inf, seed=self.seed),
                    
                    # cube state (27 elems, xyz with 3 options, rpy-eulers with 4 options)
                    gym.spaces.MultiBinary(n=(27,3*3+3*4), seed=self.seed),
            )
        )
        self.observation_space = gym.spaces.utils.flatten_space(self._observation_space_unflattened)
        
    def _get_truncation_timelimit(self):
        return np.ceil(1.5 * self.distortion_depth)

    def _initial_positioning(self):
        offset = CubeElem.edge_len/2
        coordinates = np.arange(0, self.edge_len, CubeElem.edge_len) + offset - self.edge_len/2
        # note: '+offset' is for center-of-mass of elems and '-edge_len/2' is for overall 0-center
        (xx, yy, zz) = np.meshgrid(coordinates, coordinates, coordinates)
        x, y, z = xx.flatten(), yy.flatten(), zz.flatten()
        all_xyz = np.stack([x,y,z], axis=1)
        return all_xyz

    def _colors_from_xyz(self, initial_xyz):        
        facecolors = {} 
        for face in self.faces:
            coordinate_of_interest = np.dot(face.normal_vec, initial_xyz)
            is_on_surface = (coordinate_of_interest == self.dist_to_surface)
            color = face.solved_color if is_on_surface else 'black'
            facecolors[tuple(face.normal_vec)] = color
        return facecolors

    @property
    def state(self):
        return self.observe()

    @state.setter
    def state(self, st):
        # unpack and unflatten
        st = gym.spaces.utils.unflatten(self._observation_space_unflattened, x=st)
        st_t, st_cube = st[0].squeeze(), st[1]

        #* time
        time_limit = self._get_truncation_timelimit()
        self.t = st_t + time_limit

        #* cube 
        # invert onehot encoding
        state_xyz_onehot, state_rot_onehot = st_cube[:, :9],  st_cube[:, 9:]
        state_xyz_continous = self.onehot_encoder_xyz.inverse_transform(state_xyz_onehot)
        state_rot_continous = self.onehot_encoder_rot.inverse_transform(state_rot_onehot)
        
        # update elements
        for s_xyz, s_rot, elem in zip(state_xyz_continous, state_rot_continous, self.elems):
            s = np.hstack([s_xyz, s_rot])
            elem.state = s

    def render(self, ax=None):
        l = self.edge_len
        if ax is None:       
            fig = plt.figure()
            ax = fig.add_subplot(xlim=[-1.6*l, +1.6*l], ylim=[-2.6*l, 1.6*l], aspect='equal')
        ax.axis('off')
        ax.set_aspect('equal')
        ax.set_xlim(-1.6*l, 1.6*l)
        ax.set_ylim(-2.6*l, 1.6*l)
        for face in self.faces:
            for elem in self.elems:
                elem_is_on_surface = self._surface_check(elem, face.normal_vec)
                if elem_is_on_surface:
                    patch_anchor = face.project_2d(elem.xyz)
                    patch = patches.Rectangle(xy=patch_anchor,
                                             width=CubeElem.edge_len,
                                             height=CubeElem.edge_len,
                                             facecolor=elem.face_color(face.normal_vec),
                                             edgecolor='black')
                    ax.add_patch(patch)
        return plt.gcf()

    def play(self):
        '''Start a minigame for human interaction with the environment.'''
        plt.close()
        rules = 'Solve the Rubiks-Cube. Press Escape-Key to quit.'
        print(rules)

        playing = True
        while playing:
            # todo: escape
            action_input = input('Please take an action.')
            action = tuple(action_input.split(' '))
            self.step(action)
            self.render()
            plt.close()
            
    def _reset_to_solved(self):
        '''Reset the cube to the solved position.'''
        for elem in self.elems:
            elem.reset()
        assert self.check_if_solved()
    
    def _reset_to_unsolved(self):
        '''Reset the cube to an (random) unsolved position. The complexity of this
        position is controlled by the distortion-depth attribute of the cube.'''
        self._reset_to_solved()
        applied_depth = int(np.random.uniform(low=1, high=self.distortion_depth+1))
        for _ in range(applied_depth):
            action = self.action_space.sample() # random action
            self.step(action)
        if self.check_if_solved():
            action = self.action_space.sample() # random action
            self.step(action)
    
    def reset(self, to_solved=False):
        if to_solved:
            self._reset_to_solved()
        else:
            self._reset_to_unsolved()
        observation = self.observe()
        info = {}
        self.t = 0
        return observation, info
        
    def _surface_check(self, elem, normal_vec):
        coordinate_of_interest = np.dot(elem.xyz, normal_vec)
        elem_is_on_surface = np.abs(coordinate_of_interest - self.dist_to_surface) < 1e-5
        return elem_is_on_surface

    def check_if_solved(self):
        for elem in self.elems:
            elem_is_correct = np.all(elem.initial_state == elem.state)
            if not elem_is_correct:
                return False
        return True

    # def observe(self):
    #     cube_state = np.concatenate([elem.state for elem in self.elems], axis=-1).flatten()
    #     time_state = np.array([self.distortion_depth - self.t])
    #     state = np.concatenate([cube_state, time_state], axis=-1)
    #     return state

    def observe(self):
        # time state
        time_state = np.array([0]) # todo    

        # cube state
        state_continous = np.vstack([elem.state for elem in self.elems])
        xyz_continous = state_continous[:, :3]
        rotvec_continous = state_continous[:, 3:]
        xyz_onehot = self.onehot_encoder_xyz.transform(xyz_continous)
        rotvec_onehot = self.onehot_encoder_rot.transform(rotvec_continous)
        cube_state = np.concatenate([xyz_onehot, rotvec_onehot], axis=1).flatten()
        
        observation = np.concatenate([time_state, cube_state], axis=-1)
        return observation

    def step(self, action):
        face_idx, rot_direction = np.divmod(action, 2)
        assert face_idx in np.arange(len(self.faces))
        assert rot_direction in [0,1]
        
        # change state
        self.t += 1
        rot_angle = 90 if rot_direction else -90
        normal_vec = self.faces[face_idx].normal_vec
        for elem in self.elems:
            if self._surface_check(elem, normal_vec):
                elem.rotate(axis=normal_vec, angle=rot_angle)
        
        terminated = self.check_if_solved() # terminate in goal-state only
        truncated = (self.t >= np.ceil(1.5 * self.distortion_depth)) # timelimit
        observation = self.observe()
        reward = 0. if terminated else -1.
        info = {}
        
        return observation, reward, terminated, truncated, info

if __name__ == '__main__':
    cube = RubiksCube()
    cube.reset()
    
    cube.render()
    plt.close() # todo: tmp

    cube.step((0,0))
    cube.step((1,1))

    cube.render()
    plt.close() # todo: tmp

    state = cube.observe()

    #* generate_dataset
    # dataset (to be filled)
    X = np.empty(shape=[0, *cube.observation_space.shape], dtype=cube.observation_space.dtype)
    y = np.empty(shape=[0, *cube.action_space.shape], dtype=cube.action_space.dtype)
    
    n_rollouts = 1000
    for _ in range(n_rollouts):
        cube.reset()
        n_steps = np.random.randint(low=20, high=30)
        
        # forward: solved cube to unsolved cube
        all_actions_forward, all_states_forward = [], []
        for _ in range(n_steps):
            state = cube.observe()
            action = cube.action_space.sample() # random action
            cube.step(action)
            all_actions_forward.append(action)
            all_states_forward.append(state)
        
        # backward: unsolved cube to solved cube
        # note: there is a shift in time index due to the reversing
        all_states_backward = all_states_forward[-1:0:-1]
        all_actions_backward = all_actions_forward[-2::-1]
        
        X = np.vstack([X, all_states_backward])
        y = np.vstack([y, all_actions_backward])

    print('Done!')

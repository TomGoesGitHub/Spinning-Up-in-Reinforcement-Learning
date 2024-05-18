import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class RolloutAnimation:
    '''
    Animation of the environment-rollout based on
    matplotlib.animation.FuncAnimation API
    '''
    def __init__(self, env, agent=None, n=1, max_len=None):
        self.environment = env
        self.agent = agent
        self.n = n
        self.max_len = np.inf if max_len is None else max_len
        self._reset()

    def _animate_frame(self, i):
        '''Callable that returns an iterable of artist, as required
        by matplotlib.animation.FuncAnimation. To be called at each 
        step of the environment.'''
        if self.agent is None:
            action = self.environment.action_space.sample() 
        else:
            action = self.agent.act(np.array([self.env_state]))[0].numpy()
        
        self.env_state, _, terminated, truncated, _ = self.environment.step(action)
        img = self.environment.render()
        self.frame.set_data(img)
        if truncated or terminated or (i%self.max_len==0 and i>0):
            self.count += 1
            self.env_state, _ = self.environment.reset()
        self.stop_animating = (self.count >= self.n)
        return self.frame

    def _frame_idx_generator(self):
        '''Yield increasing frame-indecies, while not exhausted.'''
        i=0
        while not self.stop_animating:
            i += 1
            yield i

    def _reset(self):
        '''Reset internal state.'''
        # state
        self.env_state, _ = self.environment.reset()
        self.stop_animating = False # conditon that exhausts the idx-generator
        self.count = 0 # number of renderered rollouts, to be increased up to self.n

        # canvas
        self.fig = plt.figure()
        self.fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)
        self.frame = plt.imshow(self.environment.render())
        plt.gca().axis('off')

    def animate(self, filepath):
        '''Create animation and save GIF to specified location.'''
        self._reset()
        animated_rollout = animation.FuncAnimation(self.fig, func=self._animate_frame,
                                                   frames=self._frame_idx_generator,
                                                   interval=20, repeat=True,
                                                   cache_frame_data=False)
        animated_rollout.save(filepath, dpi=100)


if __name__ == '__main__':
    import gymnasium as gym
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    
    my_animation = RolloutAnimation(env=gym.make("BipedalWalker-v3", render_mode='rgb_array'))
    
    print('animating...')
    my_animation.animate(filepath='tmp.gif')
    print('Done!')
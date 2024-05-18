import os
import sys
import csv

import numpy as np

sys.path.append(os.path.join(*2*[os.pardir]))
from learning.callbacks import Callback
from learning.utils.common import rollout
from environments.rubiks_cube import RubiksCube

class DistortionDepthCallback(Callback):
    def __init__(self, env, freq=1000, n_evals=50, verbose=False, filepath=None):
        super().__init__(verbose)
        self.env = env
        assert isinstance(self.env, RubiksCube) # this Callback is only applicable for the Rubiks-Cube
        self.clock = 0
        self.freq = freq
        self.threshhold = 0.9
        self.n_evals = n_evals
        self.filepath = filepath

    def _on_step_end(self):
        self.clock += 1
        if self.clock % self.freq == 0:
            # evaluate 
            n_solved = 0
            scores = []
            
            for i in range(self.n_evals):
                dataset = rollout(self.env, self.agent)
                _, _, _, rewards, terminals, _ = dataset.get_all()
                rollout_return = np.sum(np.array(rewards))
                scores.append(rollout_return)
                is_solved = terminals[-1]
                n_solved += is_solved
            
            mean, std = np.mean(scores), np.std(scores)
            succes_rate = n_solved / self.n_evals

            # logging
            info_text = f'After {self.clock} training steps, the agent reaches the following performance:'\
                        + ' '.join([str(round(mean, 4)), u'\u00B1', str(round(std, 4))])
            print(info_text)
            
            if self.filepath:
                already_exists = os.path.exists(self.filepath)
                with open(self.filepath, mode='a', newline='\n') as csvfile:
                    row = {'n_steps': self.clock, 'return_mean': mean, 'return_std': std}
                    writer = csv.DictWriter(csvfile, fieldnames=row.keys()) 
                    if not already_exists:
                        writer.writeheader()
                    writer.writerow(row)
            
            # adapt complexity and save model
            if succes_rate >= self.threshhold:
                old_complexity = self.env.distortion_depth
                self.env.distortion_depth += 1
                print(f'After {self.clock} rollouts, the complexity of the Cube was increased to depth={self.env.distortion_depth}.')
                self.agent.policy.reset()

                dir = 'checkpoints'
                if not os.path.exists(dir):
                    os.makedirs(dir)
                path = os.path.join(dir, f'solved_complexity={old_complexity}.weights.h5')
                self.agent.save_weights(path)
    
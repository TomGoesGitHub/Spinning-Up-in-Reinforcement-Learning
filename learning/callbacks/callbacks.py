from .callback import Callback
from ..agents.actor_critic.actor_critic import ActorCritic # todo
from ..utils.common import rollout
from ..utils.visualization.rollout_animation import RolloutAnimation
import os
import csv
import numpy as np


class PerformanceEvaluationCallback(Callback):
    '''
    Every freq steps, n_evals evaluations are ran with different random seeds
    and the mean and standard deviation are computed over those random seeds.
    
    The default parameter-values are taken from:
    https://spinningup.openai.com/en/latest/spinningup/bench.html
    '''
    def __init__(self, env, freq=10000, n_evals=10, discount=False, verbose=True, filepath=None): # todo: add different metrices
        super().__init__(verbose)
        assert verbose or filepath is not None, 'PerformanceEvaluationCallback \
                         should either be used with inline-print or file-logging.'
        self.environment = env
        self.freq = freq
        self.n_evals = n_evals
        self.discount = discount
        
        self.clock = 0 # to be updated on step
        self.evaluate_performance = True # flag, to be updated on condition
        self.filepath = filepath
    
    def _evaluate_performance(self):
        scores = []
        for _ in range(self.n_evals):
            rollout_data = rollout(self.environment, self.agent)
            _, _, _, rewards, _, _ = rollout_data.get_all()
            if self.discount:
                discounts = np.power(self.agent.gamma, np.arange(len(rewards)))
                rollout_return = np.sum(np.array(rewards) * discounts)
            else:
                rollout_return = np.sum(np.array(rewards))
            scores.append(rollout_return)
        mean, std = np.mean(scores), np.std(scores)

        if self.verbose:
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

    def _on_step_end(self):
        self.clock += 1
        if self.clock % self.freq == 0:
            self._evaluate_performance()
    
    def _on_rollout_end(self):
        pass

    def _on_episode_end(self):
        self.plot_learning_curve()

    def plot_learning_curve():
        pass # todo


class RenderEnvironmentCallback(Callback):
    def __init__(self, env, directory, freq, n=1, max_rollout_len=None, verbose=True):
        super().__init__(verbose)
        self.env = env
        self.freq = freq
        self.n = n
        self.max_rollout_len = max_rollout_len

        self.directory = directory
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

        self.clock = -1
    
    def init_callback(self, agent):
        # note: we explicitly do not set the environment here, since it
        #       needs to be sepcified during construction
        self.agent = agent
        self.animation = RolloutAnimation(self.env, self.agent, self.n, max_len=self.max_rollout_len)

    def _on_step_end(self):
        self.clock += 1
        if self.clock % self.freq == 0:
            filepath = os.path.join(self.directory, f'after_{self.clock}_steps.gif')
            self.animation.animate(filepath)

            

class SaveModelCallback(Callback):
    def __init__(self, dir):
        self.dir = dir
        self.epoch_count = 1

    def _on_epoch_end(self):
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)
        path = os.path.join(self.dir, f'after_epoch_{self.epoch_count:03d}.weights.h5')
        self.agent.save_weights(path)
        self.epoch_count += 1
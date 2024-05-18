import numpy as np
import os

from learning.utils.common import rollout
from work import ActorCritic # todo: put into correct package

class Callback():
    def __init__(self, verbose=False):
        self.globals = globals()
        self.locals = None # to be updated via update_locals_decorator()

        self.agent = None # to be updated via init_callback()
        self.environment = None # to be updated via init_callback()

        self.verbose = verbose

    def init_callback(self, agent, environment):
        self.agent = agent
        self.environment = environment

    def update_locals_decorator(self, func):
        def wrapped(self, *args, **kwargs):
            self.locals = locals()
            return func(*args, **kwargs)
        return wrapped

    def _on_training_start(self, locals, globals) -> None:
        self.locals = locals
        self.globals = globals

    def _on_epoch_start(self) -> None:
        pass

    def _on_rollout_start(self) -> None:
        pass

    def _on_step_start(self) -> None:
        pass

    def _on_step_end(self) -> None:
        pass

    def _on_rollout_end(self) -> None:
        pass

    def _on_epoch_end(self) -> None:
        pass

    def _on_training_end(self) -> None:
        pass


class CallbackList(Callback):
    def __init__(self, callbacks=[]):
        self.callbacks = callbacks
    
    def init_callback(self, agent, environment):
        super().init_callback(agent, environment)
        for callback in self.callbacks:
            callback.init_callback(agent, environment)
    
    def _on_training_start(self, locals, globals):
        for callback in self.callbacks:
            callback._on_training_start(locals, globals)

    def _on_training_end(self):
        for callback in self.callbacks:
            callback._on_training_end()

    def _on_epoch_start(self):
        for callback in self.callbacks:
            callback._on_epoch_start()
    
    def _on_epoch_end(self):
        for callback in self.callbacks:
            callback._on_epoch_end()

    def _on_rollout_start(self):
        for callback in self.callbacks:
            callback._on_rollout_start()
    
    def _on_rollout_end(self):
        for callback in self.callbacks:
            callback._on_rollout_end()
    
    def _on_step_start(self):
        for callback in self.callbacks:
            callback._on_step_start()

    def _on_step_end(self):
        for callback in self.callbacks:
            callback._on_step_end()



# class DistortionDepthCallback(Callback):
#     def __init__(self, freq=1000, verbose=False):
#         super().__init__(verbsoe)
#         self.clock = 0
#         self.freq = freq
#         self.threshhold = 0.95
#         self.n_tests = 50

#     def _on_rollout_end():
#         self.clock += 1
#         if self.clock % self.freq == 0:
#             n_solved = 0
#             for i in range(n_tests):
#                 data = rollout(env, agent)
#                 is_solved = data.terminated[-1]
#                 n_solved += is_solved
#             succes_rate = n_solved / self.n_tests
            
#             if succes_rate >= self.threshhold:
#                 environment.distortion_depth += 1


class PerformanceEvaluationCallback(Callback):
    '''
    Every freq steps, n_evals evaluations are ran with different random seeds
    and the mean and standard deviation are computed over those random seeds.
    
    The default parameter-values are taken from:
    https://spinningup.openai.com/en/latest/spinningup/bench.html
    '''
    supported_metrics = ['return']
    def __init__(self, verbose=True, freq=10000, n_evals=10, smoothing=11): # todo: add different metrices
        super().__init__(verbose)
        self.freq = freq
        self.n_evals = n_evals
        self.smoothing = smoothing
        
        self.clock = 0 # to be updated on step
        self.evaluate_performance = True # flag, to be updated on condition
        
        self.history_t = []
        self.history_mean = []
        self.history_std = []
    
    def _on_step_end(self):
        self.clock += 1
        if self.clock % self.freq == 0:
            # note: since an evaluation of the environment requires a reset, we  
            #       delay the evaluation until the current rollouts end.
            self.evaluate_performance = True
    
    def _on_rollout_end(self):
        if self.evaluate_performance:
            self.evaluate_performance = False
            scores = []
            for _ in range(self.n_evals):
                rollout_data = rollout(self.environment, self.agent)
                discounts = np.power(self.agent.gamma, np.arange(len(rollout_data.rewards)))
                rollout_return = np.sum(np.array(rollout_data.rewards) * discounts)
                scores.append(rollout_return)
            mean, std = np.mean(scores), np.std(scores)

            self.history_mean.append(mean)
            self.history_std.append(std)
            self.history_t.append(self.clock)

            if self.verbose:
                info_text = f'After {self.clock} training steps, the agent reaches the following performance:'\
                            + ' '.join([str(round(mean, 4)), u'\u00B1', str(round(std, 4))])
                print(info_text) # todo: logging
    
    def _on_episode_end(self):
        self.plot_learning_curve()

    def plot_learning_curve():
        pass # todo

class SaveModelCallback(Callback):
    def __init__(self, dir):
        self.dir = dir
        self.epoch_count = 1

    def _on_epoch_end(self):
        if isinstance(self.agent, ActorCritic):
            self.model.actor_network.save(os.path.join(self.dir, f'epoch{self.epoch_count:02d}_actor.keras'))
            self.model.critic_network.save(os.path.join(self.dir, f'epoch{self.epoch_count:02d}_critic.keras'))
        else:
            raise NotImplementedError # todo: add new cases on demand
        self.epoch_count += 1
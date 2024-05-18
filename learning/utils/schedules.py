import numpy as np
import tensorflow as tf
# import tensorflow as tf

class Schedule:
    '''x[t+1] = f(x[t])'''
    def __init__(self, initial_value):
        self.initial_value = initial_value
        self.value = tf.Variable(initial_value, dtype=tf.float32)

    def _step(self):
        raise NotImplementedError
    
    def reset(self):
        self.value = self.initial_value
    
    def __call__(self, step=False):
        if step:
            self._step()
        return self.value

class ConstantSchedule(Schedule):
    '''x[t+1] = x[t]'''
    def _step(self):
        pass

class LinearSchedule(Schedule):
    '''x[t+1] = x[t] + c'''
    def __init__(self, initial_value, c, bounds=(-np.inf, +np.inf)):
        super().__init__(initial_value)
        self.c = c
        self.bounds = bounds
    
    def _step(self):
        new_value = tf.clip_by_value(self.value + self.c, *self.bounds)
        self.value.assign(new_value)

# todo: tf.Variable API 
# class CompoundSchedule(Schedule):
#     '''x[t+1] = c * x[t]'''
#     def __init__(self, initial_value, c, bounds=(-np.inf, +np.inf)):
#         super().__init__(initial_value)
#         self.c = c
#         self.bounds = bounds
    
#     def _step(self):
#         self.value = np.clip(self.c * self.value, *self.bounds)

# class ExponentialShedule(Schedule):
#     '''x[t+1] = exp(c*x[t])'''
#     def __init__(self, initial_value, c, bounds=(-np.inf, +np.inf)):
#         super().__init__(initial_value)
#         self.c = c
#         self.bounds = bounds
    
#     def _step(self):
#         self.value = np.clip(np.exp(self.c * self.value), *self.bounds)
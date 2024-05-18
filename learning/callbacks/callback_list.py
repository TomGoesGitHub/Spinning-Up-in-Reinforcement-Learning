from .callback import Callback

class CallbackList(Callback):
    def __init__(self, callbacks=[]):
        super().__init__()
        self.callbacks = callbacks
    
    def init_callback(self, agent):
        super().init_callback(agent)
        for callback in self.callbacks:
            callback.init_callback(agent)
    
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
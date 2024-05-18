class Callback():
    def __init__(self, verbose=False):
        self.globals = globals()
        self.locals = None # to be updated via update_locals_decorator()

        self.agent = None # to be updated via init_callback()

        self.verbose = verbose

    def init_callback(self, agent):
        self.agent = agent

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
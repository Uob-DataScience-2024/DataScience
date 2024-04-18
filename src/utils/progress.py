from rich.progress import Progress


class CallbackProgress(Progress):
    def __init__(self, *args, new_progress=None, update=None, remove_progress=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.new_progress = new_progress
        self.update_callback = update
        self.remove_progress = remove_progress

    def add_task(self, *args, **kwargs):
        task = super().add_task(*args, **kwargs)
        if self.new_progress:
            self.new_progress(task, description=args[0], total=kwargs.get('total', -1))
        return task

    def update(self, task, *args, **kwargs):
        super().update(task, *args, **kwargs)
        if self.update_callback:
            self.update_callback(task, kwargs.get('advance', 1), description=kwargs.get('description', ''))

    def remove_task(self, task):
        super().remove_task(task)
        if self.remove_progress:
            self.remove_progress(task)

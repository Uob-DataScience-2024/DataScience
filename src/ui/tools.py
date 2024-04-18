import json
import os

import gradio as gr


class ProcessManager:
    def __init__(self, progress: gr.Progress = None, disable: bool = False):
        # self.progress = progress
        self.tasks = {}
        self.disable = disable

    def on_new_task(self, task, description, total):
        if self.disable:
            return
        self.tasks[task] = {
            'description': description,
            'total': total,
            'current': 0,
            'progress': gr.Progress()
        }

    def on_update(self, task, advance, description):
        if self.disable:
            return
        self.tasks[task]['current'] += advance
        self.tasks[task]['progress']((self.tasks[task]['current'], self.tasks[task]['total']), desc=self.tasks[task]['description'] + description)

    def on_remove(self, task):
        if self.disable:
            return
        del self.tasks[task]


def load_config(config_path):
    with open(config_path, 'r') as f:
        json_data = json.load(f)
    return json_data["config_ordered"][0]


def save_config(config_dir, *configs):
    current = len([x for x in os.listdir(config_dir) if x.endswith('.json')])
    index = 0
    while True:
        config_path = os.path.join(config_dir, f'config_{current + index}.json')
        if not os.path.exists(config_path):
            break
        index += 1
    with open(config_path, 'w') as f:
        json.dump({"config_ordered": configs}, f)

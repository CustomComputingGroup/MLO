import logging
from copy import copy, deepcopy
import platform
from threading import Semaphore

from model.run import Run
from visualizer import ParallelisedVisualizer, SingleThreadVisualizer

class Controller(object):

    def __init__(self, restart):
        self.restart = restart
        self.trials = {}
        max_training_process = 3
        self.training_sema = Semaphore(value=max_training_process)

        # Use different visualizer on OS X
        if platform.system() == 'Darwin':
            self.visualizer = SingleThreadVisualizer(self)
        else:
            self.visualizer = ParallelisedVisualizer(self)
            
        self.visualizer.daemon = True
            
    def take_over_control(self):
        self.visualizer.start()
        self.view.initialize(self)
        self.visualizer.join(10000000) ## check... gui ctr+c might not work, setting 100000000000 time might work :)
        
    def start_run(self, name, fitness, configuration):
        if not name:
            name = 'Default Run'
        run = Run(name, fitness, configuration, self)
        run.daemon = True
        # Special case when we are restarting a previously crashed run
        if self.restart:
            run.restart()
            return run

        run.run()
        return run

    def load_run(self, run_path):
        run = Run(None, None, None, self)
        run.results_folder_path = run_path
        run.load()

    def register_trial(self, trial):
        self.trials[trial.get_name()] = trial

    def get_trial_status(self, name):
        return self.trials[name].get_status()

    def find_trial(self, trial_name):
        return self.trials[trial_name]

    def delete_trial(self, name):
        trial = self.trials[name]
        self.trials = {key: value for key, value
                       in self.trials.items()
                       if key != name}

    def pause_trial(self, name):
        trial = self.trials[name]
        trial.wait = True
        trial.status = "Paused"
        self.view_update(trial)

    def resume_trial(self, name):
        trial = self.trials[name]
        trial.wait = False
        trial.status = "Running"
        self.view_update(trial)

    def get_graph_attributes(self, trial, name):
        return self.plot_view.get_attributes(name)

    def visualize(self, snapshot, render_function):
        self.visualizer.add_job(render_function, snapshot)

    def view_update(self, trial):
        self.view.update(trial)

    def kill_visualizer(self):
        self.visualizer.terminate()

    def acquire_training_sema(self):
        self.training_sema.acquire()

    def release_training_sema(self):
        self.training_sema.release()

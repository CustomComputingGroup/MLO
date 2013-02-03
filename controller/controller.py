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
        self.runs = {}
        max_training_process = 3
        self.training_sema = Semaphore(value=max_training_process)

        # Use different visualizer on OS X
        if platform.system() == 'Darwin':
            self.visualizer = SingleThreadVisualizer(self)
        else:
            self.visualizer = ParallelisedVisualizer(self)
            
        self.visualizer.daemon = True
            
    ## gui methods
            
    def view_update(self, trial):
        self.view.update(trial)
            
    def take_over_control(self):
        self.visualizer.start()
        self.view.initialize(self)
        self.visualizer.join(10000000) ## check... gui ctr+c might not work, setting 100000000000 time might work :)
        
    ## Run class methods
        
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
        run.set_results_folder_path(run_path)
        run.load()

    def delete_run(self, name):
        run = self.runs[name]
        self.runs = {key: value for key, value
                       in self.trials.items()
                       if key != name}
                       
    def register_run(self, run):
        self.runs[run.get_name()] = run

    def get_run_status(self, name):
        return self.runs[name].get_status()

    def find_run(self, name):
        return self.runs[name]

    def delete_run_trials(self, name):
        run = find_run(name)
        runs_trials = run.get_trials()
        for trial in runs_trials:
            delete_trial(self, trial.get_name())
                       
    def pause_run(self, name):
        run = self.runs[name]
        run.set_wait(True)
        run.set_status("Paused")
        runs_trials = run.get_trials()
        for trial in runs_trials:
            pause_trial(self, trial.get_name())
        
    def resume_run(self, name):
        run = self.runs[name]
        run.set_wait(False)
        run.set_status("Running")
        runs_trials = run.get_trials()
        for trial in runs_trials:
            resume_trial(self, trial.get_name())
        
    ### trial managment methods
        
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
        trial.set_wait(True)
        trial.set_status("Paused")
        self.view_update(trial)

    def resume_trial(self, name):
        trial = self.trials[name]
        trial.set_wait(False)
        trial.set_status("Running")
        self.view_update(trial)

    ## visualizer methods

    def visualize(self, snapshot, render_function):
        self.visualizer.add_job(render_function, snapshot)

    def kill_visualizer(self):
        self.visualizer.terminate()

    ## training schemas for the regressors. neccesary as they are not thread safe. 
        
    def acquire_training_sema(self):
        self.training_sema.acquire()

    def release_training_sema(self):
        self.training_sema.release()

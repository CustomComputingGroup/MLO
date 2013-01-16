import logging
from copy import copy, deepcopy
import platform

from run import Run
from views.modes import GUIView, TerminalView
from visualizer import ParallelisedVisualizer, SingleThreadVisualizer
from threading import Semaphore


class Controller(object):

    def __init__(self, run_with_gui, restart):
        self.restart = restart
        self.trials = {}
        max_training_process = 3
        self.training_sema = Semaphore(value=max_training_process)

        # Use different visualizer on OS X
        if platform.system() == 'Darwin':
            self.visualizer = SingleThreadVisualizer(self)
        else:
            self.visualizer = ParallelisedVisualizer(self)

        if run_with_gui:
            # GUI mode
            self.view = GUIView()
            self.visualizer.daemon = True
        else:
            # Command line mode
            self.view = TerminalView()

    def take_over_control(self):
        self.visualizer.start()
        self.view.initialize(self)

    def run_in_terminal(self):
        """
        Runs trials in the terminal mode.
        """
        if not self.restart and not (self.fitness and self.configuration):
            logging.error('Benchmark and/or configuration script not '
                          'provided, terminating...')
            return

        self.start_run('Default run', self.fitness, self.configuration).join()
        self.visualizer.terminate()

    def start_run(self, name, fitness, configuration):
        if not name:
            name = 'Default Run'
        run = Run(name, fitness, configuration, self)

        # Special case when we are restarting a previously crashed run
        if self.restart:
            run.restart()
            return run

        run.run()
        return run

    def reload_run(self, run_path):
        run = Run(None, None, None, self)
        run.results_folder_path = run_path
        run.reload()

    def register_trial(self, trial):
        self.trials[trial.get_name()] = trial

    def get_trial_status(self, name):
        return self.trials[name].status

    def find_trial(self, trial_name):
        return self.trials[trial_name]

    def delete_trial(self, name):
        trial = self.trials[name]
        trial.progress_bar.Destroy()
        self.trials = {key: value for key, value
                       in self.trials.items()
                       if key != name}
        #trial.die()

    # TODO - can this be removed?
    #def finish_run(self, run):
    #   if not self.run_with_gui:
    #      logging.info('All trials finished')
    #    else:
    #       logging.info('All trials of {} finished'.format(run.name))

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
        return trial.plot_view.get_attributes(name)

    def visualize_trial(self, trial):
        snapshot = trial.plot_view.snapshot(trial)
        snapshot.update(trial.graph_dictionary)
        self.visualizer.add_job(trial.plot_view.render, snapshot, trial)

    def revisualize_trial(self, trial, generation):
        gd = trial.graph_dictionary  # Save graph_dictionary
        Trial = trial.configuration.trials_type
        vis_trial = Trial(trial.trial_no, trial.name, trial.fitness,
                          trial.configuration, trial.controller,
                          trial.run_results_folder_path)
        vis_trial.results_folder = trial.results_folder
        vis_trial.load(generation)
        vis_trial.graph_dictionary = gd

        self.visualize_trial(vis_trial)
        #self.view_graph_update(trial)

    def view_update(self, trial):
        self.view.update(trial)

    def view_graph_update(self, trial, counter_plot):
        self.view.regen(trial, counter_plot)

    def kill_visualizer(self):
        self.visualizer.terminate()

    def acquire_training_sema(self):
        self.training_sema.acquire()

    def release_training_sema(self):
        self.training_sema.release()

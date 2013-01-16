import logging
import os
import sys
from threading import Thread
from time import strftime

from utils import load_script


class Run(object):

    def __init__(self, name, fitness, configuration, controller):
        self.trials = []
        if configuration:
            self.no_of_trials = configuration.trials_count
        self.name = name
        self.fitness = fitness
        self.configuration = configuration
        self.controller = controller

    def run(self):
        # Initialise results folder
        now = strftime('%Y-%m-%d_%H-%M-%S')
        self.results_folder_path = '{}/{}'.format(
            self.configuration.results_folder_path, now)

        Trial = self.configuration.trials_type

        # Initialise trials
        for trial_no in range(1, self.no_of_trials + 1):
            trial = Trial(trial_no, self.name, self.fitness,
                          self.configuration, self.controller,
                          self.results_folder_path)
            if trial.initialise():
                self.trials.append(trial)

        self.save_run_data()

        # Run trials
        for trial in self.trials:
            trial.start()

    def join(self):
        for trial in self.trials:
            trial.join()

    def restart(self):
        """
        Restarting the Run after it crashed.
        """
        logging.debug('Restarting the Run')

        try:
            with open('run_data.txt', 'r') as run_data:
                self.results_folder_path = run_data.readline().rstrip('\n')
        except:
            logging.error('Error loading run_data.txt. Cannot restart without '
                          'this file. Terminating...')
            print sys.exc_info()[1]
            sys.exit(1)

        self.load_run_data()

        # Run trials
        for trial in self.trials:
            trial.counter_dictionary['g'] += 1
            trial.start()

    def reload(self):
        """
        Reloading Run when reopened from CSV and TXT files.
        """
        logging.debug('Reloading the Run')

        self.load_run_data()

        # Run trials
        for trial in self.trials:
            trial.counter_dictionary['g'] += 1
            trial.start()

    def save_run_data(self):
        """
        Saves settings of the run so that they can be recovered later.
        """
        # Save run data needed for automatic recovery
        with open('run_data.txt', 'w') as run_data:
            run_data.write(self.results_folder_path)
            run_data.write('\n')

        # Save run data needed for reloading the run
        fitness_file_name = os.path.abspath(self.fitness.__file__)
        if fitness_file_name[-1] == 'c':  # .pyc
            fitness_file_name = fitness_file_name[:-1]
        configuration_file_name = os.path.abspath(
            self.configuration.__file__)
        if configuration_file_name[-1] == 'c':  # .pyc
            configuration_file_name = configuration_file_name[:-1]
        with open(self.results_folder_path + '/run_data.txt', 'w') as run_data:
            run_data.write(self.name)
            run_data.write('\n')
            run_data.write(fitness_file_name)
            run_data.write('\n')
            run_data.write(configuration_file_name)
            run_data.write('\n')
            for trial in self.trials:
                run_data.write(trial.results_folder)
                run_data.write('\n')

    def load_run_data(self):
        """
        Loads settings of a previously crashed run from the disc.
        """
        trial_results_folders = []

        run_file = self.results_folder_path + '/run_data.txt'
        try:
            with open(run_file, 'r') as run_data:
                self.name = run_data.readline().rstrip('\n')
                fitness = run_data.readline().rstrip('\n')
                self.fitness = load_script(fitness, 'fitness')
                configuration = run_data.readline().rstrip('\n')
                self.configuration = load_script(configuration,
                                                 'configuration')
                if not self.fitness or not self.configuration:
                    raise 'Error loading scripts'

                for line in run_data:
                    trial_results_folders.append(line.rstrip('\n'))
        except:
            logging.error('Error loading run\'s run_data.txt. Cannot restart '
                          'without this file.')
            logging.error('Could not load fitness and/or configuration '
                          'scripts. Terminating...')
            print sys.exc_info()[1]
            sys.exit(1)

        self.no_of_trials = self.configuration.trials_count
        Trial = self.configuration.trials_type

        for trial_no in range(1, self.no_of_trials + 1):
            trial = Trial(trial_no, self.name, self.fitness,
                          self.configuration, self.controller,
                          self.results_folder_path)
            trial.results_folder = trial_results_folders[trial_no - 1]
            trial.run_initialize()
            if not trial.load():
                logging.error('Failed loading a trial from {}'.format(
                    trial.results_folder))
                trial.initialize_population()
            self.trials.append(trial)

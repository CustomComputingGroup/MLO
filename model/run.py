import logging
import os
import sys
from threading import Thread
from time import strftime
import io
import pickle 
import time
from time import strftime
from datetime import datetime, timedelta

from utils import load_script, get_trial_constructor


class Run(object):

    def __init__(self, name, fitness, configuration, controller):
        self.trials = []
        self.fitness = fitness
        self.configuration = configuration
        self.controller = controller
        fitness_file_name = None
        fitness_folder_name = None
        configuration_file_name = None
        configuration_folder_name = None
        trial_type = None
        if fitness:
            fitness_file_name = os.path.abspath(self.fitness.__file__)
            if fitness_file_name[-1] == 'c':  # .pyc
                fitness_file_name = fitness_file_name[:-1]
            fitness_folder_name = os.path.dirname(self.fitness.__file__)
        if configuration:
            configuration_file_name = os.path.abspath(
                self.configuration.__file__)
            if configuration_file_name[-1] == 'c':  # .pyc
                configuration_file_name = configuration_file_name[:-1]
            configuration_folder_name = os.path.dirname(self.configuration.__file__)
            trial_type = configuration.trials_type
        self.state_dictionary = {"fitness_file_name": fitness_file_name,
                                 "fitness_folder_name": fitness_folder_name,
                                 "configuration_file_name" : configuration_file_name,
                                 "configuration_folder_name" : configuration_folder_name,
                                 "name" : name,
                                 "wait" : False,
                                 "status" : "Running",
                                 "trial_type" : trial_type,
                                 "trials_finished": 0,
                                 "total_time":0
                                 }
        if configuration:
            self.set_no_of_trials(configuration.trials_count)
        
    def run(self):
        now = strftime('%Y-%m-%d_%H-%M-%S')
        self.results_folder_path = '{}/{}'.format(
            self.configuration.results_folder_path, now)
        try:
            self.controller.register_run(self)
        except:
            return
            
        # Initialise results folder
            
        self.set_start_time(datetime.now().strftime('%d-%m-%Y  %H:%M:%S'))
        
        # Initialise trials
        for trial_no in range(0, self.get_no_of_trials()):
            trial = self.get_trial_type_constructor()(trial_no, self, self.fitness,
                          self.configuration, self.controller,
                          self.results_folder_path)
            if trial.initialise():
                self.trials.append(trial)
            
        self.save()
        self.view_update()
        # Run trials
        for trial in self.trials:
            trial.daemon = True
            trial.start()

    ## TODO - this has to be changed... its just that ctrl+c wont be propagated otherwise...    
    def join(self):
        for trial in self.trials:
            trial.join(100000000)
        
    def load(self):
        logging.info('Loading the Run')
        try:
            with open(self.get_run_file(), 'rb') as outfile:
                dict = pickle.load(outfile)
                self.configuration = load_script(dict["configuration_file_name"],'configuration')
                self.fitness = load_script(dict["fitness_file_name"],'fitness')
                self.set_state_dictionary(dict)
                #self.set_start_time(datetime.now().strftime('%d-%m-%Y  %H:%M:%S'))
        except Exception, e:
            logging.error('Error loading run\'s run_data.txt. Cannot restart '
                          'without this file. {}'.format(str(e)))
            logging.error('Could not load fitness and/or configuration '
                          'scripts. Terminating...')            
            return False
        
        try:
            self.controller.register_run(self)
        except Exception, e:
            logging.error('Unable to register run: {}'.format(str(e)))
            return False
        
        self.view_update(visualize=False)
        try:
            for trial_no in range(0, self.get_no_of_trials()): ###
                trial = self.get_trial_type_constructor()(trial_no, self, self.fitness,  self.configuration, self.controller, self.results_folder_path)
                trial.run_initialize()
                if not trial.load():
                    logging.error('Failed loading a trial no.{}'.format( trial.get_trial_no()))
                    return False
                ##print trial.state_dictionary
                self.trials.append(trial)
        except Exception, e:
            logging.error('Failed loading/initializing trials: {}'.format(str(e)))
            return False
            
        # Run trials
        # we do not catch it, it should be handeled elsewhere
        for trial in self.trials:
            trial.daemon = True
            trial.start()
        logging.info("in..")
        return True
        
    def save(self):
        try:
            dict = self.get_state_dictionary()
            with open(self.results_folder_path + '/run_data.txt', 'w') as outfile:
                pickle.dump(dict, outfile)    
        except Exception, e:
            logging.error(str(e))
            return False
                        
    ## creates a snapshot dictionary representing the state of the trials within the run.. basically get snapshots from all the trials.
    ## feel free to add anything extra you thing would be neccesary to generate views of a run
    def snapshot(self):
        snapshot_dict = {"trial_type" : self.get_trial_type(),
                         "name" : 'run_' + self.get_name(),
                         "status": self.get_status(),
                         "generate" : True,
                         "results_folder_path" : self.results_folder_path,
                         "trials_snapshots" : [trial.snapshot() for trial in self.trials]
                         }
        return snapshot_dict
        
    def view_update(self, visualize=False):
        self.controller.view_update(run=self, visualize=visualize)
        
    def trial_notify(self, trial):
        if self.get_status() == "Finished":
            pass
        else:
            self.set_running_time((trial.get_running_time() + self.get_running_time()).seconds)
            self.state_dictionary["trials_finished"] = self.state_dictionary["trials_finished"] + 1
            logging.info(str(self.state_dictionary["trials_finished"]))
            logging.info(str(self.state_dictionary["trials_finished"]/self.get_no_of_trials()))
            logging.info(str(self.get_no_of_trials()))
            if self.state_dictionary["trials_finished"] == self.get_no_of_trials():
                self.set_status("Finished")
                self.view_update(visualize=True)
            self.save()
        self.view_update(visualize=True)
        
        
    #############
    ## GET/SET ##
    #############
    def get_results_folder(self):
        return self.results_folder_path
    
    def get_run_type(self):
        return  self.state_dictionary["trial_type"]
    
    def get_trials_finished(self):
        return self.state_dictionary["trials_finished"]
        
    def get_trials(self):
        return self.trials
        
    def get_configuration(self):
        return self.configuration
    
    def get_start_time(self):
        return self.state_dictionary['start_time']
        
    def set_start_time(self, start_time):
        self.state_dictionary['start_time'] = start_time
        
    def get_predicted_time(self):
        return self.state_dictionary['start_time'] # TODO
        
    def get_running_time(self):
        return timedelta(seconds = self.state_dictionary["total_time"] )
        
    def set_running_time(self, time):
        self.state_dictionary["total_time"] = time
        
    def get_main_counter(self): ## TODO - 
        return 1
    
    def get_status(self):
        return self.state_dictionary["status"]

    def set_status(self, status):
        if not (self.state_dictionary["status"] == "Finished"):
            self.state_dictionary["status"] = status
            self.view_update(visualize=False)
            for trial in self.trials:
                trial.set_status(status)
    
    def get_wait(self):
        return self.state_dictionary["wait"]
    
    def set_wait(self, wait):
        self.state_dictionary["wait"] = wait
        for trial in self.trials:
            trial.set_wait(wait)
        
    def get_run_file(self):
        return self.results_folder_path + '/run_data.txt'
    
    def set_state_dictionary(self, dict):
        self.state_dictionary = dict
        
    def get_state_dictionary(self):
        return self.state_dictionary

    def get_no_of_trials(self):
        return self.state_dictionary["no_of_trials"]
        
    def set_no_of_trials(self, num):
        self.state_dictionary["no_of_trials"] = num    
        
    def set_name(self, name):
        self.state_dictionary["name"] = name
        
    def get_name(self):
        return self.state_dictionary["name"]
        
    def set_trial_type(self, trial_type):
        self.state_dictionary["trial_type"] = trial_type

    def get_trial_type(self):
        return self.state_dictionary["trial_type"]
        
    def get_trial_type_constructor(self):
        return get_trial_constructor(self.get_trial_type())
        
    def set_results_folder_path(self, folder):
        self.results_folder_path = folder

    def get_results_folder_path(self):
        return self.results_folder_path
        
    def get_configuration_script_file(self):
        return self.state_dictionary["configuration_file_name"].split("/")[-1]
        
    def get_fitness_script_file(self):
        return self.state_dictionary["fitness_file_name"].split("/")[-1]
        
    def get_configuration_script_folder(self):
        return "/".join(self.state_dictionary["configuration_file_name"].split("/")[:-1])
        
    def get_fitness_script_folder(self):
        return "/".join(self.state_dictionary["fitness_file_name"].split("/")[:-1])
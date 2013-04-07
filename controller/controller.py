import logging
from copy import copy, deepcopy
import platform
from threading import Semaphore
import pickle
import io
import os

from model.run import Run
from visualizer import ParallelisedVisualizer, SingleThreadVisualizer
from utils import get_trial_type_visualizer, get_run_type_visualizer

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
            
    def view_update(self, trial=None, run=None, visualize=False):
        if self.view: ## might not have been initialized yet
            try:
                self.view.update(trial=trial, run=run, visualize=visualize)
            except Exception,e:
                logging.debug("Error while visualizing.. " + str(e))
            
    def take_over_control(self):
        self.visualizer.start()
        self.view.initialize(self)
        self.visualizer.join(10000000) ## check... gui ctr+c might not work, setting 100000000000 time might work :)
        
    ## Run class methods
        
    def remove_run_name_jobs(self, run_name):
        self.visualizer.remove_run_name_jobs(run_name)
        
    def start_run(self, name, fitness, configuration, terminal_mode = False):
        if not name:
            name = 'Default_run'
        run = Run(name, fitness, configuration, self)
        run.daemon = True
        # Special case when we are restarting a previously crashed run
        run.terminal_mode=terminal_mode
        run.run()
        return run

    def load_run(self, run_path):
        logging.info(run_path) 
        run = Run(None, None, None, self)
        run.set_results_folder_path(run_path)
        return run.load()
        
    def delete_run(self, name):
        try:
            self.delete_run_trials(name)
        except Exception, e:
            pass
        try:
            self.remove_run_path(name)
        except Exception, e:
            pass
        try:
            self.remove_run_name_jobs(name)
        except Exception, e:
            pass
        try:
            run = self.runs[name]
            del self.runs[name] 
        except Exception, e:
            pass
            
    def get_run_visualization_dict(self, trial_type):
        try:
            return self.profile_dict["run_visualization_dict"][trial_type]
        except:
            logging.info("You have never visualized runs of this trial type, will update your profile using default configuration")
            self.update_run_visualization_dict(trial_type, get_run_type_visualizer(trial_type)["default"].get_default_attributes())
            return self.profile_dict["run_visualization_dict"][trial_type]
       
    def update_run_visualization_dict(self, trial_type, dict):
        self.profile_dict["run_visualization_dict"][trial_type] = dict
        self.save_profile_dict()
     
    def get_most_recent_run_name(self):
        return self.profile_dict["last_run"]
                       
    def register_run(self, run):
        if self.runs.has_key(run.get_name()):
            logging.info("Run name already exists, please select a different name or remove the conflicting run: " + run.get_name())
            raise
        else:
            self.runs[run.get_name()] = run
            self.profile_dict["last_run"] = run.get_name()
            self.profile_dict["most_recent_fitness_script_file"] = run.get_fitness_script_file()
            self.profile_dict["most_recent_fitness_script_folder"] = run.get_fitness_script_folder()
            self.profile_dict["most_recent_configuration_script_file"] = run.get_configuration_script_file()
            self.profile_dict["most_recent_configuration_script_folder"] = run.get_configuration_script_folder()
            self.profile_dict["run_paths"][run.get_name()] = run.get_results_folder_path()
            self.save_profile_dict()

    def get_run_status(self, name):
        return self.runs[name].get_status()

    def find_run(self, name):
        return self.runs[name]

    def run_exists(self, name):
        try:
            self.find_run(name)
            return True
        except:
            return False
        
    def delete_run_trials(self, name):
        run = self.find_run(name)
        runs_trials = run.get_trials()
        for trial in runs_trials:
            self.delete_trial(trial.get_name())
                               
    def pause_run(self, name):
        run = self.runs[name]
        run.set_paused()
            
    def resume_run(self, name):
        run = self.runs[name]
        run.set_running()
            
    ### trial managment methods
        
    def register_trial(self, trial):
        self.trials[trial.get_name()] = trial

    def get_trial_status(self, name):
        return self.trials[name].get_status()

    def find_trial(self, trial_name):
        return self.trials[trial_name]

    def delete_trial(self, name):
        trial = self.trials[name]
        trial.set_kill(True)
        del self.trials[name]
                       
    def pause_trial(self, name):
        trial = self.trials[name]
        trial.set_paused()

    def resume_trial(self, name):
        trial = self.trials[name]
        trial.set_running()

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

    def get_most_recent_fitness_script_folder(self):
        try:
            return self.profile_dict["most_recent_fitness_script_folder"]
        except:
            return os.getcwd()
                
    def get_most_recent_configuration_script_folder(self):
        try:
            return self.profile_dict["most_recent_configuration_script_folder"]
        except:
            return os.getcwd()
            
    def get_most_recent_fitness_script_file(self):
        try:
            return self.profile_dict["most_recent_fitness_script_file"]
        except:
            return ''
                
    def get_most_recent_configuration_script_file(self):
        try:
            return self.profile_dict["most_recent_configuration_script_file"]
        except:
            return ''
                
    def get_trial_visualization_dict(self, trial_type):
        try:
            return self.profile_dict["trial_visualization_dict"]["trial_type"]
        except:
            logging.info("You have never visualized this trial type, will update your profile using default configuration")
            self.update_trial_visualization_dict(trial_type, get_trial_type_visualizer(trial_type)["default"].get_default_attributes())
            return self.profile_dict["trial_visualization_dict"]["trial_type"]

    def update_trial_visualization_dict(self, trial_type, dict):
        self.profile_dict["trial_visualization_dict"]["trial_type"] = dict
        self.save_profile_dict()
    
    def save_profile_dict(self):
        try:
            logging.info("Updated controller profile dictionary")
            #logging.info(str(self.profile_dict))
            with io.open(self.profile_dict["dir"], 'wb') as outfile:
                pickle.dump(self.profile_dict, outfile)            
        except Exception, e:
            logging.info("Problem with controller profile dictionary")
            logging.error(str(e))
            return False
       
    def remove_run_path(self, name):
        del self.profile_dict["run_paths"][name]
        logging.info(self.profile_dict["run_paths"])
        self.save_profile_dict()
       
    def load_profile_dict(self, provided_dir=None):
        if provided_dir is None:
            dir = os.getcwd() + "/profile"
        try:
            with open(dir, 'rb') as outfile:
                dict = pickle.load(outfile)
            dict["dir"] = dir
            self.profile_dict = dict
            logging.info("Loaded profile dictionary")
            if self.restart:
                logging.info("Restarting previous runs..")
                delete_runs = []
                
                for run_name, run_path in self.profile_dict["run_paths"].items():
                    logging.info("Restarted run " + run_name)
                    success = self.load_run(run_path)
                    logging.info(str(success))
                    if success: 
                        self.pause_run(run_name)
                    else:
                        logging.info("Restarting run " + run_name + " failed")
                        logging.info("Conflicting run will be removed from the profile")
                        ## delete whatever was loaded for the run
                        delete_runs.append(run_name)
                for run_name in delete_runs: ## this is neccesary as you cannot delete elements from an iterator while
                                            ## iterating over it
                    self.delete_run(run_name)
        except Exception, e:
            logging.info("A new controller profile dictionary is going to be created")
            logging.info("The new profile was saved as backup, verify error: " + str(e))
            self.profile_dict = { "trial_visualization_dict" : {},
                                  "run_visualization_dict" : {},
                                  "last_run" : "",
                                  "dir" : os.getcwd() + "/profile",
                                  "most_recent_dir" : None,
                                  "most_recent_fitness_script_file" : '',
                                  "most_recent_fitness_script_folder" : os.getcwd(),
                                  "most_recent_configuration_script_file" : '',
                                  "most_recent_configuration_script_folder" : os.getcwd(),
                                  "run_paths": {}
                                }
            self.save_profile_dict()


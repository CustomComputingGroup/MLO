import logging
import os
import sys
from threading import Thread
import time
from time import strftime
from datetime import datetime, timedelta
from copy import copy, deepcopy
import io
import pickle
import re
from numpy import multiply, array, ceil, floor, maximum, minimum, mean, min, max, sqrt, square, transpose
from numpy.random import uniform, rand , random
from numpy.linalg import norm
from scipy.stats import uniform as scipy_uniform
import operator

from deap import base, creator, tools
toolbox = base.Toolbox()
##TODO - clean it up... only should be loaded when needed... 

from ..surrogatemodels.surrogatemodel import DummySurrogateModel, ProperSurrogateModel, LocalSurrogateModel
from ..surrogatemodels.costmodel import DummyCostModel, ProperCostModel
import lhs 

class Trial(Thread):

    def __init__(self, trial_no, my_run, fitness, configuration, controller, run_results_folder_path):
        Thread.__init__(self)
        self.fitness = fitness
        self.controller = controller
        
        # now it is a list
        self.surrogate_model = []
        # True if the user has selected to pause the trial
        
        if configuration.surrogate_type == "dummy":
            if self.fitness.objectives>1:
                for i in range(self.fitness.objectives):
                    self.surrogate_model.append(DummySurrogateModel(configuration,self.controller,self.fitness))
            else:
                self.surrogate_model = DummySurrogateModel(configuration,
                                                       self.controller,self.fitness)
        elif configuration.surrogate_type == "local":
            if self.fitness.objectives>1:
                for i in range(self.fitness.objectives):
                    self.surrogate_model.append(LocalSurrogateModel(configuration,self.controller,self.fitness))
            else:
                self.surrogate_model = LocalSurrogateModel(configuration,
                                                       self.controller,self.fitness)
        else:
            if self.fitness.objectives>1:
                for i in range(self.fitness.objectives):
                    self.surrogate_model.append(ProperSurrogateModel(configuration,self.controller,self.fitness))
            else:
                self.surrogate_model = ProperSurrogateModel(configuration,self.controller,self.fitness)
        # Contains all the counter variables that may be used for visualization
        counter_dictionary = {}
        counter_dictionary['fit'] = 0 ## we always want to record fitness of the best configurations
        counter_dictionary['cost'] = 0.0 ## we always want to record fitness of the best configurations
        
        timer_dictionary = {}
        timer_dictionary['Running_Time'] = 0 ## we always want to record fitness of the best configurations
        timer_dictionary['Model Training Time'] = 0 
        timer_dictionary['Cost Model Training Time'] = 0 
        timer_dictionary['Model Predict Time'] = 0
        timer_dictionary['Cost Model Predict Time'] = 0
        self.configuration = configuration
        
        self.my_run = my_run
        run_name = self.my_run.get_name()
        
        self.state_dictionary = {
            'status' : 'Waiting',
            'retrain_model' : False,
            'model_failed' : False,
            'run_results_folder_path' : run_results_folder_path,
            'run_name' : run_name,
            'trial_type' : self.configuration.trials_type, 
            'trial_no' : trial_no,
            'name' : str(run_name) + '_' + str(trial_no),
            'all_particles_in_invalid_area' : False,
            'wait' : True,
            'generations_array' : [],
            'best_fitness_array' : [],
            'enable_traceback' : configuration.enable_traceback,
            'counter_dictionary' : counter_dictionary,
            'timer_dict' : timer_dictionary,
            'best' : None,
            'generate' : False,
            'fitness_state' : None,
            'fresh_run' : False
            # True if the user has selected to pause the trial
        }
        self.set_start_time(datetime.now().strftime('%d-%m-%Y  %H:%M:%S'))
        self.kill = False
        self.total_time = timedelta(seconds=0)
        self.previous_time = datetime.now()
        self.controller.register_trial(self)
        self.view_update(visualize = False)
    ####################
    ## Helper Methods ##
    ####################
    def training_set_updated(self):
        retrain_model_temp = self.get_retrain_model()
        self.set_retrain_model(False)
        return retrain_model_temp
        
    def is_better(self, a, b):
        try:
            if self.configuration.goal == "min":
                return a < b
            elif self.configuration.goal == "max":
                return a > b
            else:
                logging.info("configuration.goal attribute has an invalid value:" + str( self.configuration.goal))
        except:
            logging.info("configuration.goal has not been defined")
        logging.info("configuration.goal is assumed to be minimization")
        self.configuration.goal = "min"
        return a < b
    
    def hypercube(self):
        #find maximum
        max_diag = deepcopy(self.get_population()[0])
        for part in self.get_population():
            max_diag = maximum(part,max_diag)
        ###find minimum vectors
        min_diag = deepcopy(self.get_population()[0])
        for part in self.get_population():
            min_diag = minimum(part,min_diag)
        return [max_diag,min_diag]
    
    def train_surrogate_model(self, population):
        logging.info('Training surrogate model')
        start = datetime.now()
        if self.fitness.objectives>1:
            for i in range(self.fitness.objectives):
                logging.info("Training model for fitness function "+ str(i+1))
                self.set_model_failed(self.surrogate_model[i].train(self.hypercube()))
        else:
            self.set_model_failed(self.surrogate_model.train(self.hypercube()))
        diff = datetime.now() - start
        self.add_train_surrogate_model_time(diff)
    
    # for only one objective
        
    def predict_surrogate_model(self, population):
        start = datetime.now()
        prediction = self.surrogate_model.predict(population)
        diff = datetime.now() - start
        self.add_predict_surrogate_model_time(diff)
        return prediction
        
    
    def predict_surrogate_mo_model(self, population):
        start = datetime.now()
        prediction = []
        try:
            for i in range(self.fitness.objectives):
                prediction.append(self.surrogate_model[i].predict(population))
        except:
            logging.error("Prediction fails..")
        diff = datetime.now() - start
        self.add_predict_surrogate_model_time(diff)
        if self.fitness.objectives==1:
            return prediction[0]
        
        # process the prediction 
        else:

            Code = [] 
            MU = []
            S2 = []
            for i in range(len(prediction[0][0])):
                Code.append(0)
                S2.append(0)
            
            MU_list = []
            for p,(code,mu,s2) in enumerate(prediction):
                for i,(c,m,s) in enumerate( zip(code,mu,s2) ):
                    if c == 1:
                        Code[i] = 1
                    if S2[i] < s:
                        S2[i] = s
                MU_list.append(mu)
            MU_array = array(MU_list)
            MU_array = transpose(MU_array)[0]
            try : 
            
                for i in range(len(MU_array)):
                    tuplelet = ()
                    for j in range(self.fitness.objectives):
                        tuplelet = tuplelet + (MU_array[i][j],)
                    MU.append(tuplelet)
            except:
                pass
            return Code, MU, S2
    
    def train_cost_model(self, population):
        logging.info('Training cost model')
        start = datetime.now()
        self.set_model_failed(self.cost_model.train())
        diff = datetime.now() - start
        self.add_train_cost_model_time(diff)
        
    def predict_cost_model(self, population):
        start = datetime.now()
        prediction = self.cost_model.predict(population)
        diff = datetime.now() - start
        self.add_predict_cost_model_time(diff)
        return prediction
        
    def set_kill(self, kill):
        self.kill = kill
               
    def create_results_folder(self):
        """
        Creates folder structure used for storing results.
        Returns a results folder path or None if it could not be created.
        """
        path = str(self.get_run_results_folder_path()) + '/trial-' + str(self.get_trial_no())
        try:
            os.makedirs(path)
            os.makedirs(path + "/images")
            return path, path + "/images"
        except OSError, e:
            # Folder already exists
            logging.error('Could not create folder ' + str(e))
            return path, path + "/images"
        except Exception, e:
            logging.error('Could not create folder: ' + str(path) + ', aborting',
                          exc_info=sys.exc_info())
            return None, None
    
    ### check first if part is already within the training set
    def fitness_function(self, part):
        ##this bit traverses the particle set and checks if it has already been evaluated. 
        if self.surrogate_model.contains_training_instance(part):
        
            code, fitness = self.surrogate_model.get_training_instance(part)
            cost = self.cost_model.get_training_instance(part)
            if (fitness is None) or (code is None):
                fitness = array([self.fitness.worst_value])
            return fitness, code, cost
        self.increment_counter('fit')
        
        try:
            results, state = self.fitness.fitnessFunc(part, self.get_fitness_state())
            self.set_fitness_state(state)
        except Exception,e:          
            logging.info(str(e))
            results = self.fitness.fitnessFunc(part) ## fitness function doesnt have state
        fitness = results[0]
        code = results[1]
        addReturn = results[2]
        logging.info("Evaled " + str(part) + " fitness:" + str(fitness) + " code:" + str(code))
        try: ## not all fitness functions return benchmark exectuion cost
            cost = results[3][0]
        except:
            cost = 1.0 ## just keep it constant for all points
        self.set_counter_dictionary("cost", self.get_counter_dictionary("cost") + cost)
        self.surrogate_model.add_training_instance(part, code, fitness, addReturn)
        self.cost_model.add_training_instance(part, cost)
        self.set_retrain_model(True)
        
        if code[0] == 0:
            self.set_terminating_condition(fitness) 
            return fitness, code, cost
        else:
            return array([self.fitness.worst_value]), code, cost
            
    def fitness_function_mo(self, part):
        ##this bit traverses the particle set and checks if it has already been evaluated. 

        for i in range(len(self.surrogate_model)):
                if self.surrogate_model[i].contains_training_instance(part):
                    code, fitness = self.surrogate_model[i].get_training_instance(part)
                    cost = self.cost_model.get_training_instance(part)
                    if (fitness is None) or (code is None):
                        fitness = array([self.fitness.worst_value])
                    return fitness, code, cost
                self.increment_counter('fit')
                
        try:
            results, state = self.fitness.MOFitnessFunc(part, self.get_fitness_state())
            self.set_fitness_state(state)
        except Exception,e:
            logging.info(str(e))
            results = self.fitness.MOFitnessFunc(part) ## fitness function doesnt have state
        fitness = results[0]
        code = results[1]
        addReturn = results[2]
        logging.info("Evaled " + str(part) + " fitness:" + str(fitness) + " code:" + str(code))
        try: ## not all fitness functions return benchmark exectuion cost
            cost = results[3][0]
        except:
            cost = 1.0 ## just keep it constant for all points
        self.set_counter_dictionary("cost", self.get_counter_dictionary("cost") + cost)
        
        for i in range(self.fitness.objectives):
                self.surrogate_model[i].add_training_instance(part, code, (fitness[i],), addReturn)
                self.cost_model.add_training_instance(part, cost)
                self.set_retrain_model(True)
                
        if code[0] == 0:
            self.set_terminating_condition(fitness) 
            return fitness, code, cost
        else:
            return self.fitness.worst_value, code, cost
            
    ## indicator for the controller and viewer that the state has changed. 
    def view_update(self, visualize=False):
        self.current_time = datetime.now()
        diff = self.current_time - self.previous_time
        self.previous_time = self.current_time
        self.state_dictionary['timer_dict']['Running_Time'] = self.state_dictionary['timer_dict']['Running_Time'] + diff.seconds
        self.controller.view_update(trial=self, run=None, visualize=visualize) ##
#        self.controller.view_update(trial=self, run=self.my_run, visualize=False) ## update run state

    def increment_counter(self, counter):
        self.set_counter_dictionary(counter, self.get_counter_dictionary(counter) + 1)
                
    def save(self):
        try:
            trial_file = str(self.get_results_folder()) + '/' +  str(self.get_counter_dictionary('g')) + '.txt'
            dict = self.state_dictionary
            for i in range(self.fitness.objectives):
                    surrogate_model_state_dict = self.surrogate_model[i].get_state_dictionary()
                    dict_name = 'surrogate_model_state_dict_' + str(i)
                    dict[dict_name] = surrogate_model_state_dict
                    cost_model_state_dict = self.cost_model.get_state_dictionary()
                    
                    dict_name = 'cost_model_state_dict_' + str(i)
                    dict[dict_name] = cost_model_state_dict
            with io.open(trial_file, 'wb') as outfile:
                pickle.dump(dict, outfile)  
                if self.kill:
                    sys.exit(0)
        except Exception, e:
            logging.error(str(e))
            if self.kill:
                sys.exit(0)
            return False
            
    ## by default find the latest generation
    def load(self, generation = None):
        try:
            if generation is None:
                # Figure out what the last generation before crash was
                found = False
                for filename in reversed(os.listdir(self.get_results_folder())):
                    match = re.search(r'^(\d+)\.txt', filename)
                    if match:
                        # Found the last generation
                        generation = int(match.group(1))
                        found = True
                        break

                if not found:
                    return False
                    
            generation_file = str(generation)
            trial_file = str(self.get_results_folder()) + '/' + str(generation_file) + '.txt'
            
            with open(trial_file, 'rb') as outfile:
                dict = pickle.load(outfile)
            self.set_state_dictionary(dict)
            self.state_dictionary["generate"] = False
            self.kill = False
            for i in range(self.fitness.objectives):
                dict_name = 'surrogate_model_state_dict_' + str(i)
                self.surrogate_model[i].set_state_dictionary(dict[dict_name])
                dict_name = 'cost_model_state_dict_' + str(i)
                self.cost_model.set_state_dictionary(dict[dict_name])
            self.previous_time = datetime.now()
            logging.info("Loaded Trial")
            return True
        except Exception, e:
            logging.error("Loading error" + str(e))
            return False
        
    def exit(self):
        self.set_status('Finished')
        self.my_run.trial_notify(self)
        self.view_update(visualize = False)
        sys.exit(0)
        
    #######################
    ## Abstract Methods  ##
    #######################
        
    def initialise(self):
        raise NotImplementedError('Trial is an abstract class, '
                                   'this should not be called.')
    
    def run_initialize(self):
        raise NotImplementedError('Trial is an abstract class, '
                                  'this should not be called.')
                                  
    ## main computation loop goes here
    def run(self):
        raise NotImplementedError('Trial is an abstract class, '
                                   'this should not be called.')
                                   
    def snapshot(self):
        raise NotImplementedError('Trial is an abstract class, '
                                  'this should not be called.')
                                  
    def get_predicted_time(self):
        raise NotImplementedError('Trial is an abstract class, '
                                  'this should not be called.')

    def get_surrogate_model(self): ## returns a copy of the model... quite important not to return the model itself as ll might get F up
        raise NotImplementedError('Trial is an abstract class, '
                                  'this should not be called.')
        
    def get_cost_model(self): ## returns a copy of the model... quite important not to return the model itself as ll might get F up
        raise NotImplementedError('Trial is an abstract class, '
                                  'this should not be called.')
                                  
    #######################
    ### GET/SET METHODS ###
    #######################
        
    def get_run(self):
        return self.my_run 
        
    def get_fitness_state(self):
        return self.state_dictionary['fitness_state']
        
    def set_fitness_state(self, state):
        self.state_dictionary['fitness_state'] = state
        
    def add_train_surrogate_model_time(self, diff):
        self.state_dictionary['timer_dict']['Model Training Time'] = self.state_dictionary['timer_dict']['Model Training Time'] + diff.seconds
        
    def add_predict_surrogate_model_time(self, diff):
        self.state_dictionary['timer_dict']['Model Predict Time'] = self.state_dictionary['timer_dict']['Model Predict Time'] + diff.seconds
        
    def get_train_surrogate_model_time(self):
        return timedelta(seconds = self.state_dictionary['timer_dict']['Model Training Time'])
        
    def get_predict_surrogate_model_time(self):
        return timedelta(seconds = self.state_dictionary['timer_dict']['Model Predict Time'])
        
    def add_train_cost_model_time(self, diff):
        self.state_dictionary['timer_dict']['Cost Model Training Time'] = self.state_dictionary['timer_dict']['Cost Model Training Time'] + diff.seconds
        
    def add_predict_cost_model_time(self, diff):
        self.state_dictionary['timer_dict']['Cost Model Predict Time'] = self.state_dictionary['timer_dict']['Cost Model Predict Time'] + diff.seconds
        
    def get_train_cost_model_time(self):
        return timedelta(seconds = self.state_dictionary['timer_dict']['Cost Model Training Time'])
        
    def get_predict_cost_model_time(self):
        return timedelta(seconds = self.state_dictionary['timer_dict']['Cost Model Predict Time'])
        
    def get_running_time(self):
        return timedelta(seconds=self.state_dictionary['timer_dict']['Running_Time'])
    
    def get_main_counter_iterator(self):
        return range(0, self.get_counter_dictionary(self.get_main_counter_name()) + 1)
        
    def get_main_counter(self):
        return self.get_counter_dictionary(self.get_main_counter_name())
    
    def get_main_counter_name(self):
        return self.state_dictionary["main_counter_name"]
    
    def set_main_counter_name(self, name):
        self.state_dictionary["main_counter_name"] = name 
    
    def get_state_dictionary(self):
        return self.state_dictionary
    
    def set_state_dictionary(self, new_dictionary):
        self.state_dictionary = new_dictionary 
        
    def get_fitness(self):
        return self.fitness
        
    def get_best(self):
        return self.state_dictionary['best']
        
    def set_best(self, new_best):
        try:
            logging.info('Old global best ' + str(self.get_best()) + ' fitness:'+ str(self.get_best().fitness.values))
        except:
            pass ## there was no best
            
        self.state_dictionary['best'] = new_best
        logging.info('New global best found' + str(new_best) + ' fitness:'+ str(new_best.fitness.values))
        
    def set_best_mo(self, new_best):
        self.state_dictionary['best'] = new_best
    def get_counter_dictionary(self, counter):
        return self.state_dictionary['counter_dictionary'][counter]
        
    def set_counter_dictionary(self, counter, value):
        self.state_dictionary['counter_dictionary'][counter] = value
        
    ## should really be deep copy or something...
    ## we dont want it to be mutable outside
    def get_configuration(self):
        return self.configuration
        
    def set_start_time(self, start_time):
        self.state_dictionary['start_time'] = start_time

    def get_start_time(self):
        return self.state_dictionary['start_time']
        
    def set_surrogate_model(self, new_model):
        self.surrogate_model = new_model
        
    def set_waiting(self):
        self.set_status("Waiting")
        self.set_wait(False)
    
    def set_running(self):
        self.set_status("Running")
        self.set_wait(False)
    
    def set_paused(self):
        self.set_status("Paused")
        self.set_wait(True)
        
    def set_wait(self, new_wait): 
        self.state_dictionary["wait"] = new_wait
        
    def get_wait(self):
        return self.state_dictionary["wait"]

    def get_status(self):
        return self.state_dictionary["status"]
        
    def set_status(self, status):
        self.state_dictionary["status"] = status
        self.view_update()
        
    def set_retrain_model(self, status):
        self.state_dictionary["retrain_model"] = status

    def get_retrain_model(self):
        return self.state_dictionary["retrain_model"]
        
    def get_model_failed(self):
        return self.state_dictionary["model_failed"]
    
    def set_model_failed(self, state):
        self.state_dictionary["model_failed"] = state
    
    def get_run_results_folder_path(self):
        return self.state_dictionary["run_results_folder_path"]
        
    def get_trial_no(self):
        return self.state_dictionary["trial_no"]
    
    def get_terminating_condition(self):
        return self.state_dictionary["terminating_condition"]
    
    def set_terminating_condition(self, fitness):
        self.state_dictionary["terminating_condition"] = self.fitness.termCond(fitness[0])
        
    def get_name(self):
        return self.state_dictionary["name"]
        
    def get_results_folder(self):
        return str(self.get_run_results_folder_path()) + '/trial-' + str(self.get_trial_no()) #self.state_dictionary['results_folder']
        
    def set_images_folder(self, folder):
        self.state_dictionary['images_folder'] = folder

    def get_images_folder(self):
        return self.get_results_folder() + "/images"
        
    def get_trial_type(self):
        return self.state_dictionary["trial_type"]
    def get_design_space(self):
        return self.fitness.designSpace
        
    def get_best_fitness_array(self):
        return self.state_dictionary['best_fitness_array']
        
    def get_generations_array(self):
        return self.state_dictionary['generations_array']
        
    def set_at_least_one_in_valid_region(self, state):
        if self.state_dictionary.has_key('at_least_one_in_valid_region'):
            self.state_dictionary['at_least_one_in_valid_region'] = state
        else:
            self.state_dictionary['at_least_one_in_valid_region'] = False

    def get_at_least_one_in_valid_region(self):
        return self.state_dictionary['at_least_one_in_valid_region']

class MonteCarloTrial(Trial):

    def initialise(self):
        """
        Initialises the trial and returns True if everything went OK,
        False otherwise.
        """
        self.run_initialize()
        self.cost_model = ProperCostModel(self.configuration, self.controller, self.fitness)
        self.state_dictionary['best'] = None
        self.state_dictionary['model_failed'] = False
        self.state_dictionary['best_fitness_array'] = []
        self.state_dictionary['generations_array'] = []
        self.set_main_counter_name("i")
        self.set_counter_dictionary("i",0)

        ### Intial design space Latin hyper cube based sampling
        self.set_at_least_one_in_valid_region(False)
                
        design_space = self.get_design_space()
        D = len(design_space)
        F = D * 10
        latin_hypercube_samples = lhs.lhs(scipy_uniform,[0,1],(F,D))
        max_bounds = [d["max"] for d in design_space]
        latin_hypercube_samples = latin_hypercube_samples * max_bounds
        min_bounds = [d["min"] for d in design_space]
        latin_hypercube_samples = latin_hypercube_samples + min_bounds
        for part in latin_hypercube_samples:
            part = self.toolbox.filter_particle(self.create_particle(part))
            part.fitness.values, code, cost = self.toolbox.evaluate(part)
            self.set_at_least_one_in_valid_region((code == 0) or self.get_at_least_one_in_valid_region())
            self.update_best(part)
        F = 1 ## we got to sample the design space randomly and hope to find a valid region of space...
        while (not self.get_at_least_one_in_valid_region()):
            if self.get_configuration().eval_correct:
                part = self.filter_particle(self.create_particle(part))
                part.fitness.values, code, cost = self.toolbox.evaluate(part)
                self.set_at_least_one_in_valid_region((part.code == 0) or self.get_at_least_one_in_valid_region())
                self.update_best(part)
        self.state_dictionary["fresh_run"] = True
                
        ### generate folders
        results_folder, images_folder = self.create_results_folder()
        if not results_folder or not images_folder:
            # Results folder could not be created
            logging.error('Results and images folders cound not be created, terminating.')
            return False
        return True
    
    def run_initialize(self):
        logging.info("Initialize MonteCarloTrial no:" + str(self.get_trial_no()))
        design_space = self.get_design_space()
        try:
            eval('creator.Particle' + str(self.my_run.get_name()))
            logging.debug("Particle class for this run already exists")
        except AttributeError:
            creator.create('FitnessMax' + str(self.my_run.get_name()), base.Fitness, weights=(1.0,))
            ### we got to add specific names, otherwise the classes are going to be visible for all
            ### modules which use deap...
            
            creator.create(str('Particle' + self.my_run.get_name()), list, fitness=eval('creator.FitnessMax' + str(self.my_run.get_name())),
                           pmin=[dimSetting['max'] for dimSetting in design_space],
                           pmax=[dimSetting['min'] for dimSetting in design_space])
        self.toolbox = copy(base.Toolbox())
        self.toolbox.register('particle', self.generate, designSpace=design_space)
        self.toolbox.register('filter_particle', self.filter_particle)
        self.toolbox.register('evaluate', self.fitness_function)
        self.new_best=False
                                  
    ## main computation loop goes here
    def run(self):
        self.state_dictionary['generate'] = True
        logging.info(str(self.get_name()) + ' started')
        logging.info('Trial prepared... executing')
        
        if self.state_dictionary["fresh_run"]: ## we need this as we only want to do it for initial generation because the view
            ## the problem is that we cannot
            self.train_surrogate_model()
            self.train_cost_model()
            self.view_update(visualize = True)
            self.state_dictionary["fresh_run"] = False
            self.save()
        
        while self.get_counter_dictionary('fit') < self.get_configuration().max_iter + 1:
            
            logging.info('[' + str(self.get_name()) + '] Iteration ' + str(self.get_counter_dictionary('i')))
            logging.info('[' + str(self.get_name()) + '] Fitness ' + str(self.get_counter_dictionary('fit')))
            
            # termination condition - we put it here so that when the trial is reloaded
            # it wont run if the run has terminated already
            if self.get_terminating_condition(): 
                logging.info('Terminating condition reached...')
                break
            sample_x_times = 1
            budget = 50000.0 ## revise budget
            ## add proper copy
            best_action, best_action_function = self.farsee(self.get_surrogate_model(), self.get_cost_model(), budget, sample_x_times)
            best_x = best_action_function(self.get_surrogate_model())
            best_x = self.toolbox.filter_particle(self.create_particle(best_x))
            best_x.fitness.values, code, cost = self.toolbox.evaluate(best_x)
            self.update_best(best_x)
            
            if self.training_set_updated():
                self.train_surrogate_model()
                self.train_cost_model()
            self.view_update(visualize = True)
    #######################
    ### GET/SET METHODS ###
    #######################
    
    def generate(self,  designSpace):
        particle = [uniform(dimSetting['min'], dimSetting['max'])
                    for dimSetting
                    in designSpace]
        particle = self.create_particle(particle)
        return particle
    
    def get_predicted_time(self):
        raise NotImplementedError('Trial is an abstract class, '
                                  'this should not be called.')
    
    def get_cost_model(self): ## returns a copy of the model... quite important not to return the model itself as ll might get F up
        model = ProperCostModel(self.get_configuration(), self.controller, self.fitness)
        model.set_state_dictionary(self.cost_model.get_state_dictionary())
        return model
                                  
    def snapshot(self):
        fitness = self.fitness
        best_fitness_array = copy(self.get_best_fitness_array())
        generations_array = copy(self.get_generations_array())
        results_folder = copy(self.get_results_folder())
        images_folder = copy(self.get_images_folder())
        counter = copy(self.get_counter_dictionary('i'))
        name = self.get_name()
        return_dictionary = {
            'fitness': fitness,
            'best_fitness_array': best_fitness_array,
            'generations_array': generations_array,
            'configuration_folder_path':self.configuration.configuration_folder_path,
            'run_folders_path':self.configuration.results_folder_path,
            'results_folder': results_folder,
            'images_folder': images_folder,
            'counter': counter,
            'counter_dict':  self.state_dictionary['counter_dictionary'] ,
            'timer_dict':  self.state_dictionary['timer_dict'] ,
            'name': name,
            'fitness_state': self.get_fitness_state(),
            'run_name': self.my_run.get_name(),
            'classifier': self.get_classifier(), ## return a copy! 
            'regressor': self.get_regressor(), ## return a copy!
            'cost_model': self.get_cost_model(), ## return a copy!
            'generate' : self.state_dictionary['generate'],
            'max_iter' : self.configuration.max_iter,
            'max_fitness' : self.configuration.max_fitness,
            'best': {"data":self.get_best()}
        }
        return return_dictionary
        
    def save(self):
        try:
            trial_file = str(self.get_results_folder()) + '/' +  str(self.get_counter_dictionary('i')) + '.txt'
            dict = self.state_dictionary
            surrogate_model_state_dict = self.surrogate_model.get_state_dictionary()
            dict['surrogate_model_state_dict'] = surrogate_model_state_dict
            cost_model_state_dict = self.cost_model.get_state_dictionary()
            dict['cost_model_state_dict'] = cost_model_state_dict
            with io.open(trial_file, 'wb') as outfile:
                pickle.dump(dict, outfile)  
                if self.kill:
                    sys.exit(0)
        except Exception, e:
            logging.error(str(e))
            if self.kill:
                sys.exit(0)
            return False
            
    ## by default find the latest iteration
    def load(self, iteration = None):
        try:
            if iteration is None:
                # Figure out what the last iteration before crash was
                found = False
                for filename in reversed(os.listdir(self.get_results_folder())):
                    match = re.search(r'^(\d+)\.txt', filename)
                    if match:
                        # Found the last iteration
                        iteration = int(match.group(1))
                        found = True
                        break

                if not found:
                    return False
                    
            iteration_file = str(iteration)
            trial_file = str(self.get_results_folder()) + '/' + str(iteration_file) + '.txt'
            
            with open(trial_file, 'rb') as outfile:
                dict = pickle.load(outfile)
            self.set_state_dictionary(dict)
            self.state_dictionary["generate"] = False
            self.kill = False
            self.surrogate_model.set_state_dictionary(dict['surrogate_model_state_dict'])
            self.cost_model.set_state_dictionary(dict['cost_model_state_dict'])
            self.previous_time = datetime.now()
            logging.info("Loaded Trial")
            return True
        except Exception, e:
            logging.error("Loading error" + str(e))
            return False
        
    ## actions and wrapper methods
        
    def all_actions(self):
        return {#"max_ei": self.max_ei
                "max_ei_cost": self.max_ei_cost
                }
        
    def max_ei(self, surrogate_model):
        hypercube = self.hypercube()
        particle = surrogate_model.max_ei(designSpace=self.get_design_space(), hypercube = hypercube) ## NOT SELF
        particle = self.toolbox.filter_particle(self.create_particle(particle))
        return particle
        
    def max_ei_cost(self, surrogate_model):
        hypercube = self.hypercube()
        particle = surrogate_model.max_ei_cost(designSpace=self.get_design_space(), hypercube = hypercube, cost_func = self.predict_cost) ## NOT SELF
        particle = self.toolbox.filter_particle(self.create_particle(particle))
        return particle
        
    def keywithmaxval(self, d):
        """ a) create a list of the dict's keys and values; 
            b) return the key with the max value"""  
        v=list(d.values())
        k=list(d.keys())
        return k[v.index(max(v))]
     
    def keywithminval(self, d):
        """ a) create a list of the dict's keys and values; 
            b) return the key with the max value"""  
        v=list(d.values())
        k=list(d.keys())
        return k[v.index(min(v))]
        
    def farsee(self, surrogate_model, cost_model, budget, sample_x_times):
        rewards = dict([(action, 0.0) for action in self.all_actions().keys()])
        for i in range(sample_x_times):
            #best_action, reward = self.reward_function(surrogate_model, cost_model, budget)
            best_action, reward = self.reward_function_v1(surrogate_model, cost_model, 0.0,0.0)
            rewards[best_action] += reward
        if self.configuration.goal == "max":
            best_action = self.keywithmaxval(rewards)
            return best_action, self.all_actions()[best_action]
        else:
            best_action = self.keywithminval(rewards)
            return best_action, self.all_actions()[best_action]
            
    ## finds the best action that offers best result within reaching budget
    ## might have to be adjusted... add a decay element? more then 3-4 samples away 
    ## is just pointless
    ## to evalaute the tree up to for example 50% of the budget simply change what you feed in to 10..20..50 or so %
    def reward_function(self, surrogate_model, cost_model, budget):
        reward = {}
        for action, action_function in self.all_actions().items():
            logging.info("Looking into future.. action:" + action)
            surrogate_model_copy = surrogate_model.get_copy()
            cost_model_copy = cost_model.get_copy()
            best_x = action_function(surrogate_model_copy)
            code, MU, S2, EI, P = surrogate_model_copy.predict([best_x])
            fitness = P[0] ## sampled from the distribution
            cost = cost_model_copy.predict([best_x])[0]
            surrogate_model_copy.add_training_instance(best_x, code, fitness, [0.0]) ## always add
            if cost <= 0.0: ## for some of the regressors 
                logging.debug("detected negative cost... probably small value")
            cost_model_copy.add_training_instance(best_x, cost)
            #surrogate_model_copy.train(self.hypercube())
            #cost_model_copy.train()
            my_budget = budget - cost
            logging.info("Budget left: " + str(my_budget))
            if my_budget > 0.0:
                best_action, reward[action] = self.reward_function(surrogate_model_copy, cost_model_copy, my_budget)
                try:
                    if self.configuration.goal == "max":
                        reward[action] = max(fitness, reward[action])
                    else:
                        reward[action] = min(fitness, reward[action])
                except ValueError:
                    logging.info("kurwo..." + str(fitness) + " " + str(reward) + " " + str(reward[action]))
            else:
                reward[action] = fitness
                
        if self.configuration.goal == "max":
            best_action = self.keywithmaxval(reward)
            return best_action, reward[best_action]
        else:
            best_action = self.keywithminval(reward)
            return best_action, reward[best_action]
    
    ## favors fast convergence (what about local minimias?
    ## It cuts of the tree at one point and reports the budget used so far..
    ## the reward is calculated based on the mean best result / budget used value 
    ## obviously depends on whether we faces min/max problems
    def reward_function_v1(self, surrogate_model, cost_model, horizont, budget_used):
        reward = {}
        logging.info("Investigating horizont level: " + str(horizont))
        for action, action_function in self.all_actions().items():
            surrogate_model_copy = surrogate_model.get_copy()
            cost_model_copy = cost_model.get_copy()
            best_x = action_function(surrogate_model_copy)
            code, MU, S2, EI, P = surrogate_model_copy.predict([best_x])
            fitness = P[0] ## sampled from the distribution
            cost = cost_model_copy.predict([best_x])[0]
            surrogate_model_copy.add_training_instance(best_x, code, fitness, [0.0]) ## always add
            cost_model_copy.add_training_instance(best_x, cost)
            new_budget_used = cost + budget_used
            if horizont > 0.0:
                best_action, reward[action] = self.reward_function_v1(surrogate_model_copy, cost_model_copy, horizont-1.0, new_budget_used)
                if self.configuration.goal == "max":
                    reward[action] = max(fitness, reward[action])
                else:
                    reward[action] = min(fitness, reward[action])
            else:
                if self.configuration.goal == "max":
                    reward[action] = fitness / new_budget_used
                else:
                    reward[action] = fitness * new_budget_used
                    
        if self.configuration.goal == "max":
            best_action = self.keywithmaxval(reward)
            return best_action, reward[best_action]
        else:
            best_action = self.keywithminval(reward)
            return best_action, reward[best_action]
        
    def filter_particle(self, p):#
        design_space = self.get_design_space()  
        pmin = [dimSetting['min'] for dimSetting in design_space]
        pmax = [dimSetting['max'] for dimSetting in design_space]
        for i, val in enumerate(p):
            #dithering
            if design_space[i]['type'] == 'discrete':
                if uniform(0.0, 1.0) < (p[i] - floor(p[i])):
                    p[i] = ceil(p[i])  # + designSpace[i]['step']
                else:
                    p[i] = floor(p[i])

            #dont allow particles to take the same value
            p[i] = minimum(pmax[i], p[i])
            p[i] = maximum(pmin[i], p[i])
        return p
    ## return hypercube spanning the whole of the design space
    def hypercube(self): ##rectangle rather
        design_space = self.get_design_space()  
        max_diag = [dimSetting['max'] for dimSetting in design_space]
        min_diag = [dimSetting['min'] for dimSetting in design_space]
        #find maximum
        return [max_diag,min_diag]
        
    def update_best(self, part):
        if not self.get_best() or self.is_better(part.fitness, self.get_best().fitness):
            best = self.create_particle(part)
            best.fitness.values = part.fitness.values
            self.set_best(best)
            
    def create_particle(self, particle):
        return eval('creator.Particle' + self.my_run.get_name())(particle)
            
    def predict_cost(self, particle):
        try:
            return self.cost_model.predict(particle)
        except Exception,e:
            logging.debug("Cost model is still not avaiable: " + str(e))
            "a" + 1 
            return 1.0 ## model has not been created yet
            


class PSOTrial(Trial):

    #######################
    ## Abstract Methods  ##
    #######################
    
    def initialise(self):
        """
        Initialises the trial and returns True if everything went OK,
        False otherwise.
        """
        self.run_initialize()
        self.state_dictionary['best'] = None
        self.state_dictionary['fitness_evaluated'] = False
        self.state_dictionary['model_failed'] = False
        self.state_dictionary['new_best_over_iteration'] = False
        self.state_dictionary['population'] = None
        self.state_dictionary['best_fitness_array'] = []
        self.state_dictionary['generations_array'] = []
        self.set_main_counter_name("g")
        self.set_counter_dictionary("g",0)
        self.initialize_population()    
        results_folder, images_folder = self.create_results_folder()
        if not results_folder or not images_folder:
            # Results folder could not be created
            logging.error('Results and images folders cound not be created, terminating.')
            return False
        
        return True
        
    def run_initialize(self):
        logging.info("Initialize PSOTrial no:" + str(self.get_trial_no()))
        self.cost_model = DummyCostModel(self.configuration, self.controller, self.fitness)
        design_space = self.fitness.designSpace
        self.toolbox = copy(base.Toolbox())
        self.smin = [-1.0 * self.get_configuration().max_speed *
                (dimSetting['max'] - dimSetting['min'])
                for dimSetting in design_space]
        self.smax = [self.get_configuration().max_speed *
                (dimSetting['max'] - dimSetting['min'])
                for dimSetting in design_space]

        try:
            eval('creator.Particle' + str(self.my_run.get_name()))
            logging.debug("Particle class for this run already exists")
        except AttributeError:
            creator.create('FitnessMax' + str(self.my_run.get_name()), base.Fitness, weights=(1.0,))
            ### we got to add specific names, otherwise the classes are going to be visible for all
            ### modules which use deap...
            
            creator.create(str('Particle' + self.my_run.get_name()), list, fitness=eval('creator.FitnessMax' + str(self.my_run.get_name())),
                           smin=self.smin, smax=self.smax,
                           speed=[uniform(smi, sma) for sma, smi in zip(self.smax,
                                                                        self.smin)],
                           pmin=[dimSetting['max'] for dimSetting in design_space],
                           pmax=[dimSetting['min'] for dimSetting in design_space],
                           model=False, best=None, code=None)

        self.toolbox.register('particle', self.generate, designSpace=design_space)
        self.toolbox.register('filter_particles', self.filterParticles,
                              designSpace=design_space)
        self.toolbox.register('filter_particle', self.filterParticle,
                              designSpace=design_space)
        self.toolbox.register('population', tools.initRepeat,
                              list, self.toolbox.particle)
        self.toolbox.register('update', self.updateParticle, 
                              conf=self.get_configuration(),
                              designSpace=design_space)
        self.toolbox.register('evaluate', self.fitness_function)
        self.new_best=False
        
    def run(self):
        self.state_dictionary['generate'] = True
        
        logging.info(str(self.get_name()) + ' started')
        logging.info('Trial prepared... executing')
        self.save() ## training might take a bit...
        # Initialise termination check
        
        self.check = False
        ## we do this not to retrain model twice during the first iteration. If we ommit
        ## this bit of code the first view_update wont have a model aviable.
        reevalute = False
        if self.state_dictionary["fresh_run"]: ## we need this as we only want to do it for initial generation because the view
            ## the problem is that we cannot
            self.train_surrogate_model(self.get_population())
            self.train_cost_model(self.get_population())
            self.view_update(visualize = True)
            self.state_dictionary["fresh_run"] = False
            self.save()
            
        while self.get_counter_dictionary('g') < self.get_configuration().max_iter + 1:
            
            logging.info('[' + str(self.get_name()) + '] Generation ' + str(self.get_counter_dictionary('g')))
            logging.info('[' + str(self.get_name()) + '] Fitness ' + str(self.get_counter_dictionary('fit')))

            # termination condition - we put it here so that when the trial is reloaded
            # it wont run if the run has terminated already
        # see this
            if self.get_terminating_condition(): 
                logging.info('Terminating condition reached...')
                break
            
            # Roll population
            first_pop = self.get_population().pop(0)
            self.get_population().append(first_pop)
            
            if self.get_counter_dictionary('fit') > self.get_configuration().max_fitness:
                logging.info('Fitness counter exceeded the limit... exiting')
                break
            reevalute = False
            # Train surrogate model
            if self.training_set_updated():
                self.train_surrogate_model(self.get_population())
                self.train_cost_model(self.get_population())
                reevalute = True
            ##print self.get_population()
            code, mu, variance, ei, p = self.predict_surrogate_model(self.get_population())
            reloop = False
            if (mu is None) or (variance is None):
                logging.info("Prediction Failed")
                self.set_model_failed(True)
            else:
                logging.info("mean S2 " + str(mean(variance)))
                logging.info("max S2  " + str(max(variance)))
                logging.info("min S2  " + str(min(variance)))
                logging.info("over 0.05  " + str(min(len([v for v in variance if v > 0.05]))))
                logging.info("over 0.01  " + str(min(len([v for v in variance if v > 0.01]))))
                logging.info("mean p " + str(mean(p)))
                logging.info("max p  " + str(max(p)))
                logging.info("min p  " + str(min(p)))
                logging.info("over 0.05  " + str(min(len([v for v in p if v > 0.05]))))
                logging.info("over 0.1  " + str(min(len([v for v in p if v > 0.01]))))
                logging.info("mean ei " + str(mean(ei)))
                logging.info("max ei  " + str(max(ei)))
                logging.info("min ei  " + str(min(ei)))
                reloop = self.post_model_filter(code, mu, variance)
            ##
            if self.get_model_failed():
                logging.info('Model Failed, sampling design space')
                self.sample_design_space()
            elif reloop:
                reevalute = True
                logging.info('Evaluated some particles, will try to retrain model')
            else:#
                if reevalute:
                    self.reevalute_best()
                # Iteration of meta-heuristic
                self.meta_iterate()
                self.filter_population()
                
                #Check if perturbation is neccesary 
                if self.get_counter_dictionary('g') % self.get_configuration().M == 0:# perturb
                    self.evaluate_best()
                self.new_best = False
            # Wait until the user unpauses the trial.
            while self.get_wait():
                time.sleep(0)
            
            self.increment_main_counter()
            self.view_update(visualize = True)
        self.exit()
        
    ### returns a snapshot of the trial state
    def snapshot(self):
        fitness = self.fitness
        best_fitness_array = copy(self.get_best_fitness_array())
        generations_array = copy(self.get_generations_array())
        results_folder = copy(self.get_results_folder())
        images_folder = copy(self.get_images_folder())
        counter = copy(self.get_counter_dictionary('g'))
        name = self.get_name()
        return_dictionary = {
            'fitness': fitness,
            'best_fitness_array': best_fitness_array,
            'generations_array': generations_array,
            'configuration_folder_path':self.configuration.configuration_folder_path,
            'run_folders_path':self.configuration.results_folder_path,
            'results_folder': results_folder,
            'images_folder': images_folder,
            'counter': counter,
            'counter_dict':  self.state_dictionary['counter_dictionary'] ,
            'timer_dict':  self.state_dictionary['timer_dict'] ,
            'name': name,
            'fitness_state': self.get_fitness_state(),
            'run_name': self.my_run.get_name(),
            'classifier': self.get_surrogate_model().classifier, ## return a copy! 
            'regressor': self.get_surrogate_model().regressor, ## return a copy!
            'cost_model': self.get_cost_model(), ## return a copy!
            'meta_plot': {"particles":{'marker':"o",'data':self.get_population()}},
            'generate' : self.state_dictionary['generate'],
            'max_iter' : self.configuration.max_iter,
            'max_fitness' : self.configuration.max_fitness
        }
        return return_dictionary
        
    ####################
    ## Helper Methods ##
    ####################
    
    def checkAllParticlesEvaled(self):
        all_evaled = True
        for part in self.get_population():
            all_evaled = self.get_surrogate_model().contains_training_instance(part) and all_evaled
        
        if all_evaled: ## randomize population
            self.set_population(self.toolbox.population(self.get_configuration().population_size))
            self.toolbox.filter_particles(self.get_population())
        
    def checkCollapse(self): #TODO
        ## this method checks if the particls
        ## a) collapsed onto a single point
        ## b) collapsed onto the edge of the search space
        ## if so it reintializes them.
        minimum_diverity = 0.95 ##if over 95 collapsed reseed
        
        
        if collapsed:
            self.set_population(self.toolbox.population(self.get_configuration().population_size))
            self.toolbox.filter_particles(self.get_population())
    
    def create_particle(self, particle):
        return eval('creator.Particle' + self.my_run.get_name())(particle)
        
    
    def createUniformSpace(self, particles, designSpace):
        pointPerDimensions = 5
        valueGrid = mgrid[designSpace[0]['min']:designSpace[0]['max']:
                          complex(0, pointPerDimensions),
                          designSpace[1]['min']:designSpace[1]['max']:
                          complex(0, pointPerDimensions)]

        for i in [0, 1]:
            for j, part in enumerate(particles):
                part[i] = valueGrid[i].reshape(1, -1)[0][j]

    def filterParticles(self,  particles, designSpace):
        for particle in particles:
            self.filterParticle(particle, designSpace)
            
    def filterParticle(self, p, designSpace):
        p.pmin = [dimSetting['min'] for dimSetting in designSpace]
        p.pmax = [dimSetting['max'] for dimSetting in designSpace]

        for i, val in enumerate(p):
            #dithering
            if designSpace[i]['type'] == 'discrete':
                if uniform(0.0, 1.0) < (p[i] - floor(p[i])):
                    p[i] = ceil(p[i])  # + designSpace[i]['step']
                else:
                    p[i] = floor(p[i])

            #dont allow particles to take the same value
            p[i] = minimum(p.pmax[i], p[i])
            p[i] = maximum(p.pmin[i], p[i])

    def generate(self,  designSpace):
        particle = [uniform(dimSetting['min'], dimSetting['max'])
                    for dimSetting
                    in designSpace]
        particle = self.create_particle(particle)
        return particle

	# update the position of the particles
	# should change this part in order to change the leader election strategy
    def updateParticle(self,  part, generation, conf, designSpace):
        if conf.admode == 'fitness':
            fraction = self.fitness_counter / conf.max_fitness
        elif conf.admode == 'iter':
            fraction = generation / conf.max_iter
        else:
            raise('[updateParticle]: adjustment mode unknown.. ')

        u1 = [uniform(0, conf.phi1) for _ in range(len(part))]
        u2 = [uniform(0, conf.phi2) for _ in range(len(part))]
        
        ##########   this part particulately, leader election for every particle
        v_u1 = map(operator.mul, u1, map(operator.sub, part.best, part))
        v_u2 = map(operator.mul, u2, map(operator.sub, self.get_best(), part))
        weight = 1.0
        if conf.weight_mode == 'linear':
            weight = conf.max_weight - (conf.max_weight -
                                        conf.min_weight) * fraction
        elif conf.weight_mode == 'norm':
            weight = conf.weight
        else:
            raise('[updateParticle]: weight mode unknown.. ')
        weightVector = [weight] * len(part.speed)
        part.speed = map(operator.add,
                         map(operator.mul, part.speed, weightVector),
                         map(operator.add, v_u1, v_u2))

	# what's this mean?
        if conf.applyK is True:
            phi = array(u1) + array(u1)

            XVector = (2.0 * conf.KK) / abs(2.0 - phi -
                                            sqrt(pow(phi, 2.0) - 4.0 * phi))
            part.speed = map(operator.mul, part.speed, XVector)

	# what's the difference between these modes?
        if conf.mode == 'vp':
            for i, speed in enumerate(part.speed):
                speedCoeff = (conf.K - pow(fraction, conf.p)) * part.smax[i]
                if speed < -speedCoeff:
                    part.speed[i] = -speedCoeff
                elif speed > speedCoeff:
                    part.speed[i] = speedCoeff
                else:
                    part.speed[i] = speed
        elif conf.mode == 'norm':
            for i, speed in enumerate(part.speed):
                if speed < part.smin[i]:
                    part.speed[i] = part.smin[i]
                elif speed > part.smax[i]:
                    part.speed[i] = part.smax[i]
        elif conf.mode == 'exp':
            for i, speed in enumerate(part.speed):
                maxVel = (1 - pow(fraction, conf.exp)) * part.smax[i]
                if speed < -maxVel:
                    part.speed[i] = -maxVel
                elif speed > maxVel:
                    part.speed[i] = maxVel
        elif conf.mode == 'no':
            pass
        else:
            raise('[updateParticle]: mode unknown.. ')
        part[:] = map(operator.add, part, part.speed)

    def initialize_population(self):
        ## the while loop exists, as meta-heuristic makes no sense till we find at least one particle that is within valid region...
        
        try:
            part = self.create_particle(self.get_configuration().always_valid)
            self.toolbox.filter_particle(part)
            part.fitness.values, part.code, cost = self.fitness_function(part)
            if not self.get_best() or self.is_better(part.fitness, self.get_best().fitness):
                particle = self.create_particle(part)
                particle.fitness.values = part.fitness.values
                self.set_best(particle)
            logging.info("Always valid configuration present, evaluated")
        except:
            logging.info("Always valid configuration not-present, make sure that the valid design space is large enough so that at least one valid design is initially evalauted")
            
        self.set_at_least_one_in_valid_region(False)
        #F = copy(self.get_configuration().F)
        designSpace = self.fitness.designSpace
        D = len(designSpace)
        latin_hypercube_samples = lhs.lhs(scipy_uniform,[0,1],(self.get_configuration().population_size,D))
        max_bounds = array([d["max"] for d in designSpace])
        min_bounds = array([d["min"] for d in designSpace])
        latin_hypercube_samples = latin_hypercube_samples * (max_bounds - min_bounds)
        latin_hypercube_samples = latin_hypercube_samples + min_bounds
        population = []
        f_counter = 0
        for part in latin_hypercube_samples:
            part = self.create_particle(part)
            self.toolbox.filter_particle(part)
           # if f_counter < self.get_configuration().population_size/2.:
            part.fitness.values, part.code, cost = self.fitness_function(part)
            self.set_at_least_one_in_valid_region((part.code == 0) or self.get_at_least_one_in_valid_region())
            if not self.get_best() or self.is_better(part.fitness, self.get_best().fitness):
                particle = self.create_particle(part)
                particle.fitness.values = part.fitness.values
                self.set_best(particle)
            population.append(part)
            f_counter = f_counter + 1
        self.set_population(population)
        while (not self.get_at_least_one_in_valid_region()):
            part = self.toolbox.particle() ## a random particle
            logging.info("All particles within invalid search space.. Evaluating extra examples: " + str(part))
            part.fitness.values, part.code, cost = self.toolbox.evaluate(part)
            self.set_at_least_one_in_valid_region((part.code == 0) or self.get_at_least_one_in_valid_region())
            if not self.get_best() or self.is_better(part.fitness, self.get_best().fitness):
                particle = self.create_particle(part)
                particle.fitness.values = part.fitness.values
                self.set_best(particle)
            
        self.state_dictionary["fresh_run"] = True
        
        #what's this function do?
    def meta_iterate(self):
        #TODO - reavluate one random particle... do it.. very important!
        ##while(self.get_at_least_one_in_valid_region()):
        ##    logging.info("All particles within invalid area... have to randomly sample the design space to find one that is OK...")             
            
        #Update Bests
        logging.info("Meta Iteration")
        for part in self.get_population():
            if not part.best or self.fitness.is_better(part.fitness, part.best.fitness):
                part.best = self.create_particle(part)
                part.best.fitness.values = part.fitness.values
            if not self.get_best() or self.fitness.is_better(part.fitness, self.get_best().fitness):
                particle = self.create_particle(part)
                particle.fitness.values = part.fitness.values
                self.set_best(particle)
                self.new_best = True
                                
        #PSO
        for part in self.get_population():
            self.toolbox.update(part, self.get_counter_dictionary('g'))

    def filter_population(self):
        self.toolbox.filter_particles(self.get_population())
   
    def evaluate_best(self):        
        if self.new_best:
            self.fitness_function(self.get_best())
            logging.info('New best was found after M :' + str(self.get_best()))
        else:
            self.sample_design_space()
            '''
            perturbation = self.perturbation(radius = 100.0)                        
            logging.info('Best was already evalauted.. adding perturbation ' + str(perturbation))
            perturbed_particle = self.create_particle(self.get_best())
            code, mean, variance = self.predict_surrogate_model([perturbed_particle])
            if code is None:
                logging.debug("Code is none..watch out")
            if code is None or code[0] == 0:
                logging.info('Perturbation might be valid, evaluationg')
            for i,val in enumerate(perturbation):
                perturbed_particle[i] = perturbed_particle[i] + val       
            self.toolbox.filter_particle(perturbed_particle)
            fitness, code, cost = self.fitness_function(perturbed_particle) 
            ##check if the value is not a new best
            perturbed_particle.fitness.values = fitness
            if not self.get_best() or self.fitness.is_better(perturbed_particle.fitness, self.get_best().fitness):
                self.set_best(perturbed_particle)
            else: ## why do we do this? 
                if code[0] != 0:
                    logging.info('Best is within the invalid area ' + str(code[0]) + ', sampling design space')
                    self.sample_design_space()
            '''
        
    def increment_main_counter(self):
        self.get_best_fitness_array().append(self.get_best().fitness.values[0])
        self.get_generations_array().append(self.get_counter_dictionary(self.get_main_counter_name()))
        self.save()
        self.increment_counter(self.get_main_counter_name())

    def sample_design_space(self):
        #logging.info('Evaluating best perturbation')
        particle = self.surrogate_model.max_ei(designSpace=self.fitness.designSpace, hypercube = self.hypercube())
        if particle is None:
            logging.info("Local sampling has failed, probably all of the particles are within invalid region")
            particle = self.surrogate_model.max_ei(designSpace=self.fitness.designSpace)
            if particle is None:
                particle = self.toolbox.particle()
                logging.info("Global sampling failed as well.. Evaluating a random particle"  + str(particle))
        particle = self.create_particle(particle)   
        self.toolbox.filter_particle(particle)
        particle.fitness.values, code, cost = self.fitness_function(particle) 
        if not self.get_best() or self.is_better(particle.fitness, self.get_best().fitness):
            self.set_best(particle)
        
    ## not used currently
    # def get_perturbation_dist(self):
        # [max_diag, min_diag] = self.get_dist()
        # d = (max_diag - min_diag)/2.0
        # if best:
            # max_diag = best + d
            # for i,dd in enumerate(self.fitness.designSpace):
                # max_diag[i] = minimum(max_diag[i],dd["max"])
            # min_diag = best - d
            # for i,dd in enumerate(self.fitness.designSpace):
                # min_diag[i] = maximum(min_diag[i],dd["min"])
            # return [max_diag,min_diag]
        # else:
            # return getHypercube(pop)
            
     ## not used currently
    def get_dist(self):
        if best:
            distances = sqrt(sum(pow((self.surrogate.best),2),axis=1))  # TODO
            order_according_to_manhatan = argsort(distances)
            closest_array = [gpTrainingSet[index] for index in order_according_to_manhatan[0:conf.nClosest]]
        ###        
        ## limit to hypercube around the points
        #find maximum
        #print "[getDist] closestArray ",closestArray
        max_diag = deepcopy(closestArray[0])
        for part in closest_array:
            max_diag = maximum(part, max_diag)
        ###find minimum vectors
        min_diag = deepcopy(closest_array[0])
        for part in closest_array:
            min_diag = minimum(part, min_diag)
        return [max_diag, min_diag]
        
    def hypercube(self):
        #find maximum
        max_diag = deepcopy(self.get_population()[0])
        for part in self.get_population():
            max_diag = maximum(part,max_diag) 
        ###find minimum vectors
        min_diag = deepcopy(self.get_population()[0])
        for part in self.get_population():
            min_diag = minimum(part,min_diag)
            
        ## we always ensure that the hypercube allows particles to maintain velocity components in all directions
        
        for i,dd in enumerate(max_diag):
            if self.fitness.designSpace[i]["type"] == "discrete":
                max_diag[i] = minimum(dd + self.fitness.designSpace[i]["step"],self.fitness.designSpace[i]["max"])
            elif self.fitness.designSpace[i]["type"] == "continuous":
                small_fraction = ((self.fitness.designSpace[i]["max"] - self.fitness.designSpace[i]["min"]) / 100.)
                max_diag[i] = minimum(dd + small_fraction, self.fitness.designSpace[i]["max"])
                
        for i,dd in enumerate(min_diag):
            if self.fitness.designSpace[i]["type"] == "discrete":
                min_diag[i] = maximum(dd - self.fitness.designSpace[i]["step"],self.fitness.designSpace[i]["min"])
            elif self.fitness.designSpace[i]["type"] == "continuous":
                small_fraction = ((self.fitness.designSpace[i]["max"] - self.fitness.designSpace[i]["min"]) / 100.)
                min_diag[i] = maximum(dd - small_fraction, self.fitness.designSpace[i]["min"])
        logging.info("hypecube: " + str([max_diag,min_diag]))
        return [max_diag,min_diag]
        
    def perturbation(self, radius = 10.0):
        [max_diag,min_diag] = self.hypercube()
        d = (max_diag - min_diag)/radius
        for i,dd in enumerate(d):
            if self.fitness.designSpace[i]["type"] == "discrete":
                d[i] = maximum(dd,self.fitness.designSpace[i]["step"])
            elif self.fitness.designSpace[i]["type"] == "continuous":
                small_fraction = ((self.fitness.designSpace[i]["max"] - self.fitness.designSpace[i]["min"]) / 100.)
                d[i] = maximum(dd,small_fraction)
        dimensions = len(self.fitness.designSpace)
        pertubation =  multiply(((rand(1,dimensions)-0.5)*2.0),d)[0] #TODO add the dimensions
        return pertubation
        
        ###check if between two calls to this functions any fitness functions have been evaluted, so that the models have to be retrained
    def training_set_updated(self):
        retrain_model_temp = self.get_retrain_model()
        self.set_retrain_model(False)
        return retrain_model_temp
    
    ### TODO - its just copy and pasted ciode now..w could rewrite it realyl
    def post_model_filter(self, code, mean, variance):
        eval_counter = 1
        self.set_model_failed(not (False in [self.get_configuration().max_stdv < pred for pred in variance]))
        if self.get_model_failed():
            return False
        if (code is None) or (mean is None) or (variance is None):
            self.set_model_failed(False)
        else:
            for i, (p, c, m, v) in enumerate(zip(self.get_population(), code,
                                                 mean, variance)):
                if v > self.get_configuration().max_stdv and c == 0:
                    if eval_counter > self.get_configuration().max_eval:
                        logging.info("Evalauted more fitness functions per generation then max_eval")
                        #self.checkAllParticlesEvaled() ## if all the particles have been evalauted 
                        return True
                    p.fitness.values, p.code, cost = self.toolbox.evaluate(p)
                    eval_counter = eval_counter + 1
                else:
                    try:
                        if c == 0:
                            p.fitness.values = m
                        else:
                            p.fitness.values = [self.fitness.worst_value]
                    except:
                        p.fitness.values, p.code, cost = self.toolbox.evaluate(p)
            ## at least one particle has to have std smaller then max_stdv
            ## if all particles are in invalid zone
        return False
   
    def reevalute_best(self):
        bests_to_model = [p.best for p in self.get_population() if p.best] ### Elimate Nones -- in case M < Number of particles, important for initialb iteratiions
        if self.get_best():
            bests_to_model.append(self.get_best())
        if bests_to_model:
            logging.info("Reevaluating")
            code, bests_to_fitness, variance, ei, p = self.predict_surrogate_model(bests_to_model)
            if (code is None) or (bests_to_fitness is None) or (variance is None):
                logging.info("Prediction failed during reevaluation... omitting")
            else:
                for i,part in enumerate([p for p in self.get_population() if p.best]):
                    if code[i] == 0:
                        part.best.fitness.values = bests_to_fitness[i]
                    else:
                        part.best.fitness.values = [self.fitness.worst_value]
                if self.get_best():
                    best = self.get_best()
                    if code[-1] == 0:
                        best.fitness.values = bests_to_fitness[-1]
                    else:
                        best.fitness.values = [self.fitness.worst_value]
                    self.set_best(best)
    #######################
    ### GET/SET METHODS ###
    #######################
    
    def get_predicted_time(self):
        predicted_time = self.state_dictionary['total_time'] * self.get_configuration().max_iter / (self.get_counter_dictionary('g') + 1.0)
        return str(timedelta(seconds=predicted_time))
    
    def set_population(self, population):
        self.state_dictionary["population"] = population
        
    def get_population(self):
        return self.state_dictionary["population"]
        
    def get_best_fitness_array(self):
        return self.state_dictionary['best_fitness_array']
        
    def get_generations_array(self):
        return self.state_dictionary['generations_array']
        
    def set_at_least_one_in_valid_region(self, state):
        if self.state_dictionary.has_key('at_least_one_in_valid_region'):
            self.state_dictionary['at_least_one_in_valid_region'] = state
        else:
            self.state_dictionary['at_least_one_in_valid_region'] = False

    def get_at_least_one_in_valid_region(self):
        return self.state_dictionary['at_least_one_in_valid_region']
        
    def get_surrogate_model(self): ## returns a copy of the model... quite important not to return the model itself as ll might get F up
        model = ProperSurrogateModel(self.get_configuration(), self.controller, self.fitness)
        model.set_state_dictionary(self.surrogate_model.get_state_dictionary())
        return model

    def get_cost_model(self): ## returns a copy of the model... quite important not to return the model itself as ll might get F up
        model = DummyCostModel(self.get_configuration(), self.controller, self.fitness)
        model.set_state_dictionary(self.cost_model.get_state_dictionary())
        return model
        
    def get_classifier(self):
        return self.get_surrogate_model().get_classifier() 
        
    def get_regressor(self):
        return self.get_surrogate_model().get_regressor()
    
class MOPSOTrial(Trial):
    #######################
    ## Abstract Methods  ##
    #######################
    
    def initialise(self):
        """
        Initialises the trial and returns True if everything went OK,
        False otherwise.
        """
        results_folder, images_folder = self.create_results_folder()
        self.run_initialize()
        self.state_dictionary['best'] = None
        self.state_dictionary['fitness_evaluated'] = False
        self.state_dictionary['model_failed'] = False
        self.state_dictionary['new_best_over_iteration'] = False
        self.state_dictionary['population'] = None
        self.state_dictionary['best_fitness_array'] = []
        self.state_dictionary['generations_array'] = []
        self.state_dictionary['particle_speed_array'] = []
        self.set_main_counter_name('g')
        self.set_counter_dictionary('g',0)
        self.initialize_population()

        if not results_folder or not images_folder:
            # Results folder could not be created
            logging.error('Results and images folders cound not be created, terminating.')
            return False
        
        return True
        
    def run_initialize(self):
        logging.info("Initialize Multi Objective PSOTrial no:" + str(self.get_trial_no()))
        self.cost_model = DummyCostModel(self.configuration, self.controller, self.fitness)
        design_space = self.fitness.designSpace
        self.toolbox = copy(base.Toolbox())
        self.smin = [-1.0 * self.get_configuration().max_speed *
                (dimSetting['max'] - dimSetting['min'])
                for dimSetting in design_space]
        self.smax = [self.get_configuration().max_speed *
                (dimSetting['max'] - dimSetting['min'])
                for dimSetting in design_space]

        try:
            eval('creator.Particle' + str(self.my_run.get_name()))
            logging.debug("Particle class for this run already exists")
        except AttributeError:
            weight = ()
            for i in range(self.fitness.objectives):
                weight = weight + (1.0,)
            creator.create('FitnessMax' + str(self.my_run.get_name()), base.Fitness, weights= weight)
            ### we got to add specific names, otherwise the classes are going to be visible for all
            ### modules which use deap...
            
            creator.create(str('Particle' + self.my_run.get_name()), list, fitness=eval('creator.FitnessMax' + str(self.my_run.get_name())),
                           smin=self.smin, smax=self.smax,
                           speed=[uniform(smi, sma) for sma, smi in zip(self.smax,
                                                                        self.smin)],
                           pmin=[dimSetting['max'] for dimSetting in design_space],
                           pmax=[dimSetting['min'] for dimSetting in design_space],
                           model=False, best=None, code=None)

        self.toolbox.register('particle', self.generate, designSpace=design_space)
        self.toolbox.register('filter_particles', self.filterParticles,
                              designSpace=design_space)
        self.toolbox.register('filter_particle', self.filterParticle,
                              designSpace=design_space)
        self.toolbox.register('population', tools.initRepeat,
                              list, self.toolbox.particle)
        self.toolbox.register('update', self.updateParticle, 
                              conf=self.get_configuration(),
                              designSpace=design_space)
        self.toolbox.register('evaluate1', self.fitness_function_mo)
        self.new_best=False
        
    def initialize_population(self):
        ## the while loop exists, as meta-heuristic makes no sense till we find at least one particle that is within valid region...
        try:
            part = self.create_particle(self.get_configuration().always_valid)
            self.toolbox.filter_particle(part)
            part.fitness.values, part.code, cost = self.fitness_function(part)
            if not self.get_best() or self.is_better(part.fitness, self.get_best().fitness):
                particle = self.create_particle(part)
                particle.fitness.values = part.fitness.values
                self.set_best(particle)
            logging.info("Always valid configuration present, evaluated")
        except:
            logging.info("Always valid configuration not-present, make sure that the valid design space is large enough so that at least one valid design is initially evalauted")
            
        self.set_at_least_one_in_valid_region(False)
        #F = copy(self.get_configuration().F)
        designSpace = self.fitness.designSpace
        D = len(designSpace)
        latin_hypercube_samples = lhs.lhs(scipy_uniform,[0,1],(self.get_configuration().population_size,D))
        max_bounds = array([d["max"] for d in designSpace])
        min_bounds = array([d["min"] for d in designSpace])
        latin_hypercube_samples = latin_hypercube_samples * (max_bounds - min_bounds)
        latin_hypercube_samples = latin_hypercube_samples + min_bounds
        population = []
        f_counter = 0
        for part in latin_hypercube_samples:
            part = self.create_particle(part)
            self.toolbox.filter_particle(part)
           # if f_counter < self.get_configuration().population_size/2.:
            part.fitness.values, part.code, cost = self.fitness_function(part)
            self.set_at_least_one_in_valid_region((part.code == 0) or self.get_at_least_one_in_valid_region())
            if not self.get_best() or self.is_better(part.fitness, self.get_best().fitness):
                particle = self.create_particle(part)
                particle.fitness.values = part.fitness.values
                self.set_best(particle)
            population.append(part)
            f_counter = f_counter + 1
        self.set_population(population)
        ## get objectives
        objectives = self.fitness.objectives
        
        while (not self.get_at_least_one_in_valid_region()):
            part = self.toolbox.particle() ## a random particle
            logging.info("All particles within invalid search space.. Evaluating extra examples: " + str(part))
            part.fitness.values, part.code, cost = self.toolbox.evaluate1(part)

            self.set_at_least_one_in_valid_region((part.code == 0) or self.get_at_least_one_in_valid_region())
            flag = self.update_nondominate_archive(part)
            best = self.select_best()
            if not self.get_best() or self.get_best() != best:
                if best != 'null':
                    self.set_best_mo(best)
                else:
                    self.set_best_mo(part)
                    #if not self.get_best() or self.fitness.dominates(part.fitness.values, self.get_best().fitness.values):
                        #particle = self.create_particle(part)
                        #particle.fitness.values = part.fitness.values
                        #self.set_best(part)
                       
            
        self.state_dictionary["fresh_run"] = True
    
    # display pareto points, for testing
    def display_pareto(self, pareto_dict):
        L = []
        for k in pareto_dict.keys():
            if pareto_dict[k] != 'null':
                L.append(pareto_dict[k]['position'])
        logging.info("Currently Pareto Front points: "+ str(L))
    
    # update the external achive
    def update_nondominate_archive(self, part):
        filepath = self.get_results_folder() + "/pareto_front.txt"
        if os.path.exists(filepath):
        #read from file
            pareto_dict = {}
            with open(filepath, 'rb') as outfile:
                dict = pickle.load(outfile)
                pareto_dict = dict

            if self.nondominated(pareto_dict, part.fitness.values):
                distance = self.distance_measure(pareto_dict,part.fitness.values)
                part_dict = {
                         'position': part.fitness.values,
                         'particle': part,
                         'density': sys.maxint if distance==0 else 1/distance,
                         'sparse' : distance
                        }
                pareto_dict,irregular = self.check_update_density(pareto_dict,part_dict['position'])
                
                # if irregular, do nothing and return 
                if irregular == True: 
                    logging.info("Irregular point "+ str(part_dict['position'])+", not added")
                    return False
                
                # else add it into pareto_dict 
                # measure distance from 
                
                pareto_dict = self.remove_dominated(pareto_dict, part.fitness.values)
                
                full = 1
                for k in pareto_dict.keys():
                    if pareto_dict[k] == 'null':
                        pareto_dict[k] = part_dict
                        full = 0
                        break
                        
                # if the dict is full
                if full == 1:
                    pareto_dict = self.replace_most_density(pareto_dict,part_dict)
                logging.info("New particle with position "+ str(part_dict['position']) +" added to Pareto front set.")
                pareto_dict=self.update_density(pareto_dict)
                with io.open(filepath, 'wb') as outfile:
                    pickle.dump(pareto_dict, outfile)
                #self.display_pareto(pareto_dict)
                return True
            else:
                return False        
        else:
            #create new golden resultfile
            logging.info("New pareto front file to be created")
            part_dict = {
                         'particle': part,
                         'position': part.fitness.values,
                         'density': 0,
                         'sparse': sys.maxint
                        }
            pareto_dict = {'0': part_dict}
            for i in range(1,self.fitness.pareto_amount):
                dict_s = str(i)
                pareto_dict[dict_s] = 'null'
            with io.open(filepath, 'wb') as outfile:
                pickle.dump(pareto_dict, outfile)
                return True            
 
    ## check if pareticle is non-dominated on the external archive
    def nondominated(self, pareto_dict, values):
        bo = True
        for k in pareto_dict.keys():
            if pareto_dict[k] != 'null':
                if self.fitness.dominate(pareto_dict[k]['position'],values):
                    bo = False
        
        return bo
        
    ## remove the dominated particle by the new adder
    def remove_dominated(self, pareto_dict, values):
        for k in pareto_dict.keys():
            if pareto_dict[k] != 'null':
                if self.fitness.dominate(values, pareto_dict[k]['position']):
                    #logging.info("XXXXXXXXX"+str(values)+"Dominate "+str(pareto_dict[k]['position']))
                    logging.info("Dominated pareto point"+ str(pareto_dict[k]['position'])+ " removed ")
                    pareto_dict[k] = 'null'
        #self.display_pareto(pareto_dict)
        
        
        # update density
        pareto_dict = self.update_density(pareto_dict)
        
        return pareto_dict
   
    ## check the density of the particle
    def distance_measure(self, pareto_dict, values): 
        nearest_1 = sys.maxint
        nearest_2 = sys.maxint

        for k in pareto_dict.keys():
             if pareto_dict[k] != 'null':
                 Euclidean = 0.0
                 for (a,b) in zip(pareto_dict[k]['position'],values):
                     Euclidean = Euclidean + square(a-b)
                 Euclidean = sqrt(Euclidean)
                 if Euclidean < nearest_1:
                     nearest_1 = Euclidean 
                 else:
                     if Euclidean < nearest_2:
                         nearest_2 = Euclidean
        if nearest_2 == sys.maxint:
            nearest_2 = 0.0
        
        return nearest_1 + nearest_2
    
    
    # update density for whole pareto dict
    def update_density(self, pareto_dict):
        for k in pareto_dict.keys():
            if pareto_dict[k] != 'null':
                first,nearest_1,second,nearest_2,irregular = self.get_nearest_2(pareto_dict,pareto_dict[k]['position'])
                if first == 'null':
                    pareto_dict[k]['density'] = 0
                    pareto_dict[k]['sparse'] = sys.maxint
                else:
                    if second == 'null':
                        pareto_dict[k]['density'] = 1/nearest_1
                        pareto_dict[k]['sparse'] = nearest_1
                        
                    else:
                        if irregular != True:
                            pareto_dict[k]['density'] = 1/(nearest_1+nearest_2)
                            pareto_dict[k]['sparse'] = nearest_1 + nearest_2
        return pareto_dict
    
    # get nearest 2 points in the pareto dict, for the measure of density and sparse 
    def get_nearest_2(self, pareto_dict, values):
        nearest_1 = sys.maxint
        nearest_2 = sys.maxint
        first = 'null'
        second ='null'
        irregular = False
        irregular_count = 0
        #max_distance = self.select_least_density()['sparse']
#        logging.info("*****max_distance now is "+ str(max_distance))
        local_sum = 0.0
        pareto_population = self.get_pareto_population(pareto_dict)
        for k in pareto_dict.keys():
            if pareto_dict[k] != 'null':
                Euclidean = 0.0
                for (a,b) in zip(pareto_dict[k]['position'],values):
                    Euclidean = Euclidean + square(a-b)
                Euclidean = sqrt(Euclidean)
                if Euclidean != 0:
                    local_sum = local_sum + Euclidean
                    if Euclidean < nearest_1:
                        nearest_1 = Euclidean
                        first = k
                    else:
                        if Euclidean < nearest_2:
                            nearest_2 = Euclidean
                            second = k
 #       logging.info("*****population now is "+ str(pareto_population))
 #       logging.info("*****local_sum now is "+ str(local_sum))

        # filter irregular points
#        if pareto_population > 0:
#            logging.info("*****divide now is "+ str(local_sum/pareto_population))
#            if local_sum/pareto_population >= 5*max_distance:
#                irregular = True
#               return pareto_dict, irregular
        return first,nearest_1,second,nearest_2,irregular
        
    
    ## update density of pareto dict with one particle and check if it is a regular point
    def check_update_density(self, pareto_dict, values):  
        first,nearest_1,second,nearest_2,irregular = self.get_nearest_2(pareto_dict,values)
        if first == 'null':
            return pareto_dict,True
        else:
            if second == 'null':
                nearest_2 = 0
                pareto_dict[first]['density'] = 1/nearest_1
                pareto_dict[first]['sparse'] = nearest_1
                return pareto_dict, False
            else: 
                if irregular != True:
                    nearest_1_1 = self.get_nearest_1(pareto_dict, pareto_dict[first])
                    pareto_dict[first]['density'] = 1/(nearest_1_1 + nearest_1)
                    pareto_dict[first]['sparse'] = nearest_1_1 + nearest_1
                    nearest_2_1 = self.get_nearest_1(pareto_dict, pareto_dict[second])
                    pareto_dict[second]['density'] = 1/(nearest_2_1 + nearest_2)
                    pareto_dict[first]['sparse'] = nearest_2_1 + nearest_2
                
                return pareto_dict,irregular
    
    # get population now 
    def get_pareto_population(self, pareto_dict):
        population = 0
        for k in pareto_dict.keys():
            if pareto_dict[k] != 'null':
                population = population + 1
        return population
    
    # find the nearest particle in the pareto front dict 
    def get_nearest_1(self, pareto_dict,part_dict):
        nearest = sys.maxint
        for k in pareto_dict.keys():
            if pareto_dict[k] != 'null':
                Euclidean = 0.0
                for (a,b) in zip(pareto_dict[k]['position'],part_dict['position']):
                    Euclidean = Euclidean + square(a-b)
                Euclidean = sqrt(Euclidean)
                if Euclidean !=0 and Euclidean < nearest:
                    nearest = Euclidean
        return nearest
        
    ## replace the most density one with the current part when full 
    def replace_most_density(self, pareto_dict, part_dict):
        most_density = 0
        key = '0'
        for k in pareto_dict.keys():
            if pareto_dict[k]['density'] > most_density:
                most_density = pareto_dict[k]['density']
                key = k

        if (most_density > part_dict['density']):
            pareto_dict[key] = part_dict
        #self.display_pareto(pareto_dict)
        pareto_dict = self.update_density(pareto_dict)
        return pareto_dict
    
    # simly select the least density point in the least density point 
    def select_least_density(self):
        filepath = self.get_results_folder() + "/pareto_front.txt"
        if os.path.exists(filepath):
            #read from file
            pareto_dict = {}
            density_list = []
            with open(filepath, 'rb') as outfile:
                dict = pickle.load(outfile)
                pareto_dict = dict
            pareto_dict=self.update_density(pareto_dict)
            
            key = 'null'
            density = sys.maxint
            for k in pareto_dict.keys():
                if pareto_dict[k] != 'null':
                    density_list.append(pareto_dict[k]['density'])
                    #logging.info(str(pareto_dict[key]['density'])+"&&&&")
                    if pareto_dict[k]['density'] < density:
                        key = k
                        density = pareto_dict[k]['density']
            
            #logging.info(" *** *** **Best point selected "+ str(pareto_dict[key]['position'])+" with density "+ str(pareto_dict[key]['density'])+" Others are "+str(density_list))
            
            if key == 'null':
                return 'null'
            else:
                return pareto_dict[key]['particle']
        else:
            logging.error("No pareto file exists.")
            return 'null'
        
        
    # select best from pareto front achive use roulette, less density point with large sparse messure get more change to be selected
    def select_best(self):
        filepath = self.get_results_folder() + "/pareto_front.txt"
        if os.path.exists(filepath):
            #read from file
            pareto_dict = {}
            with open(filepath, 'rb') as outfile:
                dict = pickle.load(outfile)
                pareto_dict = dict
                #self.display_pareto(pareto_dict)
            pareto_dict=self.update_density(pareto_dict)
            density = []
            # roulette choose the best
            sum_sparse = 0.0
            for k in pareto_dict.keys():
                if pareto_dict[k] != 'null':
                    sum_sparse = sum_sparse + pareto_dict[k]['sparse']
                    density.append(pareto_dict[k]['density'])
            pivot = 0.0
            for k in pareto_dict.keys():
                if pareto_dict[k] != 'null':
                    pareto_dict[k]['lower'] = pivot
                    pivot = pivot + pareto_dict[k]['sparse']/sum_sparse
                    pareto_dict[k]['upper'] = pivot
            random_num = random()
                
            key = 'null'
            for k in pareto_dict.keys():
                if pareto_dict[k] != 'null':
                    if random_num >= pareto_dict[k]['lower'] and random_num <= pareto_dict[k]['upper']:
                        key = k
            if key != 'null':
                #logging.info(" ^^^ ^^ ^^Best point selected with density "+ str(pareto_dict[key]['density'])+"Others are "+str(density))
                return pareto_dict[key]['particle']
            else:
                return 'null'
        else:
            logging.error("No pareto file exists.")
            return 'null'
                
                
    def run(self):
        self.state_dictionary['generate'] = True
        
        logging.info(str(self.get_name()) + ' started')
        logging.info('Multi-Objective Trial prepared... executing')
        self.save() ## training might take a bit...
        # Initialise termination check
        
        self.check = False
        ## we do this not to retrain model twice during the first iteration. If we ommit
        ## this bit of code the first view_update wont have a model aviable.
        reevalute = False
        if self.state_dictionary["fresh_run"]: ## we need this as we only want to do it for initial generation because the view
            ## the problem is that we cannot

            try:
                self.train_surrogate_model(self.get_population())
                self.train_cost_model(self.get_population())
            except:
                logging.error("Cannot train the model")
            #logging.info("@@@nima Snapshot:::::"+str(self.snapshot()))
            self.view_update(visualize = True)
            self.state_dictionary["fresh_run"] = False
            self.save()
            
        while self.get_counter_dictionary('g') < self.get_configuration().max_iter + 1:
            filepath = self.get_results_folder() + "/pareto_front.txt"
            if os.path.exists(filepath):
                #read from file
                pareto_dict = {}
                with open(filepath, 'rb') as outfile:
                    dict = pickle.load(outfile)
                    pareto_dict = dict
            logging.info('------------------New Generation-----------------')
            logging.info('[' + str(self.get_name()) + '] Generation ' + str(self.get_counter_dictionary('g')))
            logging.info('[' + str(self.get_name()) + '] Fitness ' + str(self.get_counter_dictionary('fit')))
            self.display_pareto(pareto_dict)
            # termination condition - we put it here so that when the trial is reloaded
            # it wont run if the run has terminated already

            #if self.get_terminating_condition(): 
             #   logging.info('Terminating condition reached...')
              #  break
            
            # Roll population
            first_pop = self.get_population().pop(0)
            self.get_population().append(first_pop)
            
            if self.get_counter_dictionary('fit') > self.get_configuration().max_fitness:
                logging.info('Fitness counter exceeded the limit... exiting')
                break
            reevalute = False
            
            # if training set_updated Train surrogate model
            if self.training_set_updated():
                try:
                    self.train_surrogate_model(self.get_population())
                    self.train_cost_model(self.get_population())
                except:
                    logging.error("Cannot train the model")
                reevalute = True
            ##print self.get_population()
            try:
                code, mu, variance, ei, p = self.predict_surrogate_mo_model(self.get_population())
            except:
                logging.error("Cannot predict")
            reloop = False
            if (mu is None) or (variance is None):
                logging.info("Prediction Failed")
                self.set_model_failed(True)
            else:
                logging.info("mean S2 " + str(mean(variance)))
                logging.info("max S2  " + str(max(variance)))
                logging.info("min S2  " + str(min(variance)))
                logging.info("over " + str(self.get_configuration().max_stdv) + ': ' + str(min(len([v for v in variance if v > self.get_configuration().max_stdv]))))
                logging.info("over 0.01  " + str(min(len([v for v in variance if v > 0.01]))))
                logging.info("mean p " + str(mean(p)))
                logging.info("max p  " + str(max(p)))
                logging.info("min p  " + str(min(p)))
                logging.info("over 0.05  " + str(min(len([v for v in p if v > 0.05]))))
                logging.info("over 0.1  " + str(min(len([v for v in p if v > 0.01]))))
                logging.info("mean ei " + str(mean(ei)))
                logging.info("max ei  " + str(max(ei)))
                logging.info("min ei  " + str(min(ei)))
                
                # check if we should re evaluate and retrain
                reloop = self.post_model_filter(code, mu, variance)
            ##
            if self.get_model_failed():
                logging.info('Model Failed, sampling design space')
                self.sample_design_space()
            elif reloop:
                reevalute = True
                logging.info('Evaluated some particles, will try to retrain model')
            else:#
                if reevalute:
                    self.reevalute_best()
                # Iteration of meta-heuristic
                self.meta_iterate()
                self.filter_population()
                
                #Check if perturbation is neccesary 
                if self.get_counter_dictionary('g') % self.get_configuration().M == 0:# perturb
                    self.evaluate_best()
                self.new_best = False
            # Wait until the user unpauses the trial.
            while self.get_wait():
                time.sleep(0)
            
            self.increment_main_counter()
            self.view_update(visualize = True)
        self.exit()
        
    ### returns a snapshot of the trial state
    def snapshot(self):
        fitness = self.fitness
        best_fitness_array = copy(self.get_best_fitness_array())
        filepath = self.get_results_folder() + "/pareto_front.txt"
        pareto_dict = {}
        if os.path.exists(filepath):
            #read from file
            with open(filepath, 'rb') as outfile:
                dict = pickle.load(outfile)
                pareto_dict = dict
        
        generations_array = copy(self.get_generations_array())
        results_folder = copy(self.get_results_folder())
        images_folder = copy(self.get_images_folder())
        counter = copy(self.get_counter_dictionary('g'))
        name = self.get_name()
        return_dictionary = {
            'fitness': fitness,
            'best_fitness_array': best_fitness_array,
            'pareto_dict' : pareto_dict,
            'generations_array': generations_array,
            'speed_array' : self.state_dictionary['particle_speed_array'],
            'configuration_folder_path':self.configuration.configuration_folder_path,
            'run_folders_path':self.configuration.results_folder_path,
            'results_folder': results_folder,
            'images_folder': images_folder,
            'counter': counter,
            'counter_dict':  self.state_dictionary['counter_dictionary'] ,
            'timer_dict':  self.state_dictionary['timer_dict'] ,
            'name': name,
            'fitness_state': self.get_fitness_state(),
            'run_name': self.my_run.get_name(),
            'classifier1': self.get_surrogate_model()[0].classifier, ## return a copy! 
            'regressor1': self.get_surrogate_model()[0].regressor, ## return a copy!
            'classifier2': self.get_surrogate_model()[1].classifier, ## return a copy! 
            'regressor2': self.get_surrogate_model()[1].regressor, ## return a copy!
            'cost_model': self.get_cost_model(), ## return a copy!
            'meta_plot': {"particles":{'marker':"o",'data':self.get_population()}},
            'generate' : self.state_dictionary['generate'],
            'max_iter' : self.configuration.max_iter,
            'max_fitness' : self.configuration.max_fitness
        }
        return return_dictionary
        
    ####################
    ## Helper Methods ##
    ####################
    
    def checkAllParticlesEvaled(self):
        all_evaled = True
        for part in self.get_population():
            all_evaled = self.get_surrogate_model()[0].contains_training_instance(part) and all_evaled
        
        if all_evaled: ## randomize population
            self.set_population(self.toolbox.population(self.get_configuration().population_size))
            self.toolbox.filter_particles(self.get_population())
        
    def checkCollapse(self): #TODO
        ## this method checks if the particls
        ## a) collapsed onto a single point
        ## b) collapsed onto the edge of the search space
        ## if so it reintializes them.
        minimum_diverity = 0.95 ##if over 95 collapsed reseed
        
        
        if collapsed:
            self.set_population(self.toolbox.population(self.get_configuration().population_size))
            self.toolbox.filter_particles(self.get_population())
    
    def create_particle(self, particle):
        return eval('creator.Particle' + self.my_run.get_name())(particle)
        
    
    def createUniformSpace(self, particles, designSpace):
        pointPerDimensions = 5
        valueGrid = mgrid[designSpace[0]['min']:designSpace[0]['max']:
                          complex(0, pointPerDimensions),
                          designSpace[1]['min']:designSpace[1]['max']:
                          complex(0, pointPerDimensions)]

        for i in [0, 1]:
            for j, part in enumerate(particles):
                part[i] = valueGrid[i].reshape(1, -1)[0][j]

    def filterParticles(self,  particles, designSpace):
        for particle in particles:
            self.filterParticle(particle, designSpace)
            
    def filterParticle(self, p, designSpace):
        p.pmin = [dimSetting['min'] for dimSetting in designSpace]
        p.pmax = [dimSetting['max'] for dimSetting in designSpace]

        for i, val in enumerate(p):
            #dithering
            if designSpace[i]['type'] == 'discrete':
                if uniform(0.0, 1.0) < (p[i] - floor(p[i])):
                    p[i] = ceil(p[i])  # + designSpace[i]['step']
                else:
                    p[i] = floor(p[i])

            #dont allow particles to take the same value
            p[i] = minimum(p.pmax[i], p[i])
            p[i] = maximum(p.pmin[i], p[i])

    def generate(self,  designSpace):
        particle = [uniform(dimSetting['min'], dimSetting['max'])
                    for dimSetting
                    in designSpace]
        particle = self.create_particle(particle)
        return particle

        # update the position of the particles
        # should change this part in order to change the leader election strategy
        
    def meta_iterate(self):
        #TODO - reavluate one random particle... do it.. very important!
        ##while(self.get_at_least_one_in_valid_region()):
        ##    logging.info("All particles within invalid area... have to randomly sample the design space to find one that is OK...")             
            
        #Update Bests
        logging.info("Meta Iteration")
        for part in self.get_population():
            
            # personal best
            if not part.best or self.fitness.dominate(part.fitness.values, part.best.fitness.values):
                part.best = self.create_particle(part)
                part.best.fitness.values = part.fitness.values
            #if not part.best or self.fitness.dominate(part.fitness.values, part.best.fitness.values):
             #   part.best = self.create_particle(part)
              #  part.best.fitness.values = part.fitness.values
            
            # global best
            flag = self.update_nondominate_archive(part)
            best = self.select_best()
            if not self.get_best() or self.get_best() != best:
                best = self.select_best()
                if best != 'null':
                    self.set_best_mo(best)
                else:
                    self.set_best_mo(part)
            #if not self.get_best() or self.fitness.dominate(part.fitness.values, self.get_best().fitness.values):
             #   particle = self.create_particle(part)
              #  particle.fitness.values = part.fitness.values
               # self.set_best(particle)
                #self.new_best = True
                                
        #PSO
        for part in self.get_population():
            self.toolbox.update(part, self.get_counter_dictionary('g'))

    def updateParticle(self,  part, generation, conf, designSpace):
        if conf.admode == 'fitness':
            fraction = self.fitness_counter / conf.max_fitness
        elif conf.admode == 'iter':
            fraction = generation / conf.max_iter
        else:
            raise('[updateParticle]: adjustment mode unknown.. ')

        u1 = [uniform(0, conf.phi1) for _ in range(len(part))]
        u2 = [uniform(0, conf.phi2) for _ in range(len(part))]
        
        ##########   this part particulately, leader election for every particle
        leader = self.get_best()
        #logging.info("^^ ^^In updating, the best is " + str(leader))
        v_u1 = map(operator.mul, u1, map(operator.sub, part.best, part))
        v_u2 = map(operator.mul, u2, map(operator.sub, self.get_best(), part))
        
        weight = 1.0
        if conf.weight_mode == 'linear':
            weight = conf.max_weight - (conf.max_weight - conf.min_weight) * fraction
        elif conf.weight_mode == 'norm':
            weight = conf.weight
        else:
            raise('[updateParticle]: weight mode unknown.. ')
        weightVector = [weight] * len(part.speed)
        part.speed = map(operator.add,
                         map(operator.mul, part.speed, weightVector),
                         map(operator.add, v_u1, v_u2))

        
        # what's this mean?
        if conf.applyK is True:
            phi = array(u1) + array(u1)

            XVector = (2.0 * conf.KK) / abs(2.0 - phi -
                                            sqrt(pow(phi, 2.0) - 4.0 * phi))
            part.speed = map(operator.mul, part.speed, XVector)
       
        # what's the difference between these modes?
        if conf.mode == 'vp':
            for i, speed in enumerate(part.speed):
                speedCoeff = (conf.K - pow(fraction, conf.p)) * part.smax[i]
                if speed < -speedCoeff:
                    part.speed[i] = -speedCoeff
                elif speed > speedCoeff:
                    part.speed[i] = speedCoeff
                else:
                    part.speed[i] = speed
        elif conf.mode == 'norm':
            for i, speed in enumerate(part.speed):
                if speed < part.smin[i]:
                    part.speed[i] = part.smin[i]
                elif speed > part.smax[i]:
                    part.speed[i] = part.smax[i]
        elif conf.mode == 'exp':
            for i, speed in enumerate(part.speed):
                maxVel = (1 - pow(fraction, conf.exp)) * part.smax[i]
                if speed < -maxVel:
                    part.speed[i] = -maxVel
                elif speed > maxVel:
                    part.speed[i] = maxVel
        elif conf.mode == 'no':
            pass
        else:
            raise('[updateParticle]: mode unknown.. ')
        
        absolute_velocity = 0
        for e in part.speed:
            absolute_velocity = absolute_velocity + float(e) * float(e)
        if len(self.state_dictionary['particle_speed_array'])>200:
            self.state_dictionary['particle_speed_array'] = []
        if absolute_velocity>0:
            self.state_dictionary['particle_speed_array'].append(sqrt(absolute_velocity))
        else:
            self.state_dictionary['particle_speed_array'].append(0)
        #part.speedList.append(part.speed)
        #logging.info("Speed List "+ str(part.speedList))
        part[:] = map(operator.add, part, part.speed)
        


    def filter_population(self):
        self.toolbox.filter_particles(self.get_population())
   
    def evaluate_best(self):        
        if self.new_best:
            self.fitness_function_mo(self.get_best())
            
            logging.info('New best was found after M :' + str(self.get_best()))
        else:
            self.sample_design_space()
            '''
            ## TODO - clean it up...  messy
            perturbation = self.perturbation(radius = 100.0)                        
            logging.info('Best was already evalauted.. adding perturbation ' + str(perturbation))
            perturbed_particle = self.create_particle(self.get_best())
            code, mean, variance, ei, p = self.predict_surrogate_mo_model([perturbed_particle])
            if code is None:
                logging.debug("Code is none..watch out")
            if code is None or code[0] == 0:
                logging.info('Perturbation might be valid, evaluationg')
            for i,val in enumerate(perturbation):
                perturbed_particle[i] = perturbed_particle[i] + val       
            self.toolbox.filter_particle(perturbed_particle)
            fitness, code, cost = self.fitness_function_mo(perturbed_particle) 
            ##check if the value is not a new best
            perturbed_particle.fitness.values = fitness
            
            flag = self.update_nondominate_archive(perturbed_particle)
            best = self.select_best()
            if not self.get_best() or self.get_best() != best:
                best = self.select_best()
                if best != 'null':
                    self.set_best_mo(best)
                else:
                    self.set_best_mo(part)
            #if not self.get_best() or self.fitness.dominate(perturbed_particle.fitness.values, self.get_best().fitness.values):
                #self.set_best(perturbed_particle)
            else: ## why do we do this? 
                if code[0] != 0:
                    logging.info('Best is within the invalid area ' + str(code[0]) + ', sampling design space')
                    self.sample_design_space()
            '''
        
    def increment_main_counter(self):
        self.get_best_fitness_array().append(self.get_best().fitness.values[0])
        self.get_generations_array().append(self.get_counter_dictionary(self.get_main_counter_name()))
        self.save()
        self.increment_counter(self.get_main_counter_name())

    def sample_design_space(self):
        logging.info('Evaluating best perturbation')
        particle = self.surrogate_model.max_ei(designSpace=self.fitness.designSpace, hypercube = self.hypercube())
        if particle is None:
            logging.info("Local sampling has failed, probably all of the particles are within invalid region")
            particle = self.surrogate_model.max_ei(designSpace=self.fitness.designSpace)
            if particle is None:
                particle = self.toolbox.particle()
                logging.info("Global sampling failed as well.. Evaluating a random particle"  + str(particle))
        particle = self.create_particle(particle)   
        
        self.toolbox.filter_particle(particle)
        particle.fitness.values, code, cost = self.fitness_function_mo(particle) 
        flag = self.update_nondominate_archive(particle)
        best = self.select_best()
        if not self.get_best() or self.get_best() != best:
            best = self.select_best()
            if best != 'null':
                self.set_best_mo(best)
            else:
                self.set_best_mo(part)
            #if not self.get_best() or self.fitness.dominate(perturbed_particle.fitness.values, self.get_best().fitness.values):
                #self.set_best(perturbed_particle)
        
    ## not used currently
    # def get_perturbation_dist(self):
        # [max_diag, min_diag] = self.get_dist()
        # d = (max_diag - min_diag)/2.0
        # if best:
            # max_diag = best + d
            # for i,dd in enumerate(self.fitness.designSpace):
                # max_diag[i] = minimum(max_diag[i],dd["max"])
            # min_diag = best - d
            # for i,dd in enumerate(self.fitness.designSpace):
                # min_diag[i] = maximum(min_diag[i],dd["min"])
            # return [max_diag,min_diag]
        # else:
            # return getHypercube(pop)
            
     ## not used currently
    def get_dist(self):
        if best:
            distances = sqrt(sum(pow((self.surrogate.best),2),axis=1))  # TODO
            order_according_to_manhatan = argsort(distances)
            closest_array = [gpTrainingSet[index] for index in order_according_to_manhatan[0:conf.nClosest]]
        ###        
        ## limit to hypercube around the points
        #find maximum
        #print "[getDist] closestArray ",closestArray
        max_diag = deepcopy(closestArray[0])
        for part in closest_array:
            max_diag = maximum(part, max_diag)
        ###find minimum vectors
        min_diag = deepcopy(closest_array[0])
        for part in closest_array:
            min_diag = minimum(part, min_diag)
        return [max_diag, min_diag]
        
    def hypercube(self):
        #find maximum
        max_diag = deepcopy(self.get_population()[0])
        for part in self.get_population():
            max_diag = maximum(part,max_diag) 
        ###find minimum vectors
        min_diag = deepcopy(self.get_population()[0])
        for part in self.get_population():
            min_diag = minimum(part,min_diag)
            
        ## we always ensure that the hypercube allows particles to maintain velocity components in all directions
        
        for i,dd in enumerate(max_diag):
            if self.fitness.designSpace[i]["type"] == "discrete":
                max_diag[i] = minimum(dd + self.fitness.designSpace[i]["step"],self.fitness.designSpace[i]["max"])
            elif self.fitness.designSpace[i]["type"] == "continuous":
                small_fraction = ((self.fitness.designSpace[i]["max"] - self.fitness.designSpace[i]["min"]) / 100.)
                max_diag[i] = minimum(dd + small_fraction, self.fitness.designSpace[i]["max"])
                
        for i,dd in enumerate(min_diag):
            if self.fitness.designSpace[i]["type"] == "discrete":
                min_diag[i] = maximum(dd - self.fitness.designSpace[i]["step"],self.fitness.designSpace[i]["min"])
            elif self.fitness.designSpace[i]["type"] == "continuous":
                small_fraction = ((self.fitness.designSpace[i]["max"] - self.fitness.designSpace[i]["min"]) / 100.)
                min_diag[i] = maximum(dd - small_fraction, self.fitness.designSpace[i]["min"])
        logging.info("hypecube: " + str([max_diag,min_diag]))
        return [max_diag,min_diag]
        
    def perturbation(self, radius = 10.0):
        [max_diag,min_diag] = self.hypercube()
        d = (max_diag - min_diag)/radius
        for i,dd in enumerate(d):
            if self.fitness.designSpace[i]["type"] == "discrete":
                d[i] = maximum(dd,self.fitness.designSpace[i]["step"])
            elif self.fitness.designSpace[i]["type"] == "continuous":
                d[i] = maximum(dd,self.smax[i])
        dimensions = len(self.fitness.designSpace)
        pertubation =  multiply(((rand(1,dimensions)-0.5)*2.0),d)[0] #TODO add the dimensions
        return pertubation
        
        ###check if between two calls to this functions any fitness functions have been evaluted, so that the models have to be retrained
    def training_set_updated(self):
        retrain_model_temp = self.get_retrain_model()
        self.set_retrain_model(False)
        return retrain_model_temp
    
    ### TODO - its just copy and pasted code now..we could rewrite it really
    def post_model_filter(self, code, mean, variance):
        eval_counter = 1
        self.set_model_failed(not (False in [self.get_configuration().max_stdv < pred for pred in variance]))
        if self.get_model_failed():
            return False
        if (code is None) or (mean is None) or (variance is None):
            self.set_model_failed(False)
        else:
            for i, (p, c, m, v) in enumerate(zip(self.get_population(), code, mean, variance)):
                if v > self.get_configuration().max_stdv and c == 0:
                    if eval_counter > self.get_configuration().max_eval:
                        logging.info("Evalauted more fitness functions per generation than max_eval")
                        self.checkAllParticlesEvaled() ## if all the particles have been evalauted 
                        return True
                    p.fitness.values, p.code, cost = self.toolbox.evaluate1(p)
                    eval_counter = eval_counter + 1
                else:
                    try:
                        if c == 0:
                            p.fitness.values = m
                        else:
                            p.fitness.values = [self.fitness.worst_value]
                    except:
                        p.fitness.values, p.code, cost = self.toolbox.evaluate1(p)
            ## at least one particle has to have std smaller then max_stdv
            ## if all particles are in invalid zone
        return False
   
    def reevalute_best(self):
        bests_to_model = [p.best for p in self.get_population() if p.best] ### Elimate Nones -- in case M < Number of particles, important for initialb iteratiions
        if self.get_best():
            bests_to_model.append(self.get_best())
        if bests_to_model:
            logging.info("Reevaluating")
            code, bests_to_fitness, variance, ei, p = self.predict_surrogate_model(bests_to_model)
            if (code is None) or (bests_to_fitness is None) or (variance is None):
                logging.info("Prediction failed during reevaluation... omitting")
            else:
                for i,part in enumerate([p for p in self.get_population() if p.best]):
                    if code[i] == 0:
                        part.best.fitness.values = bests_to_fitness[i]
                    else:
                        part.best.fitness.values = [self.fitness.worst_value]
                if self.get_best():
                    best = self.get_best()
                    if code[-1] == 0:
                        best.fitness.values = bests_to_fitness[-1]
                    else:
                        best.fitness.values = [self.fitness.worst_value]
                    self.set_best_mo(best)
    
    
    #######################
    ### GET/SET METHODS ###
    #######################
    
    def get_predicted_time(self):
        predicted_time = self.state_dictionary['total_time'] * self.get_configuration().max_iter / (self.get_counter_dictionary('g') + 1.0)
        return str(timedelta(seconds=predicted_time))
    
    def set_population(self, population):
        self.state_dictionary["population"] = population
        
    def get_population(self):
        return self.state_dictionary["population"]
        
    def get_best_fitness_array(self):
        return self.state_dictionary['best_fitness_array']
        
    def get_generations_array(self):
        return self.state_dictionary['generations_array']
        
    def set_at_least_one_in_valid_region(self, state):
        if self.state_dictionary.has_key('at_least_one_in_valid_region'):
            self.state_dictionary['at_least_one_in_valid_region'] = state
        else:
            self.state_dictionary['at_least_one_in_valid_region'] = False

    def get_at_least_one_in_valid_region(self):
        return self.state_dictionary['at_least_one_in_valid_region']
        
    def get_surrogate_model(self): ## returns a copy of the model... quite important not to return the model itself as ll might get F up
        model = []
        for i in range(len(self.surrogate_model)):
            model.append( ProperSurrogateModel(self.get_configuration(), self.controller) )
            model[i].set_state_dictionary(self.surrogate_model[i].get_state_dictionary())
        return model

    def get_cost_model(self): ## returns a copy of the model... quite important not to return the model itself as ll might get F up
        model = DummyCostModel(self.get_configuration(), self.controller, self.fitness)
        model.set_state_dictionary(self.cost_model.get_state_dictionary())
        return model
    def get_classifier(self):
        return self.get_surrogate_model().get_classifier() 
        
    def get_regressor(self):
        return self.get_surrogate_model().get_regressor()
    
class PSOTrial_TimeAware(PSOTrial):
        
    def run_initialize(self):
        logging.info("Initialize PSOTrial_TimeAware no:" + str(self.get_trial_no()))
        self.cost_model = ProperCostModel(self.configuration, self.controller, self.fitness)
        design_space = self.fitness.designSpace
        self.toolbox = copy(base.Toolbox())
        self.smin = [-1.0 * self.get_configuration().max_speed *
                (dimSetting['max'] - dimSetting['min'])
                for dimSetting in design_space]
        self.smax = [self.get_configuration().max_speed *
                (dimSetting['max'] - dimSetting['min'])
                for dimSetting in design_space]

        try:
            eval('creator.Particle' + str(self.my_run.get_name()))
            logging.debug("Particle class for this run already exists")
        except AttributeError:
            creator.create('FitnessMax' + str(self.my_run.get_name()), base.Fitness, weights=(1.0,))
            ### we got to add specific names, otherwise the classes are going to be visible for all
            ### modules which use deap...
            
            creator.create(str('Particle' + self.my_run.get_name()), list, fitness=eval('creator.FitnessMax' + str(self.my_run.get_name())),
                           smin=self.smin, smax=self.smax,
                           speed=[uniform(smi, sma) for sma, smi in zip(self.smax,
                                                                        self.smin)],
                           pmin=[dimSetting['max'] for dimSetting in design_space],
                           pmax=[dimSetting['min'] for dimSetting in design_space],
                           model=False, best=None, code=None)

        self.toolbox.register('particle', self.generate, designSpace=design_space)
        self.toolbox.register('filter_particles', self.filterParticles,
                              designSpace=design_space)
        self.toolbox.register('filter_particle', self.filterParticle,
                              designSpace=design_space)
        self.toolbox.register('population', tools.initRepeat,
                              list, self.toolbox.particle)
        self.toolbox.register('update', self.updateParticle, 
                              conf=self.get_configuration(),
                              designSpace=design_space)
        self.toolbox.register('evaluate', self.fitness_function)
        self.new_best=False
        
    def predict_cost(self, particle):
        try:
            return self.cost_model.predict(particle)
        except Exception,e:
            logging.debug("Cost model is still not avaiable: " + str(e))
            return 1.0 ## model has not been created yet
        
    def predict_cost_raw(self, particle):
        try:
            return self.cost_model.predict_raw(particle)
        except Exception,e:
            logging.debug("Cost model is still not avaiable: " + str(e))
            return 1.0 ## model has not been created yet
        
    def updateParticle(self,  part, generation, conf, designSpace):
        if conf.admode == 'fitness':
            fraction = self.fitness_counter / conf.max_fitness
        elif conf.admode == 'iter':
            fraction = generation / conf.max_iter
        else:
            raise('[updateParticle]: adjustment mode unknown.. ')
        a1 = self.a_func(part.best, part, self.predict_cost_raw(part.best), self.predict_cost_raw(part))
        u1 = [uniform(0, conf.phi1 * a1 ) for _ in range(len(part))]
        a2 = self.a_func(self.get_best(), part, self.predict_cost_raw(self.get_best()), self.predict_cost_raw(part))
        u2 = [uniform(0, conf.phi2 * a2) for _ in range(len(part))]

        #logging.info("u1 " + str(u1))
        #logging.info("part.best " + str(part.best))
        #logging.info("part " + str(part))
        v_u1 = map(operator.mul, u1, map(operator.sub, part.best, part))
        v_u2 = map(operator.mul, u2, map(operator.sub, self.get_best(), part))

        weight = 1.0
        if conf.weight_mode == 'linear':
            weight = conf.max_weight - (conf.max_weight -
                                        conf.min_weight) * fraction
        elif conf.weight_mode == 'norm':
            weight = conf.weight
        else:
            raise('[updateParticle]: weight mode unknown.. ')
        weightVector = [weight] * len(part.speed)
        part.speed = map(operator.add,
                         map(operator.mul, part.speed, weightVector),
                         map(operator.add, v_u1, v_u2))

        if conf.applyK is True:
            phi = array(u1) + array(u1)

            XVector = (2.0 * conf.KK) / abs(2.0 - phi -
                                            sqrt(pow(phi, 2.0) - 4.0 * phi))
            part.speed = map(operator.mul, part.speed, XVector)

        if conf.mode == 'vp':
            for i, speed in enumerate(part.speed):
                speedCoeff = (conf.K - pow(fraction, conf.p)) * part.smax[i]
                if speed < -speedCoeff:
                    part.speed[i] = -speedCoeff
                elif speed > speedCoeff:
                    part.speed[i] = speedCoeff
                else:
                    part.speed[i] = speed
        elif conf.mode == 'norm':
            for i, speed in enumerate(part.speed):
                if speed < part.smin[i]:
                    part.speed[i] = part.smin[i]
                elif speed > part.smax[i]:
                    part.speed[i] = part.smax[i]
        elif conf.mode == 'exp':
            for i, speed in enumerate(part.speed):
                maxVel = (1 - pow(fraction, conf.exp)) * part.smax[i]
                if speed < -maxVel:
                    part.speed[i] = -maxVel
                elif speed > maxVel:
                    part.speed[i] = maxVel
        elif conf.mode == 'no':
            pass
        else:
            raise('[updateParticle]: mode unknown.. ')
        part[:] = map(operator.add, part, part.speed)

    def initialize_population(self):
        ## the while loop exists, as meta-heuristic makes no sense till we find at least one particle that is within valid region...
        
        self.set_at_least_one_in_valid_region(False)
        F = copy(self.get_configuration().F)
        while (not self.get_at_least_one_in_valid_region()):
            self.set_population(self.toolbox.population(self.get_configuration().population_size))
            self.toolbox.filter_particles(self.get_population())
            if self.get_configuration().eval_correct:
                self.get_population()[0] = self.create_particle(
                    self.fitness.alwaysCorrect())  # This will break
            for i, part in enumerate(self.get_population()):
                if i < F:
                    part.fitness.values, part.code, cost = self.toolbox.evaluate(part)
                    self.set_at_least_one_in_valid_region((part.code == 0) or self.get_at_least_one_in_valid_region())
                    if not self.get_best() or self.fitness.is_better(part.fitness, self.get_best().fitness):
                        particle = self.create_particle(part)
                        particle.fitness.values = part.fitness.values
                        self.set_best(particle)
            ## add one example till we find something that works
            F = 1
        ## additionally if software parameters are present, we evalaute and
        hardware_axis = [i for i, dimension in enumerate(self.fitness.designSpace) if dimension["set"] == "h"]
        ##pick first axis There is one potential issue... that the random seed for software axis doesnt change.. oh well! hope it doesnt happen (unlikely)
        for i in range(0,self.get_configuration().F): ## we need extra few to build software regression... could be done differently
            extra_particle = self.toolbox.particle() ## generate a totally random particle
            for index in hardware_axis:
                extra_particle[index] = self.get_population()[i][index] ## we make sure that hardware parameters are the same
            self.toolbox.filter_particle(extra_particle)
            self.toolbox.evaluate(extra_particle)
        self.state_dictionary["fresh_run"] = True
        
    def std_func(self, part):
        cost = self.predict_cost(part)
        try:
            ratio = (cost - self.fitness.cost_minVal) / (self.fitness.cost_maxVal - self.fitness.cost_minVal)
        except:
            ratio = 0.0
        min_stdv = self.get_configuration().min_stdv
        max_stdv = self.get_configuration().max_stdv
        return min_stdv + (ratio * max_stdv)
    
    ### TODO - its just copy and pasted ciode now..w could rewrite it realyl
    def post_model_filter(self, code, mean, variance):
        eval_counter = 1
        self.set_model_failed(not (False in [self.get_configuration().max_stdv < pred for pred in variance]))
        if self.get_model_failed():
            return False
        if (code is None) or (mean is None) or (variance is None):
            self.set_model_failed(False)
        else:
            for i, (p, c, m, v) in enumerate(zip(self.get_population(), code,
                                                 mean, variance)):
                stdv = self.std_func(p)        
                if v > stdv and c == 0:
                    if eval_counter > self.get_configuration().max_eval:
                        logging.info("Evalauted more fitness functions per generation then max_eval")
                        self.checkAllParticlesEvaled()
                        return True
                    p.fitness.values, p.code, cost = self.toolbox.evaluate(p)
                    eval_counter = eval_counter + 1
                else:
                    try:
                        if c == 0:
                            p.fitness.values = m
                        else:
                            p.fitness.values = [self.fitness.worst_value]                       
                    except:
                        p.fitness.values, p.code, cost = self.toolbox.evaluate(p)
                        logging.info("KURWA Start")
                        logging.info("KURWA End")
            ## at least one particle has to have std smaller then max_stdv
            ## if all particles are in invalid zone
        return False
   
    def a_func(self, part_1, part_2, cost_1, cost_2):
        if self.configuration.a == "a1":
            val = min(abs(cost_1-cost_2)/(0.0000001 + norm(map(operator.sub, part_1, part_2))),1.0) ## second norm.. gradient
            return val
        elif self.configuration.a == "a2":
            return min(abs(cost_2)/abs(cost_1),1.0)
        elif self.configuration.a == "a3":
            return min(abs(cost_1)/abs(cost_2), 1.0)
        else:
            return 1.0
        
    #######################
    ### GET/SET METHODS ###
    #######################
    
    def get_surrogate_model(self): ## returns a copy of the model... quite important not to return the model itself as ll might get F up
        model = ProperSurrogateModel(self.get_configuration(), self.controller)
        model.set_state_dictionary(self.surrogate_model.get_state_dictionary())
        return model
    
    def get_cost_model(self): ## returns a copy of the model... quite important not to return the model itself as ll might get F up
        model = ProperCostModel(self.get_configuration(), self.controller, self.fitness)
        model.set_state_dictionary(self.cost_model.get_state_dictionary())
        return model
        

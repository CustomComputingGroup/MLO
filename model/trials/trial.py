import logging
import os
import sys
from threading import Thread
import time
from time import strftime
from datetime import datetime
from copy import copy, deepcopy
import io
import pickle
import re
import wx
from numpy import multiply, array
from numpy.random import uniform, rand

##TODO - clean it up... only should be loaded when needed... 

from ..surrogatemodels.surrogatemodel import DummySurrogateModel, ProperSurrogateModel
from particles import *

from deap import base, creator, tools

class Trial(Thread):

    def __init__(self, trial_no, run_name, fitness, configuration, controller,
                 run_results_folder_path):
        Thread.__init__(self)
        self.fitness = fitness
        self.controller = controller
        # True if the user has selected to pause the trial
        
        self.graph_dictionary = {
            'rerendering': False,
            'graph_title': configuration.graph_title,
            'graph_names': configuration.graph_names,
            'all_graph_dicts': configuration.all_graph_dicts,
        }
        for name in configuration.graph_names:
            self.graph_dictionary['all_graph_dicts'][name]['generate'] = True
        
        if configuration.surrogate_type == "dummy":
            self.surrogate_model = DummySurrogateModel(configuration,
                                                       self.controller)
        else:
            self.surrogate_model = ProperSurrogateModel(configuration,
                                                        self.controller)

        # Contains all the counter variables that may be used for visualization
        counter_dictionary = {}
        counter_dictionary['fit'] = 0 ## we always want to record fitness of the best configurations
        self.configuration = configuration
        self.state_dictionary = {
            'status' : 'Running',
            'retrain_model' : False,
            'model_failed' : False,
            'run_results_folder_path' : run_results_folder_path,
            'run_name' : run_name,
            'trial_no' : trial_no,
            'name' : '{} Trial {}'.format(run_name, trial_no),
            'all_particles_in_invalid_area' : False,
            'wait' : False,
            'generations_array' : [],
            'best_fitness_array' : [],
            'enable_traceback' : configuration.enable_traceback,
            'graph_dictionary' : self.graph_dictionary,
            'counter_dictionary' : counter_dictionary,
            'best' : None,
            # True if the user has selected to pause the trial
        }

        self.controller.register_trial(self)
        
    ####################
    ## Helper Methods ##
    ####################
    
    def train_surrogate_model(self, population):
        logging.info('Training model')
        self.get_surrogate_model().train(population)
    
    def predict_surrogate_model(self, population):
        return self.get_surrogate_model().predict(population)
        
    def create_results_folder(self):
        """
        Creates folder structure used for storing results.
        Returns a results folder path or None if it could not be created.
        """
        path = '{}/trial-{}'.format(self.get_run_results_folder_path(), self.get_trial_no())
        try:
            os.makedirs(path)
            os.makedirs(path + "/images")
            return path, path + "/images"
        except OSError, e:
            # Folder already exists
            return path
        except Exception, e:
            logging.error('Could not create folder: {}, aborting'.format(path),
                          exc_info=sys.exc_info())
            return None, None
    
    ### check first if part is already within the training set
    def fitness_function(self, part):
        ##this bit traverses the particle set and checks if it has already been evaluated. 
        if self.get_surrogate_model().contains_training_instance(part):
            code, fitness = self.get_surrogate_model().get_training_instance(part)
            if fitness is None:
                fitness = self.get
            return code, fitness
        self.increment_counter('fit')
        fitness, code, addReturn = self.fitness.fitnessFunc(part)
        self.get_surrogate_model().add_training_instance(part, code, fitness, addReturn)
        self.set_retrain_model(True)
        
        if addReturn[0] == 0:
            self.set_terminating_condition(fitness) 
            return fitness, code
        else:
            return array([self.fitness.worst_value]), code
   
    ## indicator for the controller and viewer that the state has changed. 
    def view_update(self):
        self.controller.view_update(self)

    def increment_counter(self, counter):
        self.set_counter_dictionary(counter, self.get_counter_dictionary(counter) + 1)
                
    def save(self):
        try:
            trial_file = ('{}/{:0' + str(10) +
                  'd}.txt').format(self.get_results_folder(),
                                   self.get_counter_dictionary('g'))
            dict = self.state_dictionary
            surrogate_model_state_dict = self.get_surrogate_model().get_state_dictionary()
            dict['surrogate_model_state_dict'] = surrogate_model_state_dict
            with io.open(trial_file, 'wb') as outfile:
                pickle.dump(dict, outfile)            
        except Exception, e:
            logging.error(str(e))
            return False
            
    ## by default find the latest generation
    def load(self, generation = None):
        #try:
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
                
        generation_file = ('{:0' + str(10) +
                           'd}').format(generation)
        trial_file = '{}/{}.txt'.format(self.get_results_folder(), generation_file)
        with open(trial_file, 'rb') as outfile:
            dict = pickle.load(outfile)
        self.set_state_dictionary(dict)
        self.get_surrogate_model().set_state_dictionary(dict['surrogate_model_state_dict'])
        logging.info("Loaded Trial")
        return True
        #except Exception, e:
        #    logging.error("Loading error" + str(e))
        #    return False
        
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
        
    #######################
    ### GET/SET METHODS ###
    #######################
    
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
        
    def get_graph_dictionary(self):
        return self.graph_dictionary
        
    def get_best(self):
        return self.state_dictionary['best']
        
    def set_best(self, new_best):
        self.state_dictionary['best'] = new_best
        
    def get_counter_dictionary(self, counter):
        return self.state_dictionary['counter_dictionary'][counter]
        
    def set_counter_dictionary(self, counter, value):
        self.state_dictionary['counter_dictionary'][counter] = value
        
    def get_configuration(self):
        return self.configuration
        
    def set_start_time(self, start_time):
        self.state_dictionary['start_time'] = start_time

    def get_start_time(self):
        return self.state_dictionary['start_time']
    
    def get_surrogate_model(self):
        return self.surrogate_model
        
    def set_surrogate_model(self, new_model):
        self.surrogate_model = new_model
        
    def get_wait(self):
        return self.state_dictionary["wait"]

    def get_status(self):
        return self.state_dictionary["status"]
        
    def set_status(self, status):
        self.state_dictionary["status"] = status
        
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
        return self.name
        
    def get_results_folder(self):
        return '{}/trial-{}'.format(self.get_run_results_folder_path(), self.get_trial_no()) #self.state_dictionary['results_folder']
        
    def set_images_folder(self, folder):
        self.state_dictionary['images_folder'] = folder

    def get_images_folder(self):
        return self.get_results_folder() + "/images"
        
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
        design_space = self.fitness.designSpace

        self.smin = [-1.0 * self.get_configuration().max_speed *
                (dimSetting['max'] - dimSetting['min'])
                for dimSetting in design_space]
        self.smax = [self.get_configuration().max_speed *
                (dimSetting['max'] - dimSetting['min'])
                for dimSetting in design_space]

        creator.create('FitnessMax', base.Fitness, weights=(1.0,))
        creator.create('Particle', list, fitness=creator.FitnessMax,
                       smin=self.smin, smax=self.smax,
                       speed=[uniform(smi, sma) for sma, smi in zip(self.smax,
                                                                    self.smin)],
                       pmin=[dimSetting['max'] for dimSetting in design_space],
                       pmax=[dimSetting['min'] for dimSetting in design_space],
                       model=False, best=None, code=None)

        self.toolbox = base.Toolbox()
        self.toolbox.register('particle', generate, designSpace=design_space)
        self.toolbox.register('filter_particles', filterParticles,
                              designSpace=design_space)
        self.toolbox.register('filter_particle', filterParticle,
                              designSpace=design_space)
        self.toolbox.register('population', tools.initRepeat,
                              list, self.toolbox.particle)
        self.toolbox.register('update', updateParticle, trial=self,
                              conf=self.get_configuration(),
                              designSpace=design_space)
        self.toolbox.register('evaluate', self.fitness_function)
        
    def run(self):
        self.set_start_time(datetime.now().strftime('%d-%m-%Y  %H:%M:%S'))
        
        logging.info('{} started'.format(self.get_name()))

        logging.info('Run prepared... executing')
        # Initialise termination check
        
        self.check = False

        while self.get_counter_dictionary('g') < self.get_configuration().max_iter + 1:
            logging.info('[{}] Generation {}'.format(
                self.get_name(), self.get_counter_dictionary('g')))
            logging.info('[{}] Fitness {}'.format(
                self.get_name(), self.get_counter_dictionary('fit')))
                
            self.view_update()
                
            # termination condition - we put it here so that when the trial is reloaded
            # it wont run if the run has terminated already
            if self.get_terminating_condition(): 
                logging.info('Terminating condition reached...')
                break
            
            # Roll population
            first_pop = self.get_population().pop(0)
            self.get_population().append(first_pop)
            
            if self.get_counter_dictionary('fit') > self.get_configuration().max_fitness:
                logging.info('Fitness counter exceeded the limit... exiting')
                break

            # Train surrogate model
            if self.get_retrain_model():
                self.train_surrogate_model(self.get_population())
            ##print self.get_population()
            code, mean, variance = self.predict_surrogate_model(self.get_population())
            self.post_model_filter(code, mean, variance)

            ##
            if self.get_model_failed():
                logging.info('Model Failed, sampling design space')
                self.sample_design_space()
            else:#
                if self.training_set_updated():
                    self.reevalute_best()
                # Iteration of meta-heuristic
                self.meta_iterate()
                self.filter_population()
                
                #Check if perturbation is neccesary 
                if self.get_counter_dictionary('g') % self.get_configuration().M == 0:
                    self.evaluate_best()
                
            # Wait until the user unpauses the trial.
            while self.get_wait():
                time.sleep(0)

            self.increment_main_counter()
                
        self.set_status('Finished')
        logging.info('{} finished'.format(self.get_name()))
        sys.exit(0)
        
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
            'results_folder': results_folder,
            'images_folder': images_folder,
            'counter': counter,
            'name': name,
            'classifier': self.get_surrogate_model().classifier,
            'regressor': self.get_surrogate_model().regressor,
            'meta_plot': {"particles":{'marker':"o",'data':self.get_population()}}
        }
        return_dictionary.update(copy(self.graph_dictionary))
        return return_dictionary
        
    ####################
    ## Helper Methods ##
    ####################
        
    def initialize_population(self):
        ## the while loop exists, as meta-heuristic makes no sense till we find at least one particle that is within valid region...
        
        self.set_at_least_one_in_valid_region(False)
        while (not self.get_at_least_one_in_valid_region()):
            self.set_population(self.toolbox.population(self.get_configuration().population_size))
            self.toolbox.filter_particles(self.get_population())
            if self.get_configuration().eval_correct:
                self.get_population()[0] = creator.Particle(
                    self.fitness.alwaysCorrect())  # This will break
            for i, part in enumerate(self.get_population()):
                if i < self.get_configuration().F:
                    part.fitness.values, part.code = self.toolbox.evaluate(part)
                    self.set_at_least_one_in_valid_region((part.code == 0) or self.get_at_least_one_in_valid_region())
            ## add one example till we find something that works
            self.get_configuration().F = 1
        
    def meta_iterate(self):
        #TODO - reavluate one random particle... do it.. very important!
        ##while(self.get_at_least_one_in_valid_region()):
        ##    logging.info("All particles within invalid area... have to randomly sample the design space to find one that is OK...")             
            
        #Update Bests
        for part in self.get_population():
            if not part.best or self.fitness.is_better(part.fitness, part.best.fitness):
                part.best = creator.Particle(part)
                part.best.fitness.values = part.fitness.values
            if not self.get_best() or self.fitness.is_better(part.fitness, self.get_best().fitness):
                self.set_best(creator.Particle(part))
                self.get_best().fitness.values = part.fitness.values
                self.new_best = True
                logging.info('New global best found' + str(self.get_best()) + ' fitness:'+ str(part.fitness.values))                
        #PSO
        for part in self.get_population():
            self.toolbox.update(part, self.get_counter_dictionary('g'))

    def filter_population(self):
        self.toolbox.filter_particles(self.get_population())
   
    def evaluate_best(self):        
        if self.new_best:
            self.fitness_function(self.get_best())
            logging.info('New best was found after M')
        else:
            ## TODO - clean it up...  messy
            logging.info('Best was already evalauted.. adding perturbation')
            perturbation = self.perturbation()                        
            perturbed_particle = creator.Particle(self.get_best())
            for i,val in enumerate(perturbation):
                perturbed_particle[i] = perturbed_particle[i] + val       
            self.toolbox.filter_particle(perturbed_particle)
            fitness, code = self.fitness_function(perturbed_particle) 
            ##check if the value is not a new best
            perturbed_particle.fitness.values = fitness
            if not self.get_best() or self.fitness.is_better(perturbed_particle.fitness, self.get_best().fitness):
                self.set_best(perturbed_particle)
        self.new_best = False
        
    def increment_main_counter(self):
        self.get_best_fitness_array().append(self.get_best().fitness.values[0])
        self.get_generations_array().append(self.get_counter_dictionary(self.get_main_counter_name()))
        self.save()
        self.increment_counter(self.get_main_counter_name())

    def sample_design_space(self):
        logging.info('Evaluating best perturbation')
        perturbation = self.perturbation()                        
        hypercube = self.hypercube()
        particle = self.get_surrogate_model().max_uncertainty()
        if particle is None:
            logging.info('Evaluating a perturbation of a random particle')
            particle = self.toolbox.particle()
        perturbedParticle = creator.Particle(particle)
        for i,val in enumerate(perturbation):
            perturbedParticle[i] = perturbedParticle[i] + val       
        self.toolbox.filter_particle(perturbedParticle)
        self.fitness_function(perturbedParticle) 
        
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
    
    ### TODO - its just copy and pasted ciode now..w could rewrite it realyl
    def post_model_filter(self, code, mean, variance):
        if (code is None) or (mean is None) or (variance is None):
            self.set_model_failed(False)
        else:
            for i, (p, c, m, v) in enumerate(zip(self.get_population(), code,
                                                 mean, variance)):
                if v > self.get_configuration().max_stdv and c == 0:
                    p.fitness.values, p.code = self.toolbox.evaluate(p)
                else:
                    if c == 0:
                        p.fitness.values = m
                    else:
                        p.fitness.values = [self.fitness.worst_value]
            ## at least one particle has to have std smaller then max_stdv
            ## if all particles are in invalid zone
            self.set_model_failed(not (False in [self.get_configuration().max_stdv < pred for pred in variance]))
   
    def reevalute_best(self):
        bests_to_model = [p.best for p in self.get_population() if p.best] ### Elimate Nones -- in case M < Number of particles, important for initialb iteratiions
        if self.get_best():
            bests_to_model.append(self.get_best())
        if bests_to_model:
            logging.info("Reevaluating")
            bests_to_fitness = self.get_surrogate_model().predict(bests_to_model)[1]
            for i,part in enumerate([p for p in self.get_population() if p.best]):
                part.best.fitness.values = bests_to_fitness[i]
            if self.get_best().model:
                best.fitness.values = bests_to_fitness[-1]
        
    #######################
    ### GET/SET METHODS ###
    #######################
    
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
        
        
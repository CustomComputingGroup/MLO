import logging
import os
import sys
from threading import Thread
import time
from time import strftime
from datetime import datetime
from copy import copy, deepcopy

##TODO - clean it up... only should be loaded when needed... 

from ..surrogatemodels.surrogatemodel import DummySurrogateModel, ProperSurrogateModel
from particles import *
from trialbackup import PSOTrialBackup

from deap import base, creator, tools
from numpy.random import uniform, rand

import wx

from numpy import multiply, array


class Trial(Thread):

    def __init__(self, trial_no, run_name, fitness, configuration, controller,
                 run_results_folder_path):
        Thread.__init__(self)
        self.all_particles_in_invalid_area = False
        self.name = '{} Trial {}'.format(run_name, trial_no)
        self.run_name = run_name
        self.trial_no = trial_no
        self.fitness = fitness
        self.configuration = configuration
        self.controller = controller
        self.run_results_folder_path = run_results_folder_path
        self.status = 'Running'  # TODO - BAD
        self.retrain_model = False
        self.terminating_condition_reached = False
        self.model_failed = False
        # True if the user has selected to pause the trial
        self.wait = False

        self.graph_dictionary = {
            'rerendering': False,
            'graph_title': configuration.graph_title,
            'graph_names': configuration.graph_names,
            'all_graph_dicts': configuration.all_graph_dicts,
        }
        for name in configuration.graph_names:
            self.graph_dictionary['all_graph_dicts'][name]['generate'] = True

        #if configuration.plot_view != 'special':
        #    self.plot_view = Plot_View

        self.enable_traceback = configuration.enable_traceback
        self.GEN = configuration.max_iter
        self.generations_array = []
        self.max_fitness = configuration.max_fitness
        self.best_fitness_array = []
        
        self.clf = None
        self.gp_training_fitness = None
        self.gp_training_set = None

        if configuration.surrogate_type == "dummy":
            self.surrogate_model = DummySurrogateModel(fitness, configuration,
                                                       self.controller)
        else:
            self.surrogate_model = ProperSurrogateModel(fitness,
                                                        configuration,
                                                        self.controller)

        # Contains all the counter variables that may be used for visualization
        self.counter_dictionary = {}
        self.counter_dictionary['g'] = 0
        self.counter_dictionary['fit'] = 0

        # The last counter value to be visualized, 0 means none
        self.latest_counter_plot = 0

        self.controller.register_trial(self)
        self.backup = self.create_backup_manager()

    def initialise(self):
        """
        Initialises the trial and returns True if everything went OK,
        False otherwise.
        """
        self.results_folder, self.images_folder = self._create_results_folder()
        if not self.results_folder:
            # Results folder could not be created
            return False
            
        self.run_initialize()

        self.best = None
        self.fitness_evaluated = False
        self.gp = None
        self.model_failed = False
        self.model_retrained = True

        #TODO - maybe remove this boolean
        self.new_best_over_iteration = False
        self.M_best = self.fitness.worst_value

        self.population = None
        self.svc_training_labels = None
        self.svc_training_set = None

        self.initialize_population()
        self.view_update()
        
        return True

        
        
    def run(self):
        self.start_time = datetime.now().strftime('%d-%m-%Y  %H:%M:%S')
        
        logging.info('{} started'.format(self.get_name()))

        logging.info('Run prepared... executing')
        # Initialise termination check
        
        self.check = False
#        self.view_update()
        
        while self.counter_dictionary['g'] < self.GEN + 1:
            logging.info('[{}] Generation {}'.format(
                self.get_name(), self.counter_dictionary['g']))
            logging.info('[{}] Fitness {}'.format(
                self.get_name(), self.counter_dictionary['fit']))
                
            # Roll population
            first_pop = self.population.pop(0)
            self.population.append(first_pop)
            
            if self.counter_dictionary['fit'] > self.max_fitness:
                logging.info('Fitness counter exceeded the limit... exiting')
                break

            # Train surrogate model
            if self.training_set_updated():
                logging.info('Retraining model')
                self.surrogate_model.train(self.population)
            code, mean, variance = self.surrogate_model.predict(self.population)
            self.post_model_filter(code, mean, variance)

            ##
            if self.get_model_failed():
                logging.info('Model Failed, sampling design space')
                self.sample_design_space()
            else:
                self.reevalute_best()
                # Iteration of meta-heuristic
                self.meta_iterate()
                self.filter_population()
                
                #Check if perturbation is neccesary 
                if self.counter_dictionary['g'] % self.configuration.M == 0:
                    self.evaluate_best()
                
            # Wait until the user unpauses the trial.
            while self.wait:
                time.sleep(0)

            self.increment_counter('g')
            self.view_update()

            # termination condition
            if self.terminating_condition_reached:
                break
                
        self.status = 'Finished'
        logging.info('{} finished'.format(self.get_name()))
        sys.exit(0)
           
    def get_model_failed(self):
        return self.model_failed
        
    def _create_results_folder(self):
        """
        Creates folder structure used for storing results.
        Returns a results folder path or None if it could not be created.
        """
        path = '{}/trial-{}'.format(self.run_results_folder_path,
                                    self.trial_no)
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
            return None
    
    ### check first if part is already within the training set
    def fitness_function(self, part):
        ##this bit traverses the particle set and checks if it has already been evaluated. 
        if self.surrogate_model.contains_training_instance(part):
            return self.surrogate_model.get_training_instance(part)
        self.increment_counter('fit')
        fitness, code, addReturn = self.fitness.fitnessFunc(part)
        self.surrogate_model.add_training_instance(part, code, fitness, addReturn)
        self.retrain_model = True
        
        if addReturn[0] == 0:
            self.terminating_condition_reached = self.terminating_condition(fitness) 
            if self.terminating_condition_reached:
                logging.info('Terminating condition reached...' + str(part) + ' ' + str(fitness))
            return fitness, code
        else:
            return array([self.fitness.worst_value]), code
    
    ###check if between two calls to this functions any fitness functions have been evaluted, so that the models have to be retrained
    def training_set_updated(self):
        retrain_model_temp = self.retrain_model
        self.retrain_model = False
        return retrain_model_temp
    
    # TODO - this could be removed
    def get_name(self):
        return self.name

    ## indicator for the controller and viewer that the state has changed. 
    def view_update(self):
        self.controller.view_update(self)

    def increment_counter(self, counter):
        if counter == self.configuration.counter:
            self.best_fitness_array.append(self.best.fitness.values[0])
            self.generations_array.append(self.counter_dictionary[counter])
            self.save()
        self.counter_dictionary[counter] += 1

    def save(self):
        self.backup.save_trial(self)

    def load(self, generation=None):
        return self.backup.load_trial(self, generation)

    ### TODO - its just copy and pasted ciode now..w could rewrite it realyl
    def post_model_filter(self, code, mean, variance):
        for i, (p, c, m, v) in enumerate(zip(self.population, code,
                                             mean, variance)):
            if v > self.configuration.max_stdv and c == 0:
                p.fitness.values, p.code = self.toolbox.evaluate(p)
            else:
                if c == 0:
                    p.fitness.values = m
                else:
                    p.fitness.values = [self.fitness.worst_value]
        ## at least one particle has to have std smaller then max_stdv
        ## if all particles are in invalid zone
        self.model_failed = not (False in [self.configuration.max_stdv < pred for pred in variance])
   
    def reevalute_best(self):
        bests_to_model = [p.best for p in self.population if p.best] ### Elimate Nones -- in case M < Number of particles, important for initialb iteratiions
        if self.best:
            bests_to_model.append(self.best)
        if bests_to_model:
            logging.info("Reevaluating")
            bests_to_fitness = self.surrogate_model.predict(bests_to_model)[1]
            for i,part in enumerate([p for p in self.population if p.best]):
                part.best.fitness.values = bests_to_fitness[i]
            if self.best.model:
                best.fitness.values = bests_to_fitness[-1]
                
    # Abstract methods that can be overriden by concrete implementations (optional)
    
    ## is used by the plot.py to add specific meta-heuristic markers onto the design space... currently very primitive. 
    ## check PSOTrial for examples. 
                    
    # Abstract methods that should be overriden by concrete implementations

    def run_initialize(self):
        raise NotImplementedError('Trial is an abstract class, '
                                  'this should not be called.')

    def initialize_population(self):
        raise NotImplementedError('Trial is an abstract class, '
                                  'this should not be called.')

    def filter_population(self):
        raise NotImplementedError('Trial is an abstract class, '
                                  'this should not be called.')
        
    def meta_iterate(self):
        raise NotImplementedError('Trial is an abstract class, '
                                  'this should not be called.')
    
    def terminating_condition(self, fitness):
        return self.fitness.termCond(fitness[0])

    def create_backup_manager(self):
        """
        Returns an instance of a subclass of trialbackup.TrialBackup suited for
        saving and loading a specific Trial subclass.
        """
        raise NotImplementedError('Trial is an abstract class, '
                                  'this should not be called.')
        
    def evaluate_best():
        raise NotImplementedError('Trial is an abstract class, '
                                  'this should not be called.')
                                  
    def sample_design_space(self):
        raise NotImplementedError('Trial is an abstract class, '
                                  'this should not be called.')
        pass
        
class PSOTrial(Trial):

    def run_initialize(self):
        design_space = self.fitness.designSpace

        self.smin = [-1.0 * self.configuration.max_speed *
                (dimSetting['max'] - dimSetting['min'])
                for dimSetting in design_space]
        self.smax = [self.configuration.max_speed *
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
                              conf=self.configuration,
                              designSpace=design_space)
        self.toolbox.register('evaluate', self.fitness_function)

    def initialize_population(self):
        ## the while loop exists, as meta-heuristic makes no sense till we find at least one particle that is within valid region...
        at_least_one_in_valid_region = False
        while (not at_least_one_in_valid_region):
            self.population = self.toolbox.population(
                n=self.configuration.population_size)
            self.toolbox.filter_particles(self.population)

            if self.configuration.eval_correct:
                self.population[0] = creator.Particle(
                    self.fitness.alwaysCorrect())  # This will break
            for i, part in enumerate(self.population):
                if i < self.configuration.F:
                    part.fitness.values, part.code = self.toolbox.evaluate(part)
                    at_least_one_in_valid_region = (part.code == 0) or at_least_one_in_valid_region
            ## add one example till we find something that works
            self.configuration.F = 1

    def meta_iterate(self):
        #TODO - reavluate one random particle... do it.. very important!
        while(self.all_particles_in_invalid_area):
            logging.info("All particles within invalid area... have to randomly sample the design space to find one that is OK...")             
            
        #Update Bests
        for part in self.population:
            if not part.best or self.fitness.is_better(part.fitness, part.best.fitness):
                part.best = creator.Particle(part)
                part.best.fitness.values = part.fitness.values
            if not self.best or self.fitness.is_better(part.fitness, self.best.fitness):
                self.best = creator.Particle(part)
                self.best.fitness.values = part.fitness.values
                self.M_best = self.best
                self.new_best = True
                logging.info('New global best found' + str(self.M_best) + ' fitness:'+ str(part.fitness.values))                
        #PSO
        for part in self.population:
            self.toolbox.update(part, self.counter_dictionary['g'])

    def filter_population(self):
        self.toolbox.filter_particles(self.population)

    def create_backup_manager(self):
        return PSOTrialBackup()

    ### returns a snapshot of the trial state
    def snapshot(self):
        fitness = self.fitness
        best_fitness_array = copy(self.best_fitness_array)
        generations_array = copy(self.generations_array)
        results_folder = copy(self.results_folder)
        images_folder = copy(self.images_folder)
        counter = copy(self.counter_dictionary[self.configuration.counter])
        name = self.get_name()

        
        sm = self.surrogate_model
        if sm.trained():
            classifier = copy(sm.classifier) if hasattr(sm, 'classifier') \
                else None
            regressor = copy(sm.regressor) if hasattr(sm, 'regressor') \
                else None
        else:
            classifier = None
            regressor = None

        return_dictionary = {
            'fitness': fitness,
            'best_fitness_array': best_fitness_array,
            'generations_array': generations_array,
            'results_folder': results_folder,
            'images_folder': images_folder,
            'counter': counter,
            'name': name,
            'classifier': classifier,
            'regressor': regressor,
            'meta_plot': {"particles":{'marker':"o",'data':self.population}}
        }
        return_dictionary.update(copy(self.graph_dictionary))
        return return_dictionary
   
    def evaluate_best(self):        
        if self.new_best:
            self.fitness_function(self.best)
            logging.info('New best was found after M')
        else:
            ## TODO - clean it up...  messy
            logging.info('Best was already evalauted.. adding perturbation')
            perturbation = self.get_perturbation()                        
            perturbed_particle = creator.Particle(self.best)
            for i,val in enumerate(perturbation):
                perturbed_particle[i] = perturbed_particle[i] + val       
            self.toolbox.filter_particle(perturbed_particle)
            fitness, code = self.fitness_function(perturbed_particle) 
            ##check if the value is not a new best
            perturbed_particle.fitness.values = fitness
            if not self.best or self.fitness.is_better(perturbed_particle.fitness, self.best.fitness):
                self.best = perturbed_particle
                self.M_best = self.best
        self.new_best = False
        

    def sample_design_space(self):
        logging.info('Evaluating best perturbation')
        perturbation = self.get_perturbation()                        
        hypercube = self.get_hypercube()
        particle = self.surrogate_model.max_uncertainty()
        if particle is None:
            logging.info('Evaluating a perturbation of a random particle')
            particle = self.toolbox.particle()
        perturbedParticle = creator.Particle(particle)
        for i,val in enumerate(perturbation):
            perturbedParticle[i] = perturbedParticle[i] + val       
        self.toolbox.filter_particle(perturbedParticle)
        self.fitness_function(perturbedParticle) 
        
    ## not used currently
    def get_perturbation_dist(self):
        [max_diag, min_diag] = self.get_dist()
        d = (max_diag - min_diag)/2.0
        if best:
            max_diag = best + d
            for i,dd in enumerate(self.fitness.designSpace):
                max_diag[i] = minimum(max_diag[i],dd["max"])
            min_diag = best - d
            for i,dd in enumerate(self.fitness.designSpace):
                min_diag[i] = maximum(min_diag[i],dd["min"])
            return [max_diag,min_diag]
        else:
            return getHypercube(pop)
            
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
        
    def get_hypercube(self):
        #find maximum
        max_diag = deepcopy(self.population[0])
        for part in self.population:
            max_diag = maximum(part,max_diag)
        ###find minimum vectors
        min_diag = deepcopy(self.population[0])
        for part in self.population:
            min_diag = minimum(part,min_diag)
        return [max_diag,min_diag]
        
    def get_perturbation(self, radius = 10.0):
        [max_diag,min_diag] = self.get_hypercube()
        d = (max_diag - min_diag)/radius
        for i,dd in enumerate(d):
            if self.fitness.designSpace[i]["type"] == "discrete":
                d[i] = maximum(dd,self.fitness.designSpace[i]["step"])
            elif self.fitness.designSpace[i]["type"] == "continuous":
                d[i] = maximum(dd,self.smax[i])
        dimensions = len(self.fitness.designSpace)
        pertubation =  multiply(((rand(1,dimensions)-0.5)*2.0),d)[0] #TODO add the dimensions
        return pertubation
        
        
import logging
import os
import sys
from threading import Thread
import time
from time import strftime
from datetime import datetime

from surrogatemodel import DummySurrogateModel, ProperSurrogateModel
from particles import *
from trialbackup import PSOTrialBackup
from views.plot import Plot_View

from deap import base, creator, tools
from numpy.random import uniform
import wx


class Trial(Thread):

    def __init__(self, trial_no, run_name, fitness, configuration, controller,
                 run_results_folder_path):
        Thread.__init__(self)

        self.name = '{} Trial {}'.format(run_name, trial_no)
        self.run_name = run_name
        self.trial_no = trial_no
        self.fitness = fitness
        self.configuration = configuration
        self.controller = controller
        self.run_results_folder_path = run_results_folder_path
        self.status = 'Running'  # TODO - BAD

        # True if the user has selected to pause the trial
        self.wait = False

        self.graph_dictionary = {
            'rerendering': False,
            'graph_title': configuration.graph_title,
            'graph_names': configuration.graph_names,
            'all_graph_dicts': configuration.all_graph_dicts
        }
        for name in configuration.graph_names:
            self.graph_dictionary['all_graph_dicts'][name]['generate'] = True

        if configuration.plot_view != 'special':
            self.plot_view = Plot_View

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
        self.counter_dictionary['g'] = 1
        self.counter_dictionary['fit'] = 1

        # The last counter value to be visualized, 0 means none
        self.latest_counter_plot = 0

        self.controller.register_trial(self)

        self.backup = self.create_backup_manager()

    def initialise(self):
        """
        Initialises the trial and returns True if everything went OK,
        False otherwise.
        """
        self.results_folder = self._create_results_folder()
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

        return True

    def run(self):
        self.start_time = datetime.now().strftime('%d-%m-%Y  %H:%M:%S')
        self.view_update()

        logging.info('{} started'.format(self.get_name()))

        logging.info('Run prepared... executing')

        while self.counter_dictionary['g'] < self.GEN + 1:
            logging.info('[{}] Generation {}'.format(
                self.get_name(), self.counter_dictionary['g']))

            # Roll population
            first_pop = self.population.pop(0)
            self.population.append(first_pop)

            # Initialise termination check
            self.check = False

            if self.counter_dictionary['fit'] > self.max_fitness:
                logging.info('Fitness counter exceeded the limit... exiting')
                break

            # Train surrogate model
            while not self.surrogate_model.train(self.population):
                logging.info('Re-initializing population')
                self.initialize_population()

            code, mean, variance = \
                self.surrogate_model.model_particles(self.population)

            self.post_model_filter(self.population, code, mean, variance)
            #logging.info(self.population[0].fitness.values)

            # Iteration of meta-heuristic
            self.meta_iterate()
            self.filter_population()

            # TODO: termination condition

            #TODO: This will be done after the presentation
            '''
            if self.counter_dictionary['g'] % self.configuration.M == 0:
                if self.M_best == self.best:
                    #Return best?
                    logging.info('New best was found after M')
                    self.M_best = self.fitness.worst_value;
                    pass
                else:
                    logging.info('Perturbing things')
                    #Return perturbation?
                    pass
            '''

            # Wait until the user unpauses the trial.
            while self.wait:
                time.sleep(0)

            self.increment_counter('g')
            self.view_update()

        self.status = 'Finished'
        self.view_update()
        logging.info('{} finished'.format(self.get_name()))

    def _create_results_folder(self):
        """
        Creates a folder used for storing results.
        Returns a results folder path or None if it could not be created.
        """
        path = '{}/trial-{}'.format(self.run_results_folder_path,
                                    self.trial_no)

        try:
            os.makedirs(path)
            return path
        except OSError, e:
            # Folder already exists
            return path
        except Exception, e:
            logging.error('Could not create folder: {}, aborting'.format(path),
                          exc_info=sys.exc_info())
            return None

    # Do not know what addReturn is???
    def fitness_function(self, part):
        fitness, code, addReturn = self.fitness.fitnessFunc(part)
        self.surrogate_model.add_training_instance(part, code, fitness)
        self.increment_counter('fit')
        return fitness, code

    # TODO - this could be removed
    def get_name(self):
        return self.name

    def view_update(self):
        self.controller.view_update(self)

    def increment_counter(self, counter):
        if counter == self.configuration.counter:
            self.best_fitness_array.append(self.best.fitness.values[0])
            self.generations_array.append(self.counter_dictionary[counter])
            self.save()

            if self.counter_dictionary[counter] % \
                    self.configuration.vis_every_X_steps == 0:

                logging.debug('counter {}'.format(
                    self.counter_dictionary[counter]))
                self.controller.visualize_trial(self)

        self.counter_dictionary[counter] += 1

    def save(self):
        self.backup.save_trial(self)

    def load(self, generation=None):
        return self.backup.load_trial(self, generation)

    def post_model_filter(self, population, code, mean, variance):
        for i, (p, c, m, v) in enumerate(zip(population, code,
                                             mean, variance)):
            if v > self.configuration.max_stdv and c == 0:
                p.fitness.values, p.code = self.toolbox.evaluate(p)
            else:
                if c == 0:
                    p.fitness.values = m
                else:
                    p.fitness.values = [self.fitness.worst_value]

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

    def create_backup_manager(self):
        """
        Returns an instance of a subclass of trialbackup.TrialBackup suited for
        saving and loading a specific Trial subclass.
        """
        raise NotImplementedError('Trial is an abstract class, '
                                  'this should not be called.')


class PSOTrial(Trial):

    def run_initialize(self):
        design_space = self.fitness.designSpace

        smin = [-1.0 * self.configuration.max_speed *
                (dimSetting['max'] - dimSetting['min'])
                for dimSetting in design_space]
        smax = [self.configuration.max_speed *
                (dimSetting['max'] - dimSetting['min'])
                for dimSetting in design_space]

        creator.create('FitnessMax', base.Fitness, weights=(1.0,))
        creator.create('Particle', list, fitness=creator.FitnessMax,
                       smin=smin, smax=smax,
                       speed=[uniform(smi, sma) for sma, smi in zip(smax,
                                                                    smin)],
                       pmin=[dimSetting['max'] for dimSetting in design_space],
                       pmax=[dimSetting['min'] for dimSetting in design_space],
                       model=False, best=None, code=None)

        self.toolbox = base.Toolbox()
        self.toolbox.register('particle', generate, designSpace=design_space)
        self.toolbox.register('filter_particles', filterParticles,
                              designSpace=design_space)
        self.toolbox.register('population', tools.initRepeat,
                              list, self.toolbox.particle)
        self.toolbox.register('update', updateParticle, trial=self,
                              conf=self.configuration,
                              designSpace=design_space)
        self.toolbox.register('evaluate', self.fitness_function)

        ### TODO - do something about the commented out code
        #                 fitness=fitness)
        #self.toolbox.register('train_model', trainModel,
        #                 maxstdv=self.configuration.max_stdv,
        #                 conf=self.configuration)

        ### TODO: not sure if these are necessary
        ###       (just print out stats - will visualise do this?)
        #self.stats = tools.Statistics(lambda ind: ind.fitness.values)
        #self.stats.register('Avg', tools.mean)
        #self.stats.register('Std', tools.std)
        #self.stats.register('Min', min)
        #self.stats.register('Max', max)

    def initialize_population(self):
        self.population = self.toolbox.population(
            n=self.configuration.population_size)
        self.toolbox.filter_particles(self.population)

        if self.configuration.eval_correct:
            self.population[0] = creator.Particle(
                self.fitness.alwaysCorrect())  # This will break
        for i, part in enumerate(self.population):
            if i < self.configuration.F:
                part.fitness.values, part.code = self.toolbox.evaluate(part)

    def meta_iterate(self):
        #Update Bests
        for part in self.population:
            if not part.best or self.fitness.is_better(part.fitness,
                                                       part.best.fitness):
                part.best = creator.Particle(part)
                part.best.fitness.values = part.fitness.values
            if not self.best or self.fitness.is_better(part.fitness,
                                                       self.best.fitness):
                self.best = creator.Particle(part)
                self.best.fitness.values = part.fitness.values
                self.M_best = self.best

        #PSO
        for part in self.population:
            self.toolbox.update(part, self.counter_dictionary['g'])

    def filter_population(self):
        self.toolbox.filter_particles(self.population)

    def create_backup_manager(self):
        return PSOTrialBackup()

    def set_counter_plot(self, counter_plot):
        self.latest_counter_plot = max(counter_plot, self.latest_counter_plot)
        self.controller.view_graph_update(self, counter_plot)

from collections import OrderedDict
import copy
import csv
import logging
import re
import os
import pickle

from ..surrogatemodels.surrogatemodel import DummySurrogateModel, ProperSurrogateModel

from deap import creator


class TrialBackup(object):
    """
    Saves and loads an instance of a Trial's subclass. Necessary for
    restarting trials.
    """

    def save_trial(self, trial):
        """
        Returns True if successfully executed.
        """
        raise NotImplementedError('TrialBackup is an abstract class, '
                                  'this should not be called.')

    def load_trial(self, trial, generation=None):
        """
        Loads trial at the given (0-based) generation or at the last
        generation if generation is None.
        """
        raise NotImplementedError('TrialBackup is an abstract class, '
                                  'this should not be called.')


class PSOTrialBackup(TrialBackup):

    # Number of digits for generations used in file names
    GEN = 10

    field_names = OrderedDict([
        ('name', None),
        ('fitness_evaluated', None),
        ('model_failed', None),
        ('model_retrained', None),
        ('new_best_over_iteration', None),
        ('svc_training_labels', None),
        ('svc_training_set', None),
        ('latest_counter_plot', None)
    ])
    graph_names = OrderedDict([
        ('graph', None),
    ])
    population_field_names = OrderedDict([
        ('particle-0', None),
        ('particle-1', None),
        ('particle-fitness', None),
        ('best', None),
        ('best-fitness', None),
        ('pmax', None),
        ('pmin', None),
        ('smax', None),
        ('smin', None),
        ('speed', None)
    ])

    def save_trial(self, trial):
        # Save fields of this Trial
        trial_file = ('{}/{:0' + str(PSOTrialBackup.GEN) +
                      'd}.csv').format(trial.results_folder,
                                       trial.counter_dictionary['g'])
        with open(trial_file, 'wb') as csv_file:
            results_writer = csv.DictWriter(csv_file,
                                            PSOTrialBackup.field_names)
            results_writer.writeheader()

            results_writer.writerow({
                'name': trial.name,
                'fitness_evaluated': trial.fitness_evaluated,
                'model_failed': trial.model_failed,
                'model_retrained': trial.model_retrained,
                'new_best_over_iteration': trial.new_best_over_iteration,
                'svc_training_labels': trial.svc_training_labels,
                'svc_training_set': trial.svc_training_set}) 

        # Save the surrogate model
        sm_file = ('{}/{:0' + str(PSOTrialBackup.GEN) +
                   'd}-sm.txt').format(trial.results_folder,
                                       trial.counter_dictionary['g'])
        with open(sm_file, 'w') as txt_file:
            pickle.dump(trial.surrogate_model, txt_file)

        # http://www.youtube.com/watch?v=GvjGrWERM1U
        population_file = ('{}/{:0' + str(PSOTrialBackup.GEN) +
                           'd}-population.csv').format(
                               trial.results_folder,
                               trial.counter_dictionary['g'])
        with open(population_file, 'wb') as csv_file:
            results_writer = csv.DictWriter(
                csv_file,
                PSOTrialBackup.population_field_names)
            results_writer.writeheader()

            # Save trial.best as the first particle
            particles = [trial.best]
            particles.extend(trial.population)

            for particle in particles:
                results_writer.writerow({
                    'particle-0': particle[0],
                    'particle-1': particle[1],
                    'particle-fitness': pickle.dumps(particle.fitness),
                    'best': '{}:{}'.format(particle.best[0], particle.best[1])
                    if particle.best
                    else None,
                    'best-fitness': pickle.dumps(particle.best.fitness)
                    if particle.best
                    else None,
                    'pmax': particle.pmax,
                    'pmin': particle.pmin,
                    'smax': particle.smax,
                    'smin': particle.smin,
                    'speed': particle.speed
                })

        # Save graph_dictionary, this is best pickled
        graph_file = '{}/graph.csv'.format(trial.results_folder)
        with open(graph_file, 'wb') as csv_file:
            results_writer = csv.DictWriter(csv_file,
                                            PSOTrialBackup.graph_names)
            results_writer.writeheader()

            results_writer.writerow({
                'graph': pickle.dumps(trial.graph_dictionary)
            })

        last_generation_file = ('{:0' + str(PSOTrialBackup.GEN) +
                                'd}').format(trial.counter_dictionary['g'])
        best_fitness_file = '{}/{}-best-fitness.txt'.format(
            trial.results_folder,
            last_generation_file)
        with open(best_fitness_file, 'w') as txt_file:
            for best_fitness in trial.best_fitness_array:
                txt_file.write(str(best_fitness))
                txt_file.write('\n')

    def load_trial(self, trial, generation=None):
        if generation is None:
            # Figure out what the last generation before crash was
            found = False
            for filename in reversed(os.listdir(trial.results_folder)):
                match = re.search(r'^(\d+)\.csv', filename)
                if match:
                    # Found the last generation
                    generation = int(match.group(1))
                    found = True
                    break

            if not found:
                return False

        trial.counter_dictionary['g'] = generation
        generation_file = ('{:0' + str(PSOTrialBackup.GEN) +
                           'd}').format(generation)

        # Load fields of this Trial
        trial_file = '{}/{}.csv'.format(trial.results_folder, generation_file)
        with open(trial_file, 'rb') as csv_file:
            results_reader = csv.DictReader(csv_file,
                                            PSOTrialBackup.field_names)
            results_reader.next()  # Skip the header

            for row in results_reader:
                trial.name = row['name']
                trial.fitness_evaluated = row['fitness_evaluated']
                trial.model_failed = row['model_failed']
                trial.model_retrained = row['model_retrained']
                trial.new_best_over_iteration = row['new_best_over_iteration']
                trial.svc_training_labels = row['svc_training_labels']
                trial.svc_training_set = row['svc_training_set']
                break

        # Load the surrogate model
        sm_file = ('{}/{:0' +
                   str(PSOTrialBackup.GEN) +
                   'd}-sm.txt').format(trial.results_folder,
                                       trial.counter_dictionary['g'])
        with open(sm_file, 'r') as txt_file:
            sm = pickle.load(txt_file)
            sm.fitness = trial.fitness
            sm.configuration = trial.configuration
            sm.regressor.controller = trial.controller
            trial.surrogate_model = sm

        # Load the population
        population_file = '{}/{}-population.csv'.format(trial.results_folder,
                                                        generation_file)
        try:
            with open(population_file, 'rb') as f:
                pass
        except:
            logging.error('Could not load the population file')
            return False

        trial.population = []
        with open(population_file, 'rb') as csv_file:
            results_reader = csv.DictReader(
                csv_file,
                PSOTrialBackup.population_field_names)
            results_reader.next()  # Skip the header

            for row in results_reader:
                particle = self.create_particle()
                particle[0] = float(row['particle-0'])
                particle[1] = float(row['particle-1'])
                particle.fitness = pickle.loads(row['particle-fitness'])
                if row['best']:
                    particle.best = self.create_particle()
                    best = row['best'].split(':')
                    particle.best[0] = float(best[0])
                    particle.best[1] = float(best[1])
                    particle.best.fitness = pickle.loads(row['best-fitness'])
                else:
                    particle.best = None
                particle.pmax = self.str_array_to_float_array(row['pmax'])
                particle.pmin = self.str_array_to_float_array(row['pmin'])
                particle.smax = self.str_array_to_float_array(row['smax'])
                particle.smin = self.str_array_to_float_array(row['smin'])
                particle.speed = self.str_array_to_float_array(row['speed'])

                trial.population.append(particle)

        # Load graph_dictionary
        graph_file = '{}/graph.csv'.format(trial.results_folder)
        with open(graph_file, 'rb') as csv_file:
            results_reader = csv.DictReader(csv_file,
                                            PSOTrialBackup.graph_names)
            results_reader.next()  # Skip the header

            for row in results_reader:
                trial.graph_dictionary = pickle.loads(row['graph'])
                break

        best_fitness_file = '{}/{}-best-fitness.txt'.format(
            trial.results_folder,
            generation_file)
        with open(best_fitness_file, 'r') as txt_file:
            trial.best_fitness_array = []
            for line in txt_file:
                best_fitness = float(line.rstrip('\n'))
                trial.best_fitness_array.append(best_fitness)

        # The first particle stored in the file was trial.best
        trial.best = trial.population.pop(0)

        trial.generations_array = range(1, trial.counter_dictionary['g'] + 1)

        return True

    def create_particle(self):
        """
        Helper method for creating DEAP particles. A little hacky.
        """
        return creator.Particle(['O', 'HAI'])  # This makes it work :)

    def str_array_to_float_array(self, str_array):
        """
        Converts a string like "[1.2, 3.4]" into an array [1.2, 3.4].
        """
        return map(float, str_array[1:-1].split(','))

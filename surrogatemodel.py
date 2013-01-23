import logging

from classifiers import Classifier, SupportVectorMachineClassifier
from regressors import Regressor, GaussianProcessRegressor

from helper_functions import numpy_array_index

class SurrogateModel(object):

    def __init__(self, fitness, configuration, controller):
        self.fitness = fitness
        self.configuration = configuration
        self.classifier = Classifier()
        self.regressor = Regressor(controller)        
        
    def train(self, pop):
        raise NotImplementedError('SurrogateModel is an abstract class, this '
                                  'should not be called.')

    def model_particles(self, particles):
        MU, S2 = self.regressor.predict(particles)
        return self.classifier.predict(particles), MU, S2

    def add_training_instance(self, part, code, fitness):
        pass
        
    def contains_training_instance(self, part):
        pass    
        
    def get_training_instance(self, part):
        pass

    def __getstate__(self):
        # Don't pickle fitness and configuration
        d = dict(self.__dict__)
        del d['fitness']
        del d['configuration']
        return d

    def contains_particle(self, part):
        pass
        
    def particle_value(self, part):
        pass

class DummySurrogateModel(SurrogateModel):

    def train(self, pop):
        return True

    def model_particle(self, particle):
        return 0, 0, 0
        
    def contains_training_instance(self, part):
        return False


class ProperSurrogateModel(SurrogateModel):

    def __init__(self, fitness, configuration, controller):
        super(ProperSurrogateModel, self).__init__(fitness, configuration,
                                                   controller)

        if configuration.classifier == 'SupportVectorMachine':
            self.classifier = SupportVectorMachineClassifier()
        else:
            logging.error('Classifier type {} not found'.format(
                configuration.classifier))

        if configuration.regressor == 'GaussianProcess':
            self.regressor = GaussianProcessRegressor(controller)
        else:
            logging.error('Regressor type {} not found'.format(
                configuration.regressor))

    def train(self, pop):
        dimensions = self.fitness.dimensions
        return self.classifier.train(pop) and self.regressor.train(
            pop, self.configuration, dimensions)

    def add_training_instance(self, part, code, fitness, addReturn):
        self.classifier.add_training_instance(part, code)
        if addReturn == 0: ## only update regressor if the fitness function produced a result
            self.regressor.add_training_instance(part, fitness)
        
    def contains_training_instance(self, part):
        self.classifier.contains_training_instance(part) 
        self.regressor.contains_training_instance(part)
        return False 

    def get_training_instance(self, part):
        code = self.classifier.get_training_instance(part) 
        fitness = self.fitness.worst_value
        if self.regressor.contains_training_instance(part):
            fitness = self.regressor.get_training_instance(part)            
        return code, fitness


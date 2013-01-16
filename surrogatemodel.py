import logging

from classifiers import Classifier, SupportVectorMachineClassifier
from regressors import Regressor, GaussianProcessRegressor


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

    def __getstate__(self):
        # Don't pickle fitness and configuration
        d = dict(self.__dict__)
        del d['fitness']
        del d['configuration']
        return d


class DummySurrogateModel(SurrogateModel):

    def train(self, pop):
        return True

    def model_particle(self, particle):
        return 0, 0, 0


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

    def add_training_instance(self, part, code, fitness):
        self.classifier.add_training_instance(part, code)
        self.regressor.add_training_instance(part, fitness)

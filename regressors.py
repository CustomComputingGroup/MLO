import logging
from multiprocessing import Process, Pipe
import traceback

from particles import *

from numpy import unique, asarray, bincount, array, append, sqrt
from sklearn import preprocessing
from sklearn.gaussian_process import GaussianProcess
from sklearn.grid_search import GridSearchCV

from helper_functions import numpy_array_index

#TODO - abstract class
class Regressor(object):

    def __init__(self, controller):
        self.training_set = None
        self.training_fitness = None
        self.regr = None
        self.controller = controller

    def train(self, z):
        return True

    def predict(self, z):
        output = []
        for input_vector in z:
            output.append([0])
        return output, output

    def add_training_instance(self, part, fitness):
        if self.training_set is None:
            self.training_set = array([part])
            self.training_fitness = array([fitness])
        elif part not in self.training_set:
            contains = self.contains_training_instance(self.training_set)
            if contains:
                logging.info('A particle duplicate is being added.. check your code!!')
            else:
                self.training_set = append(self.training_set, [part], axis=0)
                self.training_fitness = append(self.training_fitness, [fitness],
                                               axis=0)

    def contains_training_instance(self, part):
        contains, index = numpy_array_index(self.training_set, part)
        return contains
            
    def get_training_instance(self, part):
        contains, index = numpy_array_index(self.training_set, part)
        if self.training_set is None:
            logging.error('cannot call get_training_instance if training_set is empty')
            return False
        elif contains:
            return self.training_fitness[index]
        else :
            logging.error('cannot call get_training_instance if training_set does not contain the particle')
            return False
                                           
    def __getstate__(self):
        # Don't pickle controller
        d = dict(self.__dict__)
        del d['controller']
        return d


class GaussianProcessRegressor(Regressor):

    def _init_(self, controller):
        super(GaussianProcessRegressor, self).__init__(controller)
        self.input_scaler = None
        self.output_scaler = None

    def train(self, pop, conf, dimensions):
        try:
            # Scale inputs and particles?
            self.input_scaler = preprocessing.Scaler().fit(self.training_set)
            scaled_training_set = self.input_scaler.transform(
                self.training_set)

            # Scale training data
            self.output_scaler = preprocessing.Scaler(with_std=False).fit(
                self.training_fitness)
            adjusted_training_fitness = self.output_scaler.transform(
                self.training_fitness)

            if conf.nugget == 0:
                gp = GaussianProcess(regr=conf.regr, corr=conf.corr,
                                     theta0=array([conf.theta0] * dimensions),
                                     thetaL=array([conf.thetaL] * dimensions),
                                     thetaU=array([conf.thetaU] * dimensions),
                                     random_start=300)
            else:
                gp = GaussianProcess(regr=conf.regr, corr=conf.corr,
                                     theta0=array([conf.theta0] * dimensions),
                                     thetaL=array([conf.thetaL] * dimensions),
                                     thetaU=array([conf.thetaU] * dimensions),
                                     random_start=300, nugget=conf.nugget)

            # Start a new process to fit the data to the gp, because gp.fit is
            # not thread-safe
            parent_end, child_end = Pipe()

            self.controller.acquire_training_sema()
            logging.info('Training regressor')
            p = Process(target=self.fit_data, args=(gp, scaled_training_set,
                                                    adjusted_training_fitness,
                                                    child_end))
            p.start()
            self.regr = parent_end.recv()
            logging.info('Regressor training successful')
            self.controller.release_training_sema()
            return True
        except Exception, e:
            logging.error('Regressor training failed.. {}'.format(e))
            return False

    def predict(self, z):
        try:
            #logging.debug(z)
            MU, S2 = self.regr.predict(self.input_scaler.transform(array(z)),
                                       eval_MSE=True)
            #logging.debug(MU)
            MU = self.output_scaler.inverse_transform(MU)
            MU = MU.reshape(-1, 1)
            S2 = sqrt(S2.reshape(-1, 1))
            return MU, S2
        except Exception, e:
            logging.error('Prediction failed.. {}'.format(e))
            return None

    def fit_data(self, gp, scaled_training_set, adjusted_training_fitness,
                 child_end):
        gp.fit(scaled_training_set, adjusted_training_fitness)
        child_end.send(gp)

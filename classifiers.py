import logging
import traceback

from particles import *

from numpy import unique, asarray, bincount, array, append, arange
from sklearn import preprocessing, svm
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedKFold


#TODO - should this be an abstract class instead?
class Classifier(object):

    def __init__(self):
        self.training_set = []
        self.training_labels = []
        self.clf = None

    def train(self, z):
        return True

    def predict(self, z):
        output = []
        for input_vector in z:
            output.append(0)
        output = array(output)
        return output

    def add_training_instance(self, part, label):
        if self.training_set == []:
            self.training_set = array([part])
            self.training_labels = array([label])
        else:
            self.training_set = append(self.training_set, [part], axis=0)
            self.training_labels = append(self.training_labels, [label],
                                          axis=0)


class SupportVectorMachineClassifier(Classifier):

    def train(self, pop):
        try:
            inputScaler = preprocessing.Scaler().fit(self.training_set)
            scaledSvcTrainingSet = inputScaler.transform(self.training_set)

            if len(unique(asarray(self.training_labels))) < 2:
                self.clf = svm.OneClassSVM()
                self.clf.fit(scaledSvcTrainingSet)
                return False
            else:
                param_grid = {
                    'gamma': 10.0 ** arange(-5, 4),
                    'C':     10.0 ** arange(-2, 9)}

                self.clf = GridSearchCV(svm.SVC(), param_grid=param_grid,
                                        cv=StratifiedKFold(
                                            y=self.training_labels.reshape(-1),
                                            k=2))

                self.clf.fit(scaledSvcTrainingSet,
                             self.training_labels.reshape(-1))
                self.clf = self.clf.best_estimator_
                logging.info('Classifier training successful')
                return True
        except Exception, e:
            logging.error('Training failed.. {}'.format(e))
            return False

    def predict(self, z):
        try:
            # Scale inputs and particles
            inputScaler = preprocessing.Scaler().fit(self.training_set)

            scaledz = inputScaler.transform(z)
            zClass = self.clf.predict(scaledz)
            return zClass
        except Exception, e:
            logging.error('Prediction failed.. {}'.format(e))
            return None

import logging
import traceback

from numpy import unique, asarray, bincount, array, append, arange
from sklearn import preprocessing, svm
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedKFold

from utils import numpy_array_index

#TODO - should this be an abstract class instead?
class Classifier(object):

    def __init__(self):
        self.training_set = None
        self.training_labels = None
        self.clf = None

    def train(self, z):
        return True

    def predict(self, z):
        output = []
        for input_vector in z:
            output.append(0)
        output = array(output)
        return output

    ## TODO - check if element is in the array... just for the sake of it
    def add_training_instance(self, part, label):
        if self.training_set is None:
            self.training_set = array([part])
            self.training_labels = array([label])
        else:
            contains = self.contains_training_instance(self.training_set)
            if contains:
                logging.info('A particle duplicate is being added.. check your code!!')
            else:
                self.training_set = append(self.training_set, [part], axis=0)
                self.training_labels = append(self.training_labels, [label],
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
            return self.training_labels[index]
        else :
            logging.error('cannot call get_training_instance if training_set does not contain the particle')
            return False


class SupportVectorMachineClassifier(Classifier):

    def train(self, pop):
        try:
            inputScaler = preprocessing.Scaler().fit(self.training_set)
            scaledSvcTrainingSet = inputScaler.transform(self.training_set)

            if len(unique(asarray(self.training_labels))) < 2:
                logging.info('Only one class encountered, we do not need to use a classifier')
                #self.clf = svm.OneClassSVM()
                #self.clf.fit(scaledSvcTrainingSet)
                
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
            logging.error('Classifier training failed.. {}'.format(e))
            return False

    def predict(self, z):
        try:
            if len(unique(asarray(self.training_labels))) < 2:
                ## TODO - rewrite it not to use a stupid loop...
                return array([self.training_labels[0][0]] * len(z))
            else:
                # Scale inputs and particles
                inputScaler = preprocessing.Scaler().fit(self.training_set)

                scaledz = inputScaler.transform(z)
                zClass = self.clf.predict(scaledz)
                return zClass
        except Exception, e:
            logging.error('Prediction failed.. {}'.format(e))
            return None

import logging
import traceback

from numpy import unique, asarray, bincount, array, append, arange
from sklearn import preprocessing, svm
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedKFold, KFold
from copy import deepcopy

from utils import numpy_array_index

from pickle import dumps

#TODO - should this be an abstract class instead?
class Classifier(object):

    def __init__(self):
        self.training_set = None
        self.training_labels = None
        self.clf = None
        self.oneclass = False

    def train(self):
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

    ###############
    ### GET/SET ###
    ###############
        
    def get_state_dictionary(self):
        dict = {'training_set' : self.training_set,
                'training_labels': self.training_labels}
        return dict
        
    def set_state_dictionary(self, dict):
        self.training_set = dict['training_set']
        self.training_labels = dict['training_labels']

    def get_parameter_string(self):
        return "Not implemented"
        
class SupportVectorMachineClassifier(Classifier):

    def train(self):
        try:
            inputScaler = preprocessing.StandardScaler().fit(self.training_set)
            scaledSvcTrainingSet = inputScaler.transform(self.training_set)
            all_labels = unique(asarray(self.training_labels))
            class_weights = dict([(i,1.0)for i in all_labels])
            class_weights[0] = 2.0
            if len(unique(asarray(self.training_labels))) < 2:
                logging.info('Only one class encountered, we do not need to use a classifier')
                #self.clf = svm.OneClassSVM()
                #self.clf.fit(scaledSvcTrainingSet)
                self.oneclass = True
            else:
                self.oneclass = False
                param_grid = {
                    'gamma': 1.3 ** arange(-10, 10),
                    'C':     1.05 ** arange(-10, 10)
                    }
                try:
                    '''
                    try:
                        self.type = 2
                        self.clf = GridSearchCV(svm.SVC(class_weight = "auto"), param_grid=param_grid,
                                        cv=StratifiedKFold(
                                            y=self.training_labels.reshape(-1),
                                        n_folds=len(self.training_labels)))
                        self.clf.fit(scaledSvcTrainingSet, self.training_labels.reshape(-1))
                    except: ##in case when we cannot construct equal proportion folds
                    '''
                    self.type = 1
                    logging.debug('Using KFold cross validation for classifier training')
                    self.clf = GridSearchCV(svm.SVC(class_weight=class_weights), param_grid=param_grid,
                                            cv=KFold(n=self.training_labels.shape[0],n_folds=self.training_labels.shape[0]))
                    self.clf.fit(scaledSvcTrainingSet, self.training_labels.reshape(-1))
                    self.clf = self.clf.best_estimator_ ## gridsearch cant be pickled...
                    #logging.info(str(self.training_labels.shape[0])))
                except Exception, e:## in case for example when we have single element of a single class, cant construct two folds
                    self.type = 0
                    logging.debug('One of the classes has only one element, cant use cross validation:' + str(e))
                    self.clf = svm.SVC(kernel='rbf', gamma=1., C = 1., class_weight=class_weights)
                    self.clf.fit(scaledSvcTrainingSet, self.training_labels.reshape(-1))
                logging.info('Classifier training successful')
            return True
        except Exception, e:
            logging.error('Classifier training failed.. {}'.format(e))
            return False

    def predict(self, z):
        try:
            if self.oneclass:
                ## TODO - rewrite it not to use a stupid loop...
                return array([self.training_labels[0][0]] * len(z))
            else:
                # Scale inputs and particles
                inputScaler = preprocessing.StandardScaler().fit(self.training_set)
                scaledz = inputScaler.transform(z)
                zClass = self.clf.predict(scaledz)
                for i,zz in enumerate(z):
                    if self.contains_training_instance(zz):
                        zClass[i]=self.get_training_instance(zz)
                return zClass
        except Exception, e:
            logging.error('Prediction failed... ' + str(e))
            return None
            
    def get_parameter_string(self):
        try:
            return str(self.clf.gamma) + "_" + str(self.clf.C)
        except:
            return "N\A"
            
    ## TODO - come up with a smart way of storing these...
    def get_state_dictionary(self):
        '''
        if self.clf is None:
            clf = None
            self.type = None
        else:
            #try:
                if self.type == 0 :
                    clf = deepcopy(self.clf.get_params(deep=True))
                    logging.info(str(clf))
                else:
                    clf = deepcopy(self.clf.best_estimator_.get_params(deep=True))
                    logging.info(str(clf))
            #except:
            #    self.type = None
        ''' 
        dict = {'training_set' : self.training_set,
                'training_labels': self.training_labels,
                'oneclass': self.oneclass,
                'clf': deepcopy(self.clf)}

        return dict
        
    ###
    def set_state_dictionary(self, dict):
        self.training_set = dict['training_set']
        self.training_labels = dict['training_labels']
        self.oneclass = dict['oneclass']
        self.clf = dict['clf']
        '''
        self.type = dict['type']

        #try:
        self.clf = svm.SVC()
        self.clf.set_params(**dict['clf'])
        #except:
        #    self.clf = None
        '''

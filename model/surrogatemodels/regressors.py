import logging
from multiprocessing import Process, Pipe
import traceback
import os
from random import randrange
import shutil
import time

from numpy import unique, asarray, bincount, array, append, sqrt, log, sort, exp, isinf, all, sum
from sklearn import preprocessing
from sklearn.gaussian_process import GaussianProcess
from sklearn.grid_search import GridSearchCV
from sklearn.cluster import KMeans
from sklearn import mixture
from sklearn.cross_validation import ShuffleSplit
from scipy.spatial.distance import euclidean
from scipy.stats import norm
from numpy.random import uniform, shuffle, permutation
#from rpy2.robjects.packages import importr
#from rpy2.robjects import r
#from rpy2.robjects.vectors import FloatVector
import gc 

from utils import numpy_array_index
from copy import deepcopy


## pyGPR
from UTIL.utils import hyperParameters
from Tools.min_wrapper import min_wrapper
from GPR.gp import gp
###old pyXGPR
##from GPR import gpr

#TODO - abstract class
class Regressor(object):

    def __init__(self, controller, conf):
        self.training_set = None
        self.training_fitness = None
        self.regr = None
        self.controller = controller
        self.conf = conf

    def get_y_best(self):
        if self.conf.goal == "min":
            return min(self.training_fitness)
        elif self.conf.goal == "max":
            return max(self.training_fitness)
            
    def train(self):
        return True

    def predict(self, z):
        output = []
        output2 = []
        for input_vector in z:
            output.append([0.0])
            output2.append([100.0])
        return output, output2

    def shuffle(self):
        p = permutation(len(self.training_fitness))
        self.training_fitness=self.training_fitness[p]
        self.training_set=self.training_set[p]

    def add_training_instance(self, part, fitness):
        if self.training_set is None:
            self.training_set = array([part])
            self.training_fitness = array([fitness])
        else:
            contains = self.contains_training_instance(self.training_set)
            if contains:
                logging.debug('A particle duplicate is being added.. check your code!!')
            else:
                self.training_set = append(self.training_set, [part], axis=0)
                self.training_fitness = append(self.training_fitness, [fitness], axis=0)

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
                                           
    def get_training_set(self):
        return self.training_set

    ## TODO - do it so it would be done in a numpy-ish way
    def e_impr(self, s2, y_mean, y_best):
        if self.conf.goal == "min":
            y_diff_vector = y_best-y_mean
        elif self.conf.goal == "max":
            y_diff_vector = y_mean - y_best
        y_diff_vector_over_s2 = y_diff_vector / s2
        result =  (y_diff_vector) * norm.cdf(y_diff_vector_over_s2) + s2 * norm.pdf(y_diff_vector_over_s2)
        result = array([[0.0] if isinf(x) else x for x in result])
        return result
        
    def prob(self, s2, y_mean, y_best):
        if self.conf.goal == "min":
            return norm.cdf(y_best, y_mean, s2)
        elif self.conf.goal == "max":
            return 1.0-norm.cdf(y_best, y_mean, s2)
           
    def sample(self, s2, y_mean, y_best):
        return norm.rvs(y_mean, s2)
            
    def get_training_fitness(self):
        return self.training_fitness
    # def __getstate__(self):
        # Don't pickle controller
        # d = dict(self.__dict__)
        # del d['controller']
        # return d
        
    def get_nlml(self):
        return None
        
    def training_set_empty(self):
        return (self.training_set is None)
        
    def get_parameter_string(self):
        return "Not Implemented"
        
    ###############
    ### GET/SET ###
    ###############
        
    def get_state_dictionary(self):
        raise NotImplementedError('Trial is an abstract class, '
                                  'this should not be called.')
        
    def set_state_dictionary(self, dict):
        raise NotImplementedError('Trial is an abstract class, '
                                  'this should not be called.')
        
class GaussianProcessRegressor(Regressor):

    def __init__(self, controller, conf):
        super(GaussianProcessRegressor, self).__init__(controller, conf)
        self.input_scaler = None
        self.output_scaler = None
        self.conf = conf
        self.gp = None

    def regressor_countructor(self):
        conf = self.conf
        dimensions = len(self.training_set[0])
        if conf.nugget == 0:
            gp = GaussianProcess(regr=conf.regr, corr=conf.corr2,
                                 theta0=array([conf.theta0] * dimensions),
                                 thetaL=array([conf.thetaL] * dimensions),
                                 thetaU=array([conf.thetaU] * dimensions),
                                 random_start=conf.random_start)
        else:
            gp = GaussianProcess(regr=conf.regr, corr=conf.corr2,
                                 theta0=array([conf.theta0] * dimensions),
                                 thetaL=array([conf.thetaL] * dimensions),
                                 thetaU=array([conf.thetaU] * dimensions),
                                 random_start=conf.random_start, nugget=conf.nugget)
        return gp

    def train(self):
        conf = self.conf
        if len(self.training_set) == 0:
            return True
        try:
            # Scale inputs and particles?
            self.input_scaler = preprocessing.StandardScaler().fit(self.training_set)
            scaled_training_set = self.input_scaler.transform(
                self.training_set)

            # Scale training data
            self.output_scaler = preprocessing.StandardScaler(with_std=False).fit(
                self.training_fitness)
            adjusted_training_fitness = self.output_scaler.transform(
                self.training_fitness)
            gp = self.regressor_countructor()
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
            if self.regr is None:
                raise Exception("Something went wrong with the regressor")
            else:
                logging.info('Regressor training successful')
                self.controller.release_training_sema()
                self.gp = gp
                return True
        except Exception, e:
            logging.info('Regressor training failed.. retraining.. ' + str(e))
            return False

    def predict(self, z):
        try:
            #logging.info("z " + str(z))
            #logging.info("z.shape " + str(z.shape))
            # Scale inputs. it allows us to realod the regressor not retraining the model
            self.input_scaler = preprocessing.StandardScaler().fit(self.training_set)
            self.output_scaler = preprocessing.StandardScaler(with_std=False).fit(
                self.training_fitness) 
                
            #logging.debug(z)
            MU, S2 = self.regr.predict(self.input_scaler.transform(array(z)), eval_MSE=True)
            #logging.debug(MU)
            S2 = sqrt(S2.reshape(-1, 1))
            MU = MU.reshape(-1, 1)
            
            y_best = self.output_scaler.transform(self.get_y_best())
            EI = self.e_impr(S2, MU, y_best)
            P = self.output_scaler.inverse_transform(self.sample(S2, MU, y_best))
            
            MU = self.output_scaler.inverse_transform(MU)
            
            return MU, S2, EI, P
        except Exception, e:
            logging.error('Prediction failed.... ' + str(e))
            return None, None, None, None

    def fit_data(self, gp, scaled_training_set, adjusted_training_fitness,
                 child_end):
        try:
            gp.fit(scaled_training_set, adjusted_training_fitness)
        except:
            gp = None
        child_end.send(gp)    
        
    def get_state_dictionary(self):
        try:
            theta_ = self.gp.theta_
        except:
            theta_ = None
            
        dict = {'training_set' : self.training_set,
                'training_fitness': self.training_fitness,
                'gp_theta': theta_}
        return dict
        
    def set_state_dictionary(self, dict):
        self.training_set = dict['training_set']
        self.training_fitness = dict['training_fitness']
        self.gp = self.regressor_countructor()
        try:
            self.gp.theta_ = dict['gp_theta']
        except:
            pass
        
## Different implementation of GPR regression, based on pyXGPR
class GaussianProcessRegressor2(Regressor):

    def __init__(self, controller, conf):
        super(GaussianProcessRegressor2, self).__init__(controller, conf)
        self.input_scaler = None
        self.output_scaler = None
        self.scaled_training_set = None
        self.adjusted_training_fitness = None
        self.covfunc = None
        self.gp=None
        self.conf = conf
            
    def train(self):
        try:
            MU_best=None
            gp_best=None
            S2_best=None
            nml_best=None
            self.gp=None
            conf = self.conf
            dimensions = len(self.training_set[0])
             # Scale inputs and particles?
            self.input_scaler = preprocessing.StandardScaler().fit(self.training_set)
            self.scaled_training_set = self.input_scaler.transform(
                self.training_set)

            # Scale training data
            self.output_scaler = preprocessing.StandardScaler(with_std=False).fit(
                self.training_fitness)
            self.adjusted_training_fitness = self.output_scaler.transform(
                self.training_fitness)
            ## retrain a number of times and pick best likelihood
            for i in xrange(conf.random_start):
                if conf.corr == "isotropic":
                    self.covfunc = ['kernels.covSum', ['kernels.covSEiso','kernels.covNoise']]
                    gp = [log(uniform(low=conf.thetaL, high=conf.thetaU)) for d in xrange(dimensions)]
                elif conf.corr == "anisotropic":
                    self.covfunc = ['kernels.covSum', ['kernels.covSEard','kernels.covNoise']]
                    gp = [log(uniform(low=conf.thetaL, high=conf.thetaU)) for d in xrange(dimensions+1)]
                else:
                    logging.error("The specified kernel function is not supported for GPR")
                    return False
                    
                gp.append(log(uniform(low=0.001, high=0.1)))
                try:
                    gp = array(gp)
                    gp, nml = gpr.gp_train(gp, self.covfunc, self.scaled_training_set, self.adjusted_training_fitness)
                    if gp[-1] < -2.0 :
                        if (((not nml_best) or (nml < nml_best))):
                            gp_best = gp
                            nml_best = nml
                except Exception,e:
                    logging.debug("Exception in gp_tren " + str(e))
                    pass
            ## the gp with highest likelihood becomes the new hyperparameter set
            self.set_gp(gp_best)
            if gp_best is None:
                raise Exception("Didnt manage to optimie hyperparameters")
            logging.info('Regressor training successful')
            return True
        except Exception, e:
            logging.info('Regressor training failed.. ' + str(e))
            return False
            
    def predict(self, z):
        if self.get_gp() is None:
            logging.error('Train GP before using it!!')
            return None, None, None
        try:
            # Scale inputs. it allows us to realod the regressor not retraining the model
            self.input_scaler = preprocessing.StandardScaler().fit(self.training_set)
            self.output_scaler = preprocessing.StandardScaler(with_std=False).fit(
                self.training_fitness) 
            self.adjusted_training_fitness = self.output_scaler.transform(
                self.training_fitness)
            self.scaled_training_set = self.input_scaler.transform(
                self.training_set)
                
            ## do predictions
            results = gpr.gp_pred(self.get_gp(), self.covfunc, self.scaled_training_set, self.adjusted_training_fitness, self.input_scaler.transform(array(z))) # get predictions for unlabeled data ONLY
            S2 = results[1]
            y_best = self.output_scaler.transform(self.get_y_best())
            EI = self.e_impr(S2, results[0], y_best)
            P = self.output_scaler.inverse_transform(self.sample(S2, results[0], y_best))
            
            MU = self.output_scaler.inverse_transform(results[0])

            ##get rid of negative variance... refer to some papers (there is a lot of it out there)
            for s,s2 in enumerate(S2):
                if s2 < 0.0:
                    S2[s] = 0.0
                    

            return MU, S2, EI, P
        except Exception, e:
            logging.error('Prediction failed... ' + str(e))
            return None, None, None
    
    def set_gp(self, gp):
        self.gp = gp
    
    def get_gp(self):
        return self.gp
        
    def get_state_dictionary(self):
        dict = {'training_set' : self.training_set,
                'training_fitness': self.training_fitness,
                'covfunc': self.covfunc,
                'gp': self.gp}
        return deepcopy(dict)
        
    def set_state_dictionary(self, dict):
        self.training_set = dict['training_set']
        self.training_fitness = dict['training_fitness']
        self.gp = dict['gp']
        self.covfunc = dict['covfunc']
        
## Different implementation of GPR regression, based on pyGPR
class GaussianProcessRegressor3(Regressor):
        
    def __init__(self, controller, conf):
        super(GaussianProcessRegressor3, self).__init__(controller, conf)
        self.input_scaler = None
        self.output_scaler = None
        self.scaled_training_set = None
        self.adjusted_training_fitness = None
        self.hyp=None
        self.conf = conf
        self.meanfunc = ['means.meanZero']#[ ['means.meanSum'], [ ['means.meanLinear'] , ['means.meanConst'] ] ]
        self.inffunc  = ['inf.infExact']
        self.likfunc  = ['lik.likGauss']
        self.covfunc = None
        self.nlml = 1000.0
        self.press = 99999999999.0
        self.transLog = False
    def train(self):
        return self.train_cross()
    
    def train_nlml(self):
        ## not sure how importatn this crap is..
        ## SET (hyper)parameters
        hyp = hyperParameters()   
        sn = 0.001; hyp.lik = array([log(sn)])        
        conf = self.conf
        dimensions = len(self.training_set[0])
        hyp.mean = [0.5 for d in xrange(dimensions)]
        hyp.mean.append(1.0)
        hyp.mean = array(hyp.mean)
        hyp.mean = array([])
         # Scale inputs and particles?
        self.input_scaler = preprocessing.StandardScaler().fit(self.training_set)
        self.scaled_training_set = self.input_scaler.transform(self.training_set)

        # Scale training data
        self.output_scaler = preprocessing.StandardScaler(with_std=False).fit(log(self.training_fitness - self.shift_by()))
        self.adjusted_training_fitness = self.output_scaler.transform(log(self.training_fitness - self.shift_by()))
        ## retrain a number of times and pick best likelihood
        nlml_best = None
        i = 0
        while i < conf.random_start:
            if conf.corr == "isotropic":
                self.covfunc = [['kernels.covSum'], [['kernels.covSEiso'],['kernels.covNoise']]]
                hyp.cov = [log(uniform(low=conf.thetaL, high=conf.thetaU)) for d in xrange(2)]
            elif conf.corr == "anisotropic":
                self.covfunc = [['kernels.covSum'], [['kernels.covSEard'],['kernels.covNoise']]]
                hyp.cov = [log(uniform(low=conf.thetaL, high=conf.thetaU)) for d in xrange(dimensions+1)]           
            elif conf.corr == "anirat": ## todo
                self.covfunc = [['kernels.covSum'], [['kernels.covRQard'],['kernels.covNoise']]]
                hyp.cov = [log(uniform(low=conf.thetaL, high=conf.thetaU)) for d in xrange(dimensions)]
                hyp.cov.append(log(uniform(low=conf.thetaL, high=conf.thetaU)))
                hyp.cov.append(log(uniform(low=conf.thetaL, high=conf.thetaU)))
            elif conf.corr == "matern3":
                self.covfunc = [['kernels.covSum'], [['kernels.covMatern'],['kernels.covNoise']]]
                hyp.cov = [log(uniform(low=conf.thetaL, high=conf.thetaU)) for d in xrange(2)]
                hyp.cov.append(log(3))                        
            elif conf.corr == "matern5":
                self.covfunc = [['kernels.covSum'], [['kernels.covMatern'],['kernels.covNoise']]]
                hyp.cov = [log(uniform(low=conf.thetaL, high=conf.thetaU)) for d in xrange(2)]
                hyp.cov.append(log(5))                
            elif conf.corr == "rqard":
                self.covfunc = [['kernels.covSum'], [['kernels.covRQard'],['kernels.covNoise']]]
                hyp.cov = [log(uniform(low=conf.thetaL, high=conf.thetaU)) for d in xrange(dimensions+2)]
            elif conf.corr == "special":
                self.covfunc = [['kernels.covSum'], [['kernels.covSEiso'],['kernels.covMatern'],['kernels.covNoise']]]
                hyp.cov = [log(uniform(low=conf.thetaL, high=conf.thetaU)) for d in xrange(2)]
                hyp.cov = hyp.cov + [log(uniform(low=conf.thetaL, high=conf.thetaU)) for d in xrange(2)]
                hyp.cov.append(log(3))
            else:
                logging.error("The specified kernel function is not supported")
                return False
                
            hyp.cov.append(log(uniform(low=0.1, high=1.0)))
            try:
                vargout = min_wrapper(hyp,gp,'BFGS',self.inffunc,self.meanfunc,self.covfunc,self.likfunc,self.scaled_training_set ,self.adjusted_training_fitness,None,None,True)
                hyp = vargout[0]
                vargout = gp(hyp, self.inffunc,self.meanfunc,self.covfunc,self.likfunc,self.scaled_training_set ,self.adjusted_training_fitness, None,None,False)      
                nlml = vargout[0]
                
                ### we add some sensible checking..
                ## matern we dont want to overfit
                ## we know that the function is not just noise hence < -1
                ## we know for anisotorpic that at least one parameter has to have some meaning
                ## 
                if (hyp.cov[-1] < -1.) and not ((conf.corr == "matern3") and hyp.cov[0] < 0.0) and not ((conf.corr == "anisotropic") and all(hyp.cov[:-2] < 0.0)):
                    logging.info(str(nlml) + " " + str(hyp.cov))
                    if (((not nlml_best) or (nlml < nlml_best))):
                        self.hyp = hyp
                        nlml_best = nlml
                else:
                    logging.info("hyper parameter out of spec: " + str(nlml) + " " + str(hyp.cov) + " " + str(hyp.cov[-1]))
                    i = i - 1 
            except Exception, e:
                logging.debug("Regressor training Failed: " + str(e))
                i = i - 1 
            i = i + 1            
        if (not nlml_best):        
            logging.debug("Regressor training Failed")
            return False
        ## the gp with highest likelihood becomes the new hyperparameter set
        self.nlml = nlml_best
        logging.info('Regressor training successful ' + str(self.hyp.cov) + " " + str(nlml_best))
        return True
            
    def train_cross(self):
        ## not sure how importatn this crap is..
        ## SET (hyper)parameters
        n_iters = len(self.training_set) * 5
        hyp = hyperParameters()   
        sn = 0.001; hyp.lik = array([log(sn)])        
        conf = self.conf
        dimensions = len(self.training_set[0])
        hyp.mean = [0.5 for d in xrange(dimensions)]
        hyp.mean.append(1.0)
        hyp.mean = array(hyp.mean)
        hyp.mean = array([])
         # Scale inputs and particles?
        self.input_scaler = preprocessing.StandardScaler().fit(self.training_set)
        self.scaled_training_set = self.input_scaler.transform(self.training_set)

        # Scale training data
        if self.transLog:
            self.output_scaler = preprocessing.StandardScaler(with_std=False).fit(log(self.training_fitness - self.shift_by()))
            self.adjusted_training_fitness = self.output_scaler.transform(log(self.training_fitness - self.shift_by()))
        else:
            self.output_scaler = preprocessing.StandardScaler(with_std=False).fit(self.training_fitness)
            self.adjusted_training_fitness = self.output_scaler.transform(self.training_fitness)
        ## retrain a number of times and pick best likelihood
        press_best = None
        best_hyp = None
        i = 0
        index_array = ShuffleSplit(len(self.scaled_training_set), n_iter=n_iters, train_size=0.8, test_size=0.2) ##we use 10% of example to evaluate our 
        while i < conf.random_start:
            if conf.corr == "isotropic":
                self.covfunc = [['kernels.covSum'], [['kernels.covSEiso'],['kernels.covNoise']]]
                hyp.cov = [log(uniform(low=conf.thetaL, high=conf.thetaU)) for d in xrange(2)]
            elif conf.corr == "anisotropic":
                self.covfunc = [['kernels.covSum'], [['kernels.covSEard'],['kernels.covNoise']]]
                hyp.cov = [log(uniform(low=conf.thetaL, high=conf.thetaU)) for d in xrange(dimensions+1)]           
            elif conf.corr == "anirat": ## todo
                self.covfunc = [['kernels.covSum'], [['kernels.covRQard'],['kernels.covNoise']]]
                hyp.cov = [log(uniform(low=conf.thetaL, high=conf.thetaU)) for d in xrange(dimensions)]
                hyp.cov.append(log(uniform(low=conf.thetaL, high=conf.thetaU)))
                hyp.cov.append(log(uniform(low=conf.thetaL, high=conf.thetaU)))
            elif conf.corr == "matern3":
                self.covfunc = [['kernels.covSum'], [['kernels.covMatern'],['kernels.covNoise']]]
                hyp.cov = [log(uniform(low=conf.thetaL, high=conf.thetaU)) for d in xrange(2)]
                hyp.cov.append(log(3))                        
            elif conf.corr == "matern5":
                self.covfunc = [['kernels.covSum'], [['kernels.covMatern'],['kernels.covNoise']]]
                hyp.cov = [log(uniform(low=conf.thetaL, high=conf.thetaU)) for d in xrange(2)]
                hyp.cov.append(log(5))                
            elif conf.corr == "rqard":
                self.covfunc = [['kernels.covSum'], [['kernels.covRQard'],['kernels.covNoise']]]
                hyp.cov = [log(uniform(low=conf.thetaL, high=conf.thetaU)) for d in xrange(dimensions+2)]
            elif conf.corr == "special":
                self.covfunc = [['kernels.covSum'], [['kernels.covSEiso'],['kernels.covMatern'],['kernels.covNoise']]]
                hyp.cov = [log(uniform(low=conf.thetaL, high=conf.thetaU)) for d in xrange(2)]
                hyp.cov = hyp.cov + [log(uniform(low=conf.thetaL, high=conf.thetaU)) for d in xrange(2)]
                hyp.cov.append(log(3))
            else:
                logging.error("The specified kernel function is not supported")
                return False
            
            hyp.cov.append(log(uniform(low=0.01, high=0.5)))
            ## 50% propability to usen the previous best hyper-parameters:
            if self.hyp:
                random_number = uniform(0.,1.)
                if random_number < 0.5:
                    hyp = self.hyp
                
            try:
                vargout = min_wrapper(hyp,gp,'BFGS',self.inffunc,self.meanfunc,self.covfunc,self.likfunc,self.scaled_training_set ,self.adjusted_training_fitness,None,None,True)
                hyp = vargout[0]
                ### we add some sensible checking..
                ## matern we dont want to overfit
                ## we know that the function is not just noise hence < -1
                ## we know for anisotorpic that at least one parameter has to have some meaning
                ## 
                press = 0.0
                for train_indexes, test_indexes in index_array:
                    test_set = self.scaled_training_set[test_indexes]
                    training_set = self.scaled_training_set[train_indexes]
                    test_fitness = self.adjusted_training_fitness[test_indexes]
                    training_fitness = self.adjusted_training_fitness[train_indexes]
                    vargout = gp(hyp, self.inffunc,self.meanfunc,self.covfunc, self.likfunc, training_set, training_fitness, test_set)      
                    predicted_fitness = vargout[2]
                    press = press + self.calc_press(predicted_fitness, test_fitness)
                    
                #if (hyp.cov[-1] < -1.) and not ((conf.corr == "matern3") and hyp.cov[0] < 0.0) and not ((conf.corr == "anisotropic") and all(hyp.cov[:-2] < 0.0)):
                logging.info("Press " + str(press) + " " + str(hyp.cov))
                if (((not press_best) or (press < press_best))):
                    best_hyp = hyp
                    press_best = press
            except Exception, e:
                logging.debug("Regressor training Failed: " + str(e))
            i = i + 1            
        if (not press_best):        
            logging.debug("Regressor training Failed")
            return False
        else:
            if best_hyp:
                self.hyp = best_hyp
        ## the gp with highest likelihood becomes the new hyperparameter set
        self.press = press_best
        logging.info('Regressor training successful ' + str(self.hyp.cov) + " " + str(press_best))
        return True
            
    def calc_press(self, y, yhat):
        return sum((y - yhat)*(y - yhat))
            
    def calc__press(self, y, yhat):
        return sum((y - yhat)*(y - yhat))
            
    def get_parameter_string(self):
        try:
            return str(self.nlml) + "_".join([str(round(i,3)) for i in self.hyp.cov])
        except Exception, e:
            try:
                return str(self.press) + "_".join([str(round(i,3)) for i in self.hyp.cov])
            except Exception, e:
                logging.debug(str(e))
                return "Not Trained"
            
    def shift_by(self): ## we need to due this due to log transformation of the training data
        nugget = 0.000000001
        if min(self.training_fitness) <= 0.0:
            return min(self.training_fitness) - nugget
        else:
            return 0.0
        
    def predict(self, z):
        if self.hyp is None:
            logging.error('Train GP before using it!!')
            return None, None, None, None
    # try:
        z = array(z) ## to ensure same input format
        if z.shape[0] == 1: ## for some reason cand do proper predictions for one element...
            shape_was_one = True
            z = array([z[0],z[0]])
        else:
            shape_was_one = False
        # Scale inputs. it allows us to realod the regressor not retraining the model
        self.input_scaler = preprocessing.StandardScaler().fit(self.training_set)
        
        if self.transLog:
            self.output_scaler = preprocessing.StandardScaler(with_std=False).fit(log(self.training_fitness - self.shift_by())) 
            self.adjusted_training_fitness = self.output_scaler.transform(log(self.training_fitness - self.shift_by()))
        else:
            self.output_scaler = preprocessing.StandardScaler(with_std=False).fit(self.training_fitness)
            self.adjusted_training_fitness = self.output_scaler.transform(self.training_fitness)

        self.scaled_training_set = self.input_scaler.transform(self.training_set)
        ## do predictions
        try:
            vargout = gp(self.hyp, self.inffunc,self.meanfunc,self.covfunc,self.likfunc,self.scaled_training_set ,self.adjusted_training_fitness, self.input_scaler.transform(z))      
        except Exception,e:
            logging.error(str(e))
            return None, None, None, None
        if self.transLog:
            MU = exp(self.output_scaler.inverse_transform(vargout[2]) + self.shift_by()) 
        else:
            MU = self.output_scaler.inverse_transform(vargout[2])
        S2 = vargout[3]
        ##get rid of negative variance... refer to some papers (there is a lot of it out there)
        for s,s2 in enumerate(S2):
            if s2 < 0.0:
                S2[s] = 0.0
        
        ## we are using unadjusted
        if self.conf.goal == "min":
            y_best = min(self.adjusted_training_fitness)       
        if self.conf.goal == "max":
            y_best = max(self.adjusted_training_fitness)
        EI = self.e_impr(S2, vargout[2], y_best)
        P = self.output_scaler.inverse_transform(self.sample(S2, vargout[2], y_best))
    
        for i,zz in enumerate(z):
            if self.contains_training_instance(zz):
                #logging.info(str(zz))
                S2[i]=0.0
                MU[i]=self.get_training_instance(zz)
                EI[i]=0.0
                P[i]=1.0
        
        
        if shape_was_one: ## for some reason cand do proper predictions for one element...
            MU = array([MU[0]])
            S2 = array([S2[0]])
            EI = array([EI[0]])
            P = array([P[0]])
            
        return MU, S2, EI, P
        #except Exception, e:
        #    logging.error('Prediction failed... ' + str(e))
        #    return None, None
        
    def get_state_dictionary(self):
        dict = {'training_set' : self.training_set,
                'training_fitness': self.training_fitness,
                'covfunc': self.covfunc,
                'nlml': self.nlml,
                'press': self.press,
                'hyp': self.hyp}
        return deepcopy(dict)
    
    def set_state_dictionary(self, dict):
        self.training_set = dict['training_set']
        self.training_fitness = dict['training_fitness']
        self.hyp = deepcopy(dict['hyp'])
        self.covfunc = deepcopy(dict['covfunc'])
        self.nlml = deepcopy(dict['nlml'])
        self.press = deepcopy(dict['press'])
        
        
#=================Below should not be used... unstable and poorly tested
## RPy breaks quite a bit... although potentially could deal with non-stationary problems.
        
## Different implementation of GPR regression, based on rpy
## working with r is hard... lack of concurrent access
## have to copy wiorking trees as tgp is saving trace in the working directory NIGHTMARE
## 
class GaussianProcessRegressorRpy(Regressor):
        
    def __init__(self, controller, conf):
        super(GaussianProcessRegressorRpy, self).__init__(controller, conf)
        self.input_scaler = None
        self.output_scaler = None
        self.scaled_training_set = None
        self.adjusted_training_fitness = None
        self.gp = None
        #try:
        self.folder_counter = 0
        self.my_dir = "/tmp/dummy/"
        
    def train(self):#
        time.sleep(1)
        try:
            self.my_dir = "/tmp/r_sess_" + str(os.getpid()) + "_" + str(self.folder_counter)
            os.mkdir(self.my_dir)
        except:
            self.folder_counter = self.folder_counter + 1
            self.my_dir = "/tmp/r_sess_" + str(os.getpid()) + "_" + str(self.folder_counter)
            os.mkdir(self.my_dir)
            
        #gc.collect()
        logging.info('Regressor training started..')
        
        ## not sure how importatn this crap is..
        ## SET (hyper)parameters
         # Scale inputs and particles?
        self.input_scaler = preprocessing.StandardScaler().fit(self.training_set)
        #self.scaled_training_set = r.matrix(FloatVector(self.input_scaler.transform(self.training_set).T.ravel().tolist()),nrow=self.training_fitness.shape[0])
        self.scaled_training_set = r.matrix(FloatVector(self.training_set.T.ravel().tolist()),nrow=self.training_fitness.shape[0])
        #logging.info())
        #logging.info(str( self.scaled_training_set))
        # Scale training data
        self.output_scaler = preprocessing.StandardScaler(with_std=False).fit(self.training_fitness)
        #self.adjusted_training_fitness = FloatVector(self.output_scaler.transform(self.training_fitness).ravel().tolist())
        self.adjusted_training_fitness = FloatVector(self.training_fitness.ravel().tolist())
        #logging.info(str( self.output_scaler.transform(self.training_fitness).ravel().tolist()))
        #logging.info(str( self.adjusted_training_fitness))
        #logging.info(str( self.training_fitness.shape))
        #logging.info(str( self.output_scaler.transform(self.training_fitness).ravel().tolist()))
        old_dir = os.getcwd()
        os.chdir(self.my_dir)
        self.tgp = importr("tgp")
        self.gp = self.tgp.btgp(self.scaled_training_set, self.adjusted_training_fitness)
        os.chdir(old_dir)        
        #self.tgp.plot_tgp(self.gp)
        ## the gp with highest likelihood becomes the new hyperparameter set
        logging.info('Regressor training successful')
        return True
            
    def predict(self, z):
        #gc.collect()
        time.sleep(1)
        if self.gp is None:
            logging.error('Train GP before using it!!')
            return None, None
        # Scale inputs. it allows us to realod the regressor not retraining the model
        self.input_scaler = preprocessing.StandardScaler().fit(self.training_set)
        self.output_scaler = preprocessing.StandardScaler(with_std=False).fit(self.training_fitness)
        trans_z = r.matrix(FloatVector(self.input_scaler.transform(array(z)).T.ravel().tolist()),ncol=self.training_set.shape[1])
        old_dir = os.getcwd()
        os.chdir(self.my_dir)
        self.tgp = importr("tgp")
        output = self.tgp.predict_tgp(self.gp,trans_z)        
        os.chdir(old_dir) 
        MU =output[15]
        MU = self.output_scaler.inverse_transform([[mu] for mu in MU])
        #logging.info(str(MU))
        S2 = array([[s2] for s2 in  output[23]])
        #logging.info(str(S2))
        ## do predictions
        ##get rid of negative variance... refer to some papers (there is a lot of it out there)
        for s,s2 in enumerate(S2):
            if s2 < 0.0:
                S2[s] = [0.0]
        
        return MU, S2
        #except Exception, e:
        #    logging.error('Prediction failed... ' + str(e))
        #    return None, None
        
    def get_state_dictionary(self):
        dict = {'gp' : self.gp, 
                'training_set' : self.training_set,
                'training_fitness': self.training_fitness,
                'my_dir': self.my_dir,
                'folder_counter' : self.folder_counter
                }
            
        return deepcopy(dict)
    
    def set_state_dictionary(self, dict):
        self.gp = dict['gp']
        self.training_set = dict['training_set']
        self.training_fitness = dict['training_fitness']
        self.folder_counter = dict['folder_counter']
        self.my_dir = dict['my_dir']
        self.rand = randrange(0,10000000,1)
        try:
            new_my_dir = "/tmp/r_sess_" + str(os.getpid()) + "_" + str(self.folder_counter) + "_" + str(self.rand)
            shutil.copytree(self.my_dir, new_my_dir)
            self.my_dir = new_my_dir
        except Exception, e:
            logging.debug("Fuck..." + str(e))
        return deepcopy(dict)
        
## Different implementation of GPR regression
class KMeansGaussianProcessRegressor(Regressor):

    def __init__(self, controller, conf):
        super(KMeansGaussianProcessRegressor, self).__init__(controller, conf)
        self.controller = controller
        self.state_dict = {}
        self.no_of_clusters = 3
        self.increase_max_no_of_clusters_by = 4
        self.reuse = 0.6
        self.conf = conf
        for i in range(0, 20):  #TODO -should be changed... stupid hardcoding fix
            self.state_dict[i] = self.regressor_countructor() 
        self.kmeans = self.kmeans_constructor(3)
            
    def train(self):
        
        dtype = [('distance', 'float'), ('index', int)]
        for num_of_clusters in range(3,self.increase_max_no_of_clusters_by+3):
            self.no_of_clusters = num_of_clusters
            self.kmeans = self.kmeans_constructor(num_of_clusters)
            self.kmeans.fit(self.training_set)
            succesfull = True
            for i in range(0,num_of_clusters):  
                labels = self.kmeans.predict(self.training_set)
                distances = [(euclidean(self.kmeans.cluster_centers_[i],sample), j) for j, sample in enumerate(self.training_set) if labels[j] != i ]
                distance_array = array(distances, dtype=dtype) ## not most efficient but readable
                cropped_distance_array = distance_array[0:int(len(self.training_set)/float(num_of_clusters))] ##lets assume trainingset > 1
                #logging.debug("cropped_distance_array" + str(len(cropped_distance_array)))
                #logging.debug("distances " + str(len(distances)))
                include_array = [index for distance, index in cropped_distance_array]
                #labels = self.kmeans.predict(self.training_set )
                self.state_dict[i] = self.regressor_countructor() ### need to reset the regressor
                for j, sample in enumerate(self.training_set):
                    if labels[j] == i or j in include_array:
                        self.state_dict[i].add_training_instance(sample,self.training_fitness[j])
                logging.debug("i:" + str(i) + " " + str(len(self.state_dict[i].training_set)))
                succesfull = self.state_dict[i].train() and succesfull
                if not succesfull:
                    logging.info("breaking training...")
                    break
            if succesfull:
                return True
            else:
                logging.info("Trying increasing number of clusters... currently: " + str(num_of_clusters))
        return False
            
    def kmeans_constructor(self, num_of_clusters):
        return KMeans(init='k-means++', n_clusters=num_of_clusters, n_init=10)
            
    def regressor_countructor(self):
        return GaussianProcessRegressor3(self.controller, self.conf)
            
    def predict(self, z):
        try:
            labels = self.kmeans.predict(z)
            MU = [0]*len(z)
            S2 = [0]*len(z)
            for i in range(0,self.no_of_clusters):  
                for j, sample in enumerate(z):
                    if labels[j] == i:
                        mu, s2 = self.state_dict[i].predict([sample])
                        MU[j] = mu[0]
                        S2[j] = s2[0]
            return array(MU), array(S2)
        except Exception,e:
            logging.debug("Something went wrong with the model..:" + str(e))
            return None, None
        
    def get_state_dictionary(self):
        try:
            state_dict = {  "no_of_clusters": self.no_of_clusters,
                            'training_set' : self.training_set,
                            'kmeans_cluster_centers_' : self.kmeans.cluster_centers_,
                            'kmeans_labels_' : self.kmeans.labels_,
                            'kmeans_inertia_' : self.kmeans.inertia_,
                            'training_fitness': self.training_fitness}
         ##   logging.info(str(self.kmeans.cluster_centers_))
            for i in range(0,self.no_of_clusters):  
                state_dict[i] = self.state_dict[i].get_state_dictionary()
            return state_dict
        except:
            logging.info("Model sa not been initialized..")
            return {}
        
    def set_state_dictionary(self, dict):
        try:
            self.training_set = dict['training_set']
            self.training_fitness = dict['training_fitness']
            self.no_of_clusters = dict['no_of_clusters']
            self.kmeans = self.kmeans_constructor(self.no_of_clusters)
            self.kmeans.cluster_centers_ = dict['kmeans_cluster_centers_']
            self.kmeans.labels_ = dict['kmeans_labels_']
            self.kmeans.inertia_ = dict['kmeans_inertia_']
            for i in range(0,self.no_of_clusters): 
                self.state_dict[i] = self.regressor_countructor()
                self.state_dict[i].set_state_dictionary(dict[i])
        except:
            logging.info("Supplied Empty dictionary..")
            
## Different implementation of GPR regression
class DPGMMGaussianProcessRegressor(Regressor):

    def __init__(self, controller, conf):
        super(DPGMMGaussianProcessRegressor, self).__init__(controller, conf)
        self.controller = controller
        self.state_dict = {}
        self.no_of_clusters = 2
        self.max_no_of_clusters = 4
        self.radius = 0.1 ## TODO
        self.conf = conf
        for i in range(0, self.max_no_of_clusters): 
            self.state_dict[i] = self.regressor_countructor() 
        self.dpgmm = self.dpgmm_constructor()
            
    def train(self):
        self.no_of_clusters = min(self.max_no_of_clusters, len(self.training_set))
        self.dpgmm = self.dpgmm_constructor()
        self.dpgmm.fit(self.training_set)
        labels = self.dpgmm.predict(self.training_set)
        ##proba = self.dpgmm.predict_proba(self.training_set)
        for i in range(0,self.no_of_clusters):  
            self.state_dict[i] = self.regressor_countructor() ### need to reset the regressor
            for j, sample in enumerate(self.training_set):
                if labels[j] == i:
                    self.state_dict[i].add_training_instance(sample,self.training_fitness[j])
            if (not self.state_dict[i].training_set_empty()):
                self.state_dict[i].train()
            
    def dpgmm_constructor(self):
        return mixture.DPGMM(n_components=self.no_of_clusters, covariance_type='diag', alpha=100.,n_iter=100)
            
    def regressor_countructor(self):
        return GaussianProcessRegressor2(self.controller, self.conf)
            
    def predict(self, z):
        try:
            labels = self.dpgmm.predict(z)
            logging.info(str(labels))
            MU = [0]*len(z)
            S2 = [0]*len(z)
            for i in range(0,self.no_of_clusters):  
                for j, sample in enumerate(z):
                    if labels[j] == i:
                        mu, s2 = self.state_dict[i].predict([sample])
                        MU[j] = mu[0]
                        S2[j] = s2[0]
            return array(MU), array(S2)
        except: 
            traceback.print_tb
        
    def get_state_dictionary(self):
        try:
            state_dict = {  "no_of_clusters": self.no_of_clusters,
                            'training_set' : self.training_set,
                            'dpgmm' : deepcopy(self.dpgmm),
                            'training_fitness': self.training_fitness}
            for i in range(0,self.no_of_clusters):  
                state_dict[i] = self.state_dict[i].get_state_dictionary()
            return state_dict
        except Exception, e:
            logging.info("Supplied Empty dictionary.." + str(e))
            return {}
        
    def set_state_dictionary(self, dict):
        try:
            self.training_set = dict['training_set']
            self.training_fitness = dict['training_fitness']
            self.no_of_clusters = dict['no_of_clusters']
            self.dpgmm = dict['dpgmm']
            for i in range(0,self.no_of_clusters): 
                self.state_dict[i] = self.regressor_countructor()
                self.state_dict[i].set_state_dictionary(dict[i])
        except Exception, e:
            logging.info("Supplied Empty dictionary.." + str(e))
            
            
            

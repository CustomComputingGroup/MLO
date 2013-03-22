import logging
from multiprocessing import Process, Pipe
import traceback

from numpy import unique, asarray, bincount, array, append, sqrt, log, sort
from sklearn import preprocessing
from sklearn.gaussian_process import GaussianProcess
from sklearn.grid_search import GridSearchCV
from sklearn.cluster import KMeans
from sklearn import mixture
from scipy.spatial.distance import euclidean
from numpy.random import uniform, shuffle, permutation

from utils import numpy_array_index
from copy import deepcopy

from GPR import gpr

#TODO - abstract class
class Regressor(object):

    def __init__(self, controller, conf):
        self.training_set = None
        self.training_fitness = None
        self.regr = None
        self.controller = controller
        self.conf = conf

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
                try:
                    self.training_set = append(self.training_set, [part], axis=0)
                    self.training_fitness = append(self.training_fitness, [fitness],
                                               axis=0)
                except:
                    logging.info(str(self.training_fitness))
                    raise Exception("KURWO")

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
    # def __getstate__(self):
       # Don't pickle controller
        # d = dict(self.__dict__)
        # del d['controller']
        # return d
        
    def training_set_empty(self):
        return (self.training_set is None)
        
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
            MU, S2 = self.regr.predict(self.input_scaler.transform(array(z)),
                                       eval_MSE=True)
            #logging.debug(MU)
            MU = self.output_scaler.inverse_transform(MU)
            MU = MU.reshape(-1, 1)
            S2 = sqrt(S2.reshape(-1, 1))
            return MU, S2
        except Exception, e:
            logging.error('Prediction failed.... ' + str(e) + "KURWA")
            return None, None

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
        
## Different implementation of GPR regression
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
            MU_best = None
            gp_best = None
            S2_best = None
            nml_best = None
            self.gp = None
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
            try:
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
                        
                    gp.append(log(uniform(low=0.0001, high=0.01)))
                    try:
                        gp = array(gp)
                        gp, nml = gpr.gp_train(gp, self.covfunc, self.scaled_training_set, self.adjusted_training_fitness)
                        if gp[-1] > -3.0 :
                            raise Exception("Error to large",nml)
                        if (((not nml_best) or (nml < nml_best))):
                            gp_best = gp
                            nml_best = nml
                    except Exception,e:
                        pass
                ## the gp with highest likelihood becomes the new hyperparameter set
                self.set_gp(gp_best)
            except Exception,e:
                ### will try to retarin till succesful
                logging.info('Regressor training failed.. retraining.. ' + str(e))
                self.train()
            logging.info('Regressor training successful')
            return True
        except Exception, e:
            logging.info('Regressor training failed.. retraining.. ' + str(e))
            return False
            
    def predict(self, z):
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
            MU = self.output_scaler.inverse_transform(results[0])
            S2 = results[1]
            ##get rid of negative variance... refer to some papers (there is a lot of it out there)
            for s,s2 in enumerate(S2):
                if s2 < 0.0:
                    S2[s] = 0.0
            return MU, S2
        except Exception, e:
            logging.error('Prediction failed... ' + str(e))
            return None, None
    
    def set_gp(self, gp):
        self.gp = gp
    
    def get_gp(self):
        return self.gp
        
    def get_state_dictionary(self):
        dict = {'training_set' : self.training_set,
                'training_fitness': self.training_fitness,
                'covfunc': self.covfunc,
                'gp': self.gp}
        return dict
        
    def set_state_dictionary(self, dict):
        self.training_set = dict['training_set']
        self.training_fitness = dict['training_fitness']
        self.gp = dict['gp']
        self.covfunc = dict['covfunc']
        
## Different implementation of GPR regression
class KMeansGaussianProcessRegressor(Regressor):

    def __init__(self, controller, conf):
        super(KMeansGaussianProcessRegressor, self).__init__(controller, conf)
        self.controller = controller
        self.state_dict = {}
        self.no_of_clusters = 0
        self.max_no_of_clusters = 3
        self.reuse = 0.6
        self.conf = conf
        for i in range(0, self.max_no_of_clusters): 
            self.state_dict[i] = self.regressor_countructor() 
        self.kmeans = self.kmeans_constructor()
            
    def train(self):
        self.no_of_clusters = min(self.max_no_of_clusters, len(self.training_set)) 
        self.kmeans = self.kmeans_constructor()
        self.kmeans.fit(self.training_set)
        dtype = [('distance', 'float'), ('index', int)]
        succesfull = True
        for i in range(0,self.no_of_clusters):  
            
            labels = self.kmeans.predict(self.training_set)
            distances = [(euclidean(self.kmeans.cluster_centers_[i],sample), j) for j, sample in enumerate(self.training_set) if labels[j] != i ]
            distance_array = array(distances, dtype=dtype) ## not most efficient but readable
            cropped_distance_array = distance_array[0:int(len(self.training_set)/float(self.no_of_clusters))] ##lets assume trainingset > 1
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
        return succesfull
        
    def kmeans_constructor(self):
        return KMeans(init='k-means++', n_clusters=self.no_of_clusters, n_init=10)
            
    def regressor_countructor(self):
        return GaussianProcessRegressor2(self.controller, self.conf)
            
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
            self.kmeans = self.kmeans_constructor()
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
        self.no_of_clusters = 0
        self.max_no_of_clusters = 2
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
            
            
            
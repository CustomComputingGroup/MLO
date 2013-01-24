import logging
from multiprocessing import Process, Pipe
import traceback

from particles import *

from numpy import unique, asarray, bincount, array, append, sqrt, log
from sklearn import preprocessing
from sklearn.gaussian_process import GaussianProcess
from sklearn.grid_search import GridSearchCV

from utils import numpy_array_index

from GPR import gpr


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
                                     random_start=conf.random_start)
            else:
                gp = GaussianProcess(regr=conf.regr, corr=conf.corr,
                                     theta0=array([conf.theta0] * dimensions),
                                     thetaL=array([conf.thetaL] * dimensions),
                                     thetaU=array([conf.thetaU] * dimensions),
                                     random_start=conf.random_start, nugget=conf.nugget)

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

        
## Different implementation of GPR regression
class GaussianProcessRegressor2(Regressor):

    def _init_(self, controller):
        super(GaussianProcessRegressor2, self).__init__(controller)
        self.input_scaler = None
        self.output_scaler = None
        self.scaled_training_set = None
        self.adjusted_training_fitness = None
        self.covfunc = None
            
    def train(self, pop, conf, dimensions):
        try:
            MU_best=None
            gp_best=None
            S2_best=None
            nml_best=None
            self.gp=None
             # Scale inputs and particles?
            self.input_scaler = preprocessing.Scaler().fit(self.training_set)
            self.scaled_training_set = self.input_scaler.transform(
                self.training_set)

            # Scale training data
            self.output_scaler = preprocessing.Scaler(with_std=False).fit(
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
                        gp,nml = gpr.gp_train(gp, self.covfunc, self.scaled_training_set, self.adjusted_training_fitness)
                        if gp[-1] > -3.0 :
                            raise Exception("Error to large",nml)
                        if (((not nml_best) or (nml < nml_best))):
                            gp_best = gp
                            nml_best = nml
                    except Exception,e:
                        pass
                ## the gp with highest likelihood becomes the new hyperparameter set
                self.gp = gp_best
            except Exception,e:
                ### will try to retarin till succesful
                logging.info('Regressor training failed.. retraining.. {}'.format(e))
                self.train(pop, conf, dimensions)
            #logging.info('Regressor training successful')
            return True
        except Exception, e:
            logging.error('Regressor training failed.. {}'.format(e))
            return False
            
    def predict(self, z):
        try:
            results = gpr.gp_pred(self.gp, self.covfunc, self.scaled_training_set, self.adjusted_training_fitness, self.input_scaler.transform(array(z))) # get predictions for unlabeled data ONLY
            MU = self.output_scaler.inverse_transform(results[0])
            S2 = results[1]
            ##get rid of negative variance... refer to some papers (there is a lot of it out there)
            for s,s2 in enumerate(S2):
                if s2 < 0.0:
                    S2[s] = 0.0
            return MU, S2
        except Exception, e:
            logging.error('Prediction failed.. {}'.format(e))
            return None
    
    
    # def returnMaxS2(pop,resultsFolder,iterations,toolbox,npts=200,d1=0,d2=1,fitness=None,gp=None,hypercube=None):
    # global gpTrainingSet,gpTrainingFitness,clf
    
    # designSpace = fitness.designSpace
    # if len(designSpace)==2:
        #make up data.
        # if hypercube:
            # print "[returnMaxS2]: using hypercube ",hypercube
            # x = linspace(hypercube[1][0],hypercube[0][0],npts)
            # y = linspace(hypercube[1][1],hypercube[0][1],npts) 
        # else:
            # x = linspace(designSpace[0]["min"],designSpace[0]["max"],npts)
            # y = linspace(designSpace[1]["min"],designSpace[1]["max"],npts)
        # x,y = meshgrid(x,y)
        # x=reshape(x,-1)
        # y=reshape(y,-1)
        # z = array([[a,b] for (a,b) in zip(x,y)])
    # else:
        # x,y,v = mgrid[designSpace[0]["min"]:designSpace[0]["max"]:(int(designSpace[0]["max"]-designSpace[0]["min"])+1)*1.0j,designSpace[1]["min"]:designSpace[1]["max"]:(int(designSpace[1]["max"]-designSpace[1]["min"])+1)*1.0j , designSpace[2]["min"]:designSpace[2]["max"]:(int(designSpace[2]["max"]-designSpace[2]["min"])+1)*1.0j]
        # x=reshape(x,-1)
        # y=reshape(y,-1)
        # v=reshape(v,-1)
        # z = array([[a,b,c] for (a,b,c) in zip(x,y,v)])

    # try:
        # zClass = classifyUsingSvm(z)
    # except Exception,e:
        # print "[returnMaxS2]: classifying failed.. ",e
        # pass
        
    # try:     
        # zReal = array([fitness.fitnessFunc(a)[0][0] for a in z])
        # zRealClass = array([fitness.fitnessFunc(a)[1][0] for a in z])
        # minn = argmin(zReal)
        # filteredminn = []
        # filteredZ=[]
        # for i,zreal in enumerate(zReal):
            # if zRealClass[i]==0.0:
                # filteredminn.append(zreal)
                # filteredZ.append(z[i])
    
        # (MU,S2,modelFailed,gp) = toolbox.trainModel(z,gpTrainingSet=gpTrainingSet,gpTrainingFitness=gpTrainingFitness,gp=gp)
        # filteredS2=[]
        # filteredZ=[]
        # for i,s2 in enumerate(S2):
            # if zClass[i]==0.0:
                # filteredS2.append(s2)
                # filteredZ.append(z[i])
        # S2 = array(filteredS2) 
        # return filteredZ[argmax(S2)]
    # except Exception,e:
        # if enableTracebacks:
            # print traceback.print_exc()
        # print "[returnMaxS2]: finding maximum S2 failed... ommiting",e
        # return toolbox.particle()
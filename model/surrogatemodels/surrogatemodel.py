import logging

from classifiers import Classifier, SupportVectorMachineClassifier
from regressors import Regressor, GaussianProcessRegressor, GaussianProcessRegressor2, GaussianProcessRegressor3, KMeansGaussianProcessRegressor, DPGMMGaussianProcessRegressor, GaussianProcessRegressorRpy

from utils import numpy_array_index
from scipy.interpolate import griddata
from scipy.stats import norm
from numpy import linspace, meshgrid, reshape, array, argmax, mgrid, ones, arange
import itertools 
import pdb

class SurrogateModel(object):

    def __init__(self, configuration, controller, fitness):
        self.configuration = configuration
        self.fitness = fitness
        self.controller = controller
        self.was_trained = False
        
    def train(self, hypercube):
        raise NotImplementedError('SurrogateModel is an abstract class, this '
                                  'should not be called.')
    def trained(self):
        return self.was_trained
                                  
    def predict(self, particles):
        raise NotImplementedError('SurrogateModel is an abstract class, this '
                                  'should not be called.')

    def add_training_instance(self, part, code, fitness, addReturn):
        pass
        
    def contains_training_instance(self, part):
        pass    
        
    def get_training_instance(self, part):
        pass

    # def __getstate__(self):
        # Don't pickle fitness and configuration
        # d = dict(self.__dict__)
        # del d['configuration']
        # del d['fitness']
        # return d

    def contains_particle(self, part):
        pass
        
    def particle_value(self, part):
        pass
    
    def model_failed(self, part):
        pass
        
    def max_uncertainty(self, designSpace, hypercube=None, npts=200):
        pass

    def get_state_dictionary(self):
        raise NotImplementedError('Trial is an abstract class, '
                                  'this should not be called.')
        
    def set_state_dictionary(self, dict):
        raise NotImplementedError('Trial is an abstract class, '
                                  'this should not be called.')   

    def get_copy(self):
        raise NotImplementedError('Trial is an abstract class, '
                                  'this should not be called.')
                                  
    def get_regressor(self):
        raise NotImplementedError('Trial is an abstract class, '
                                  'this should not be called.')
                                  
    def get_classifier(self):
        raise NotImplementedError('Trial is an abstract class, '
                                  'this should not be called.')
        
class DummySurrogateModel(SurrogateModel):

    ## TODO - add dummy regressor/classifier
    def __init__(self, configuration, controller, fitness):
        super(DummySurrogateModel, self).__init__(configuration,
                                                   controller,
                                                   fitness)
        self.regressor = Regressor(controller, configuration)
        self.classifier = Classifier()

    def get_regressor(self):
        return self.regressor
                                  
    def get_classifier(self):
        return self.classifier
        
    def predict(self, particles):
        MU, S2 = self.regressor.predict(particles)
        return self.classifier.predict(particles), MU, S2

    def train(self, hypercube):
        self.was_trained = True
        return True

    def model_particle(self, particle):
        return 0, 0, 0
        
    def contains_training_instance(self, part):
        return False

    def model_failed(self, part):
        return False
        
    def get_state_dictionary(self):
        return {}
        
    def set_state_dictionary(self, dict):
        pass
        
    def get_copy(self):
        model_copy = DummySurrogateModel(self.configuration, self.controller)
        return model_copy
        
class ProperSurrogateModel(SurrogateModel):

    def __init__(self, configuration, controller, fitness):
        super(ProperSurrogateModel, self).__init__(configuration,
                                                   controller,
                                                   fitness)
                                                   
        if configuration.classifier == 'SupportVectorMachine':
            self.classifier = SupportVectorMachineClassifier()
        else:
            logging.error('Classifier type ' + str(configuration.classifier) + '  not found')
        self.regressor = self.regressor_constructor()
        
        try:
            if self.configuration.sample_on == "ei":
                self.max_uncertainty = self.max_ei
            elif self.configuration.sample_on == "s":
                self.max_uncertainty = self.max_s2
        except:
            if self.max_uncertainty:
                pass
            else:
                logging.debug("Sampling scheme wasnt specified, using Expected Improvment")
                self.max_uncertainty = self.max_ei
        
    def get_regressor(self):
        return self.regressor
                                  
    def get_classifier(self):
        return self.classifier
        
    def get_copy(self):
        model_copy = ProperSurrogateModel(self.configuration, self.controller)
        model_copy.set_state_dictionary(self.get_state_dictionary())
        return model_copy
            
    def predict(self, particles):
        try:
            #logging.debug("Using tranformation function for the regressor")
            trans_particles = particles
        except:
            trans_particles = [self.fitness.transformation_function(part) for part in particles]
        MU, S2, EI, P = self.regressor.predict(trans_particles)
        return self.classifier.predict(particles), MU, S2, EI, P

    def train(self, hypercube=None):
        self.was_trained = True
        if self.classifier.train() and self.regressor.train():
            logging.info("Trained Surrogate Model")
        else:
            logging.info("Couldnt Train Surrogate Model")
            return False
            
    def regressor_constructor(self):
        controller = self.controller
        configuration = self.configuration
        if self.configuration.regressor == 'GaussianProcess':
            return GaussianProcessRegressor(controller, configuration)
        elif self.configuration.regressor == 'GaussianProcess2':
            return GaussianProcessRegressor2(controller, configuration)          
        elif self.configuration.regressor == 'GaussianProcess3':
            return GaussianProcessRegressor3(controller, configuration)        
        elif self.configuration.regressor == 'KMeansGaussianProcessRegressor':
            return KMeansGaussianProcessRegressor(controller, configuration)        
        elif self.configuration.regressor == 'DPGMMGaussianProcessRegressor':
            return DPGMMGaussianProcessRegressor(controller, configuration)
        elif self.configuration.regressor == 'R':
            return GaussianProcessRegressorRpy(controller, configuration)
        else:
            raise Exception('Regressor type ' + str(configuration.regressor) + '  not found')
        
    def add_training_instance(self, part, code, fitness, addReturn):
        self.classifier.add_training_instance(part, code)
        if addReturn[0] == 0: ## only update regressor if the fitness function produced a result
            try:
                trans_part = self.fitness.transformation_function(part)
                #logging.debug("Using tranformation function for the regressor")
            except:
                trans_part = part
            self.regressor.add_training_instance(trans_part, fitness)
        
    def contains_training_instance(self, part):
        try:
            trans_part = self.fitness.transformation_function(part)
            #logging.debug("Using tranformation function for the regressor")
        except:
            trans_part = part
        return self.regressor.contains_training_instance(trans_part) or self.classifier.contains_training_instance(part)  

    def get_training_instance(self, part):
        code = self.classifier.get_training_instance(part) 
        fitness = None
        if self.regressor.contains_training_instance(part):
            fitness = self.regressor.get_training_instance(part)            
        return code, fitness
        
    def model_failed(self, part):
        return False
        
    def max_ei(self, designSpace, hypercube=None, npts=10):
        D = len(designSpace)
        n_bins = npts*ones(D)
        
        grid = False
        if grid:
            if hypercube:
                result = mgrid[[slice(h_min, h_max, npts*1.0j) for h_max, h_min , n in zip(hypercube[0],hypercube[1], n_bins)]]
                z = result.reshape(D,-1).T
            else:
                bounds = [(d["min"],d["max"]) for d in designSpace]
                result = mgrid[[slice(row[0], row[1], npts*1.0j) for row, n in zip(bounds, n_bins)]]
                z = result.reshape(D,-1).T
                '''
                x,y,v = mgrid[designSpace[0]["min"]:designSpace[0]["max"]:(int(designSpace[0]["max"]-designSpace[0]["min"])+1)*1.0j,designSpace[1]["min"]:designSpace[1]["max"]:(int(designSpace[1]["max"]-designSpace[1]["min"])+1)*1.0j , designSpace[2]["min"]:designSpace[2]["max"]:(int(designSpace[2]["max"]-designSpace[2]["min"])+1)*1.0j]
                x=reshape(x,-1)
                y=reshape(y,-1)
                v=reshape(v,-1)
                z = array([[a,b,c] for (a,b,c) in zip(x,y,v)])
                '''
            try:             
                zClass, MU, S2, EI, P = self.predict(z)
                filteredEI=[]
                filteredZ=[]
                for i,ei in enumerate(EI):
                    if zClass[i]==0:
                        filteredEI.append(ei)
                        filteredZ.append(z[i])
                EI = array(filteredEI) 
                return filteredZ[argmax(EI)]
            except Exception,e:
                logging.error("Finding max S2 failed: " + str(e))
                return None
        else: ## more memory efficient yet slower
            maxEI = None
            maxEIcord = None
            maxEI2 = None
            maxEIcord2 = None
            space_def = []
            if hypercube:
                for counter, d in enumerate(designSpace):
                    if d["type"] == "discrete":
                        space_def.append(arange(hypercube[1][counter],hypercube[0][counter]+d["step"],d["step"])) ## arrange works up to...
                    else:
                        space_def.append(arange(hypercube[1][counter],hypercube[0][counter],((hypercube[0][counter]-hypercube[1][counter])/100.0)))
            else:
                for d in designSpace:
                    if d["type"] == "discrete":
                        space_def.append(arange(d["min"],d["max"]+d["step"],d["step"])) ## arrange works up to...
                    else:
                        space_def.append(arange(d["min"],d["max"],((d["max"]-d["min"])/100.0)))
            for z in itertools.product(*space_def):
                if not self.contains_training_instance(array(z)):
                    
                    #pdb.set_trace()
                    zClass, MU, S2, EI, P = self.predict(array([z]))
                    #logging.info(str(z) + " " + str(zClass[0]) + " " + str(EI[0]))
                    if maxEI < EI[0] and zClass[0]==0: ## no need for None checking
                        maxEI = EI[0]
                        maxEIcord = z
                    if maxEI2 < EI[0]: ## no need for None checking
                        maxEI2 = EI[0]
                        maxEIcord2 = z
            logging.info("Maximum Expected Improvment is at:" + str(maxEIcord))
            logging.info("Maximum Expected Improvment without classifier is at:" + str(maxEIcord2))
            return maxEIcord2
            
    def max_ei_cost(self, designSpace, hypercube=None, npts=10, cost_func = None):
        D = len(designSpace)
        n_bins = npts*ones(D)
        
        grid = False
        if grid:
            if hypercube:
                result = mgrid[[slice(h_min, h_max, npts*1.0j) for h_max, h_min , n in zip(hypercube[0],hypercube[1], n_bins)]]
                z = result.reshape(D,-1).T
            else:
                bounds = [(d["min"],d["max"]) for d in designSpace]
                result = mgrid[[slice(row[0], row[1], npts*1.0j) for row, n in zip(bounds, n_bins)]]
                z = result.reshape(D,-1).T
                '''
                x,y,v = mgrid[designSpace[0]["min"]:designSpace[0]["max"]:(int(designSpace[0]["max"]-designSpace[0]["min"])+1)*1.0j,designSpace[1]["min"]:designSpace[1]["max"]:(int(designSpace[1]["max"]-designSpace[1]["min"])+1)*1.0j , designSpace[2]["min"]:designSpace[2]["max"]:(int(designSpace[2]["max"]-designSpace[2]["min"])+1)*1.0j]
                x=reshape(x,-1)
                y=reshape(y,-1)
                v=reshape(v,-1)
                z = array([[a,b,c] for (a,b,c) in zip(x,y,v)])
                '''
            try:             
                zClass, MU, S2, EI, P = self.predict(z)
                filteredEI=[]
                filteredZ=[]
                for i,ei in enumerate(EI):
                    if zClass[i]==0:
                        filteredEI.append(ei)
                        filteredZ.append(z[i])
                EI = array(filteredEI) 
                return filteredZ[argmax(EI)]
            except Exception,e:
                logging.error("Finding max S2 failed: " + str(e))
                return None
        else: ## more memory efficient yet slower
            maxEI = None
            maxEIcord = None
            space_def = []
            for d in designSpace:
                if d["type"] == "discrete":
                    space_def.append(arange(d["min"],d["max"],d["step"]))
                else:
                    space_def.append(arange(d["min"],d["max"],((d["max"]-d["min"])/100.0)))
            for z in itertools.product(*space_def):
                zClass, MU, S2, EI, P = self.predict([z])
                EI_over_cost = EI / cost_func(z)
                if maxEI < EI: ## no need for None checking
                    maxEI = EI
                    maxEIcord = z
            return z
            
    def max_s2(self, designSpace, hypercube=None, npts=10):
        if len(designSpace)==2:
            # make up data.
            if hypercube:
                x = linspace(hypercube[1][0],hypercube[0][0],npts)
                y = linspace(hypercube[1][1],hypercube[0][1],npts) 
            else:
                x = linspace(designSpace[0]["min"],designSpace[0]["max"],npts)
                y = linspace(designSpace[1]["min"],designSpace[1]["max"],npts)
            x,y = meshgrid(x,y)
            x=reshape(x,-1)
            y=reshape(y,-1)
            z = array([[a,b] for (a,b) in zip(x,y)])
        else:
            D = len(designSpace)
            n_bins =  npts*ones(D)
            if hypercube:
                result = mgrid[[slice(h_min, h_max, npts*1.0j) for h_max, h_min , n in zip(hypercube[0],hypercube[1], n_bins)]]
                z = result.reshape(D,-1).T
            else:
                bounds = [(d["min"],d["max"]) for d in designSpace]
                result = mgrid[[slice(row[0], row[1], npts*1.0j) for row, n in zip(bounds, n_bins)]]
                z = result.reshape(D,-1).T

            '''
            x,y,v = mgrid[designSpace[0]["min"]:designSpace[0]["max"]:(int(designSpace[0]["max"]-designSpace[0]["min"])+1)*1.0j,designSpace[1]["min"]:designSpace[1]["max"]:(int(designSpace[1]["max"]-designSpace[1]["min"])+1)*1.0j , designSpace[2]["min"]:designSpace[2]["max"]:(int(designSpace[2]["max"]-designSpace[2]["min"])+1)*1.0j]
            x=reshape(x,-1)
            y=reshape(y,-1)
            v=reshape(v,-1)
            z = array([[a,b,c] for (a,b,c) in zip(x,y,v)])
            '''
        try:             
            zClass, MU, S2 = self.predict(z)
            #logging.info(str(MU))
            #logging.info(str(zClass))
            #logging.info(str(S2))
            filteredS2=[]
            filteredZ=[]
            for i,s2 in enumerate(S2):
                if zClass[i]==0:
                    filteredS2.append(s2)
                    filteredZ.append(z[i])
            S2 = array(filteredS2) 
            return filteredZ[argmax(S2)]
        except Exception,e:
            logging.error("Finding max S2 failed: " + str(e))
            return None
            
    def get_state_dictionary(self):
        return {"regressor_state_dict" : self.regressor.get_state_dictionary(), "classifier_state_dicts" : self.classifier.get_state_dictionary()}
        
    def set_state_dictionary(self, dict):
        self.regressor.set_state_dictionary(dict["regressor_state_dict"])
        self.classifier.set_state_dictionary(dict["classifier_state_dicts"])

class LocalSurrogateModel(ProperSurrogateModel):

    def __init__(self, configuration, controller, fitness):
        super(LocalSurrogateModel, self).__init__(configuration,
                                                   controller,
                                                   fitness)
        D = len(fitness.designSpace)
        self.max_r = D*10
        self.regressor = self.regressor_constructor()
        self.local_regressor = self.regressor_constructor()
        
        try:
            if self.configuration.sample_on == "ei":
                self.max_uncertainty = self.max_ei
            elif self.configuration.sample_on == "s":
                self.max_uncertainty = self.max_s2
        except:
            if self.max_uncertainty:
                pass
            else:
                logging.debug("Sampling scheme wasnt specified, using Expected Improvment")
                self.max_uncertainty = self.max_ei
                
    def max_ei(self, designSpace, hypercube=None, npts=10):
        logging.info("IN")
        if hypercube: ## train the regressor globally and use it for sampling
            logging.info("Training global regressor for sampling")
            self.train_global()
            return super(LocalSurrogateModel, self).max_ei(designSpace=designSpace, hypercube=hypercube, npts=npts)
        else: ## use the local regressor, in order to keep the max_ei method intact we need to swap regressors for a moment
            temp_regressor = self.regressor
            self.regressor = self.local_regressor
            logging.info("Using local regressor for sampling")
            results = super(LocalSurrogateModel, self).max_ei(designSpace=designSpace, hypercube=hypercube, npts=npts)
            self.regressor = temp_regressor
            return results 
            
    def get_regressor(self):
        return self.local_regressor
                                  
    def get_classifier(self):
        return self.classifier
            
    def train_global(self): 
        super(LocalSurrogateModel, self).train() ## train global...
                
    def predict(self, particles):
        try:
            #logging.debug("Using tranformation function for the regressor")
            trans_particles = particles
        except:
            trans_particles = [self.fitness.transformation_function(part) for part in particles]
        MU, S2, EI, P = self.local_regressor.predict(trans_particles)
        return self.classifier.predict(particles), MU, S2, EI, P
                
    def train_local(self, hypercube):
        self.local_regressor = self.regressor_constructor()
        regressor_training_fitness = self.regressor.get_training_fitness()
        regressor_training_set = self.regressor.get_training_set()
        [maxDiag,minDiag] = hypercube
        logging.info(str(hypercube))
        for k, part in enumerate(regressor_training_set):
            if all(part <= maxDiag):
                if all(part >= minDiag):
                    self.local_regressor.add_training_instance(part, regressor_training_fitness[k])
        ## add most recent
        valid_examples = min(self.max_r,len(regressor_training_set))
        for k,part in enumerate(regressor_training_set[-self.max_r:]):
            if not self.local_regressor.contains_training_instance(part):
                self.local_regressor.add_training_instance(part, regressor_training_fitness[k-valid_examples])
        return self.local_regressor.train()
        
    def train(self, hypercube):
        self.was_trained = True
        if self.classifier.train() and self.train_local(hypercube):
            logging.info("Trained Surrogate Model")
        else:
            logging.info("Couldnt Train Surrogate Model")
            return False
                        
    def get_state_dictionary(self):
        return {"regressor_state_dict" : self.regressor.get_state_dictionary(), "classifier_state_dicts" : self.classifier.get_state_dictionary(), 
            "local_regressor_state_dict":self.local_regressor.get_state_dictionary()}
        
    def set_state_dictionary(self, dict):
        self.regressor.set_state_dictionary(dict["regressor_state_dict"])
        self.local_regressor.set_state_dictionary(dict["local_regressor_state_dict"])
        self.classifier.set_state_dictionary(dict["classifier_state_dicts"])

    
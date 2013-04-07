import logging

from classifiers import Classifier, SupportVectorMachineClassifier
from regressors import Regressor, GaussianProcessRegressor, GaussianProcessRegressor2, GaussianProcessRegressor3, KMeansGaussianProcessRegressor, DPGMMGaussianProcessRegressor

from utils import numpy_array_index
from scipy.interpolate import griddata
from numpy import linspace, meshgrid, reshape, array, argmax, mgrid, ones

class SurrogateModel(object):

    def __init__(self, configuration, controller):
        self.configuration = configuration
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
        
class DummySurrogateModel(SurrogateModel):

    ## TODO - add dummy regressor/classifier
    def __init__(self, configuration, controller):
        super(DummySurrogateModel, self).__init__(configuration,
                                                   controller)
        self.regressor = Regressor(controller, configuration)
        self.classifier = Classifier()

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
        
class ProperSurrogateModel(SurrogateModel):

    def __init__(self, configuration, controller):
        super(ProperSurrogateModel, self).__init__(configuration,
                                                   controller)
        self.controller = controller
        self.configuration = configuration

        if configuration.classifier == 'SupportVectorMachine':
            self.classifier = SupportVectorMachineClassifier()
        else:
            logging.error('Classifier type ' + str(configuration.classifier) + '  not found')
        self.regressor = self.regressor_constructor()

    def predict(self, particles):
        MU, S2 = self.regressor.predict(particles)
        return self.classifier.predict(particles), MU, S2

    def train(self, hypercube):
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
        else:
            raise Exception('Regressor type ' + str(configuration.regressor) + '  not found')
        
    def add_training_instance(self, part, code, fitness, addReturn):
        self.classifier.add_training_instance(part, code)
        if addReturn[0] == 0: ## only update regressor if the fitness function produced a result
            self.regressor.add_training_instance(part, fitness)
        
    def contains_training_instance(self, part):
        return self.regressor.contains_training_instance(part) or self.classifier.contains_training_instance(part)  

    def get_training_instance(self, part):
        code = self.classifier.get_training_instance(part) 
        fitness = None
        if self.regressor.contains_training_instance(part):
            fitness = self.regressor.get_training_instance(part)            
        return code, fitness
        
    def model_failed(self, part):
        return False

    def max_uncertainty(self, designSpace, hypercube=None, npts=10):
        if len(designSpace)==2:
            # make up data.
            if hypercube:
                logging.info("[returnMaxS2]: using hypercube ",hypercube)
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
                #logging.info("hypercube" + str(hypercube))
                logging.info(str(hypercube))
                logging.info(str(hypercube[0]))
                logging.info(str(hypercube[1]))
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
            filteredS2=[]
            filteredZ=[]
            for i,s2 in enumerate(S2):
                if zClass[i]==0.0:
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

    def __init__(self, configuration, controller):
        super(LocalSurrogateModel, self).__init__(configuration,
                                                   controller)
        self.max_r = 10
        self.local_regressor = self.regressor_constructor()
        self.use_local = False
        
    def predict(self, particles):
        if self.use_local:
            MU, S2 = self.local_regressor.predict(particles)
        else:
            MU, S2 = self.regressor.predict(particles)
        return self.classifier.predict(particles), MU, S2
        
                
    def train_local(self, hypercube):
        self.local_regressor = self.regressor_constructor()
        regressor_training_fitness = self.regressor.get_training_fitness()
        regressor_training_set = self.regressor.get_training_set()
        [maxDiag,minDiag] = hypercube
        for k,part in enumerate(regressor_training_set):
            if all(part <= maxDiag):
                if all(part >= minDiag):
                    self.local_regressor.add_training_instance(part, regressor_training_fitness[k])
        ## add most recent
        valid_examples = max(self.max_r,len(regressor_training_set))
        for k,part in enumerate(regressor_training_set[-conf.max_r:]):
            if not any([array_equal(part,pp) for pp in limitedGpTrainingSet]):
                self.local_regressor.add_training_instance(part, regressor_training_fitness[k-valid_examples])
        return self.local_regressor.train()
        
    def train(self, hypercube):
        self.was_trained = True
        self.use_local = False
        classifier_trained = self.classifier.train()
        regressor_trained = self.regressor.train()
        if not regressor_trained:
            regressor_trained = self.train_local(hypercube)
            self.use_local = regressor_trained
        if classifier_trained and regressor_trained:
            if self.use_local:
                logging.info("Trained Local Surrogate Model")
            else:
                logging.info("Trained Surrogate Model")
        else:
            logging.info("Couldnt Train Surrogate Model")
            return False
                        
    def get_state_dictionary(self):
        return {"regressor_state_dict" : self.regressor.get_state_dictionary(), "classifier_state_dicts" : self.classifier.get_state_dictionary(), "use_local" : self.use_local,
            "local_regressor_state_dict":self.local_regressor.get_state_dictionary()}
        
    def set_state_dictionary(self, dict):
        self.regressor.set_state_dictionary(dict["regressor_state_dict"])
        self.regressor.set_state_dictionary(dict["local_regressor_state_dict"])
        self.classifier.set_state_dictionary(dict["classifier_state_dicts"])
        self.use_local = dict["use_local"]


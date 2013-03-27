import logging

from classifiers import Classifier, SupportVectorMachineClassifier
from regressors import Regressor, GaussianProcessRegressor, GaussianProcessRegressor2, KMeansGaussianProcessRegressor, DPGMMGaussianProcessRegressor

from utils import numpy_array_index
from scipy.interpolate import griddata
from numpy import linspace, meshgrid, reshape, array, argmax, mgrid, ones

class SurrogateModel(object):

    def __init__(self, configuration, controller):
        self.configuration = configuration
        self.was_trained = False
        
    def train(self):
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

    def train(self):
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

        if configuration.classifier == 'SupportVectorMachine':
            self.classifier = SupportVectorMachineClassifier()
        else:
            logging.error('Classifier type ' + str(configuration.classifier) + '  not found')

        if configuration.regressor == 'GaussianProcess':
            self.regressor = GaussianProcessRegressor(controller, configuration)
        elif configuration.regressor == 'GaussianProcess2':
            self.regressor = GaussianProcessRegressor2(controller, configuration)        
        elif configuration.regressor == 'KMeansGaussianProcessRegressor':
            self.regressor = KMeansGaussianProcessRegressor(controller, configuration)        
        elif configuration.regressor == 'DPGMMGaussianProcessRegressor':
            self.regressor = DPGMMGaussianProcessRegressor(controller, configuration)
        else:
            logging.error('Regressor type ' + str(configuration.regressor) + '  not found')
                
    def predict(self, particles):
        for i in range(0,3):
            MU, S2 = self.regressor.predict(particles)
            if (not (MU is None)) or (not (S2 is None)) :
                return self.classifier.predict(particles), MU, S2
            logging.info("Regressor prediction failed, attempting shuffle and retraining.. attempt " + str(i) + " out of 3")
            self.regressor.shuffle()
        return self.classifier.predict(particles), MU, S2
                
    def train(self):
        self.was_trained = True
        class_trained = self.classifier.train() 
        for i in range(0,3):
            if self.regressor.train():
                return class_trained
            logging.info("Regressor training failed, attempting shuffle and retraining.. attempt " + str(i) + " out of 3")
            self.regressor.shuffle()
        return False
        
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
                logging.info("kurwa...")
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


import logging

from classifiers import Classifier, SupportVectorMachineClassifier
from regressors import Regressor, GaussianProcessRegressor, GaussianProcessRegressor2

from utils import numpy_array_index
from scipy.interpolate import griddata
from numpy import linspace, meshgrid, reshape, array, argmax

class SurrogateModel(object):

    def __init__(self, configuration, controller):
        self.configuration = configuration
        self.classifier = Classifier()
        self.regressor = Regressor(controller)        
        self.was_trained = False
        
    def train(self, pop):
        raise NotImplementedError('SurrogateModel is an abstract class, this '
                                  'should not be called.')
    def trained(self):
        return self.was_trained
                                  
    def predict(self, particles):
        MU, S2 = self.regressor.predict(particles)
        return self.classifier.predict(particles), MU, S2

    def add_training_instance(self, part, code, fitness):
        pass
        
    def contains_training_instance(self, part):
        pass    
        
    def get_training_instance(self, part):
        pass

    def __getstate__(self):
        # Don't pickle fitness and configuration
        d = dict(self.__dict__)
        del d['configuration']
        del d['fitness']
        return d

    def contains_particle(self, part):
        pass
        
    def particle_value(self, part):
        pass
    
    def model_failed(self, part):
        pass
        
    def max_uncertainty(self):
        pass

    def get_state_dictionary(self):
        return {"regressor_state_dict" : self.regressor.get_state_dictionary(), "classifier_state_dicts" : self.classifier.get_state_dictionary()}
        
    def set_state_dictionary(self, dict):
        self.regressor.set_state_dictionary(dict["regressor_state_dict"])
        self.classifier.set_state_dictionary(dict["classifier_state_dicts"])
        
class DummySurrogateModel(SurrogateModel):

    def train(self, pop):
        self.was_trained = True
        return True

    def model_particle(self, particle):
        return 0, 0, 0
        
    def contains_training_instance(self, part):
        return False

    def model_failed(self, part):
        return False
        
class ProperSurrogateModel(SurrogateModel):

    def __init__(self, configuration, controller):
        super(ProperSurrogateModel, self).__init__(configuration,
                                                   controller)

        if configuration.classifier == 'SupportVectorMachine':
            self.classifier = SupportVectorMachineClassifier()
        else:
            logging.error('Classifier type {} not found'.format(
                configuration.classifier))

        if configuration.regressor == 'GaussianProcess':
            self.regressor = GaussianProcessRegressor(controller)
        elif configuration.regressor == 'GaussianProcess2':
            self.regressor = GaussianProcessRegressor2(controller)
        else:
            logging.error('Regressor type {} not found'.format(
                configuration.regressor))

    def train(self, pop):
        dimensions = len(pop[0])
        self.was_trained = True
        return self.classifier.train(pop) and self.regressor.train(
            pop, self.configuration, dimensions)
        
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

    def max_uncertainty(self, designSpace, hypercube=None, npts=200):
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
            x,y,v = mgrid[designSpace[0]["min"]:designSpace[0]["max"]:(int(designSpace[0]["max"]-designSpace[0]["min"])+1)*1.0j,designSpace[1]["min"]:designSpace[1]["max"]:(int(designSpace[1]["max"]-designSpace[1]["min"])+1)*1.0j , designSpace[2]["min"]:designSpace[2]["max"]:(int(designSpace[2]["max"]-designSpace[2]["min"])+1)*1.0j]
            x=reshape(x,-1)
            y=reshape(y,-1)
            v=reshape(v,-1)
            z = array([[a,b,c] for (a,b,c) in zip(x,y,v)])
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
            logging.error("Finding max S2 failed: {}".format(e))
            return None

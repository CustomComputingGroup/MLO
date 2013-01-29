from math import e, pow
from deap import benchmarks
from numpy import array

dimensions = 2  # Dimensionality of solution space

# Min and max fitness values
minVal = 0.0
maxVal = 10.0

# Defines names for classification codes
error_labels = {0: 'Valid', 1: 'Invalid'}
rotate = False


# Defines the problem to be maximization or minimization
def is_better(a, b):
    return a < b

worst_value = maxVal


# Example fitness function for surrogate model testing
def fitnessFunc(part):
    code = 0 if is_valid(part) else 1
    return benchmarks.sphere(part), array([code]), array([0])
    


# Example function to define if a design is valid or invalid
def is_valid(part):
    return True#part[0] ** int(part[1]) / part[1] < e


# Example Termination condition
def termCond(best):
    return best < 0.00001


# Name of the benchmark
def name():
    return 'sphere'

# Example definition of the design space
designSpace = []
for d in xrange(dimensions):
    designSpace.append({'min': -20, 'max': 20, 'step': 0.5,
                        'type': 'continuous'})

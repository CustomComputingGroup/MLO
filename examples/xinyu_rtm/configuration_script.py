results_folder_path = '/homes/mk306/log'
import os
configuration_folder_path = os.path.split(os.path.realpath(__file__))[0]+"/"
##set to wherever you want the images to be stored
#images_folder_path = 
enable_traceback = True
eval_correct = False

goal = "min"

max_eval = 1
### Basic setup

trials_count = 1
population_size = 30

max_fitness = 200.0
max_iter = 5000
max_speed = 0.2
max_stdv = 0.1
min_stdv = 0.1

surrogate_type = 'proper'  # Can be proper or dummy
F = 30  # The size of the initial training set
M = 10  # How often to perturb the population, used in discrete problems


### Trial-specific variables

#trials_type = 'PSOTrial_TimeAware'
trials_type = 'PSOTrial'

phi1 = 2.0
phi2 = 2.0

weight_mode = 'norm'
max_weight = 1.0
min_weight = 0.4
weight = 1.0

mode = 'exp'
exp = 2.0
admode = 'iter'  # Advancement mode, can be fitness
max_eval = 3
applyK = False
KK = 0.73

a="a1"
### Visualisation

vis_every_X_steps = 1 # How often to visualize
counter = 'g'  # The counter that visualization uses as a 'step'
max_counter = max_iter  # Maximum value of counter

### Regressor and classifier type
#regressor = 'KMeansGaussianProcessRegressor'
regressor = 'GaussianProcess3'
classifier = 'SupportVectorMachine'

### GPR Regression settings
regr = 'linear'
corr2 = 'squared_exponential'
corr = 'matern3'
theta0 = 0.01
thetaL = 0.0001
thetaU = 3.0
nugget = 3
random_start = 10
run_name= corr + "_" + surrogate_type + "_" + trials_type

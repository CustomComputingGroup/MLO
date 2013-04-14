results_folder_path = '/homes/mk306/log'
configuration_folder_path = os.path.split(os.path.realpath(__file__))[0]+"/"
##set to wherever you want the images to be stored
#images_folder_path = 
enable_traceback = True
eval_correct = False


### Basic setup

trials_count = 1
population_size = 20

max_fitness = 100.0
max_iter = 5000
max_speed = 0.01
max_stdv = 0.05

surrogate_type = 'local'  # Can be proper or dummy
F = 10  # The size of the initial training set
M = 20  # How often to perturb the population, used in discrete problems
max_eval = 1

### Trial-specific variables

trials_type = 'PSOTrial'

phi1 = 2.0
phi2 = 2.0

weight_mode = 'norm'
max_weight = 1.0
min_weight = 0.4
weight = 1.0

mode = 'exp'
exp = 1.5
admode = 'iter'  # Advancement mode, can be fitness

applyK = False
KK = 0.73


### Visualisation

vis_every_X_steps = 5 # How often to visualize
counter = 'g'  # The counter that visualization uses as a 'step'
max_counter = max_iter  # Maximum value of counter

# Default values for describing graph visualization
plot_view = 'default'

### Regressor and classifier type
#regressor = 'KMeansGaussianProcessRegressor'
regressor = 'GaussianProcess3'
classifier = 'SupportVectorMachine'

### GPR Regression settings
regr = 'quadratic'
corr2 = 'squared_exponential'
corr = 'anisotropic'
theta0 = 0.01
thetaL = 0.00001
thetaU = 3.0
nugget = 5
random_start = 100

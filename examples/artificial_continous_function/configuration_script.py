results_folder_path = '/data/log'

##set to wherever you want the images to be stored
#images_folder_path = 
enable_traceback = True
eval_correct = False


### Basic setup

trials_count = 1
population_size = 20

max_fitness = 50.0
max_iter = 5000
max_speed = 0.01
max_stdv = 0.05

surrogate_type = 'proper'  # Can be proper or dummy
F = 10  # The size of the initial training set
M = 100  # How often to perturb the population, used in discrete problems


### Trial-specific variables

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

applyK = False
KK = 0.73


### Visualisation

vis_every_X_steps = 10 # How often to visualize
counter = 'g'  # The counter that visualization uses as a 'step'
max_counter = max_iter  # Maximum value of counter

# Default values for describing graph visualization
plot_view = 'default'

### Regressor and classifier type
regressor = 'GaussianProcess2'
classifier = 'SupportVectorMachine'

### GPR Regression settings
#regr = 'quadratic'
#corr = 'squared_exponential'
corr = 'isotropic'
theta0 = 0.01
thetaL = 0.0001
thetaU = 3.0
nugget = 3
random_start = 200
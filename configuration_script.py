from trial import PSOTrial

results_folder_path = 'data/log'
enable_traceback = True
eval_correct = False


### Basic setup

trials_count = 1
population_size = 10

max_fitness = 250.0
max_iter = 1
max_speed = 0.025
max_stdv = 0.05

surrogate_type = 'proper'  # Can be proper or dummy
F = 10  # The size of the initial training set
M = 2  # How often to perturb the population, used in discrete problems


### Trial-specific variables

trials_type = PSOTrial

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

vis_every_X_steps = 1  # How often to visualize
counter = 'g'  # The counter that visualization uses as a 'step'
max_counter = max_iter  # Maximum value of counter

# Default values for describing graph visualization
graph_title = 'Title'
graph_names = ['Progression', 'Fitness', 'Mean', 'SVM']

graph_dict1 = {'subtitle': 'Currently Best Found Solution',
               'x-axis': 'Iteration',
               'y-axis': 'Fitness',
               'font size': '10',
               'position': '221'}
graph_dict2 = {'subtitle': 'Fitness Function',
               'x-axis': 'X',
               'y-axis': 'Y',
               'z-axis': 'Fitness',
               'font size': '10',
               'colour map': 'PuBu',
               'position': '222'}
graph_dict3 = {'subtitle': 'Regression Mean',
               'x-axis': 'X',
               'y-axis': 'Y',
               'z-axis': 'Fitness',
               'font size': '10',
               'colour map': 'PuBuGn',
               'position': '223'}
graph_dict4 = {'subtitle': 'SVM',
               'x-axis': 'X',
               'y-axis': 'Y',
               'font size': '10',
               'colour map': 'PuBu',
               'x-colour': 'black',
               'o-colour': 'black',
               'position': '224'}

all_graph_dicts = {'Progression': graph_dict1,
                   'Fitness': graph_dict2,
                   'Mean': graph_dict3,
                   'SVM': graph_dict4}

plot_view = 'default'


### Regressor and classifier type
regressor = 'GaussianProcess'
classifier = 'SupportVectorMachine'


### Regression
regr = 'quadratic'
corr = 'squared_exponential'
theta0 = 0.01
thetaL = 0.001
thetaU = 3.0
nugget = 3

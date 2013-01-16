***********************************************************************************************************************
MLO Readme
************************************************************************************************************************
About MLO

The optimization of reconfigurable applications often requires
substantial effort from the designer who has to analyze the
application, create models and benchmarks and subsequently use them to
optimize the application. One could try to employ exhaustive search of
the application parameter space to carry out optimization yet it is
unrealistic since benchmark evaluations involve bit-stream generation
and code execution which takes hours of computing time. Recently it
has been shown useful to use surrogate models combined with fitness
functions for computationally expensive optimization problems in
various fields. As these models are orders of magnitude cheaper, they
can substantially decrease optimization cost thus allowing for an
automated approach. This is the motivation behind Machine Learning
Optimizer which we apply to non-linear and multi-modal problem of
heterogeneous application parameter optimization. We use regressors to
model performance of the design like execution time or throughput,
while searching for the global optimum using meta-heuristics. We
classify the parameter space using support vector machines to identify
designs that would fail constraints; over-map on resources, produce
inaccurate results or other.

************************************************************************************************************************
MLO v1.0

I developed the initial version of MLO incrementally, adding feautres to fix problems I encountered. Lack of initially specified architecture resulted in choatic and unstable code. Just use what you need.

Installation Directory Includes
first iteration of MLO:

run_optimizer.sh script that configures the environment and runs iterative_optimizer.py

iterative_optimizer.py script that runs intial version of mlo. configuration and benchmark scripts should work with this version of the optimizer... possibly with a little work. only look at it for reference

run_deap.sh script that configures the environment and runs deap_basic.py
deap_basic.py current version of MLO. The script will try to optimize the benchmark containted within supplied benchmark script, using the provided configuration file for mlo. functions should be fairly obvious and are commented. start looking at main and follow to run. 

confTemplate.py basic confiugartion with all the parameters explained.

/ and examples/ containts all the benchmark I currently posses. 

Installation
installation should be fairly obvious due to pythons nature.. its all scripted :) Make sure Sklearn, Scipy and Numpy are installed.. all should work fine. if not contact me. Get deap from, http://code.google.com/p/deap/. 

NOTE: might be worth to clean up the dir so that svn files dont screw version control system that you choose!


Running MLO
./run_deap.sh benchmark_script configuration_script and thats it!

for example 
./run_deap.sh examples/benchmark_fitness2.py  confTemplate.py 

Current benchmark examples

1.examples/benchmark_fitness.py reconfigurable radio optimization. The data is stored in a form of regression results. 
2.examples/benchmark_fitness2.py 2 or 3 dimensional optimization.. Data that I collected is stored in external file and used by the script to build fitness function. Uses AnsonExec.csv and AnsonCores.csv as well as script generateFitnessFunction.py to create fitness function. Probably should tidy it up. 
3.fitnessTemplate.py  continuous and very simple examples of possible fitness functions. They are configurable in respect to number of dimensions and search space. For reference look into : http://deap.gel.ulaval.ca/doc/dev/api/benchmarks.html#module-deap.benchmarks
Just sub “return (benchmarks.rosenbrock(particle),array([0.0]),array([0.0]))” to match the appropriate function. For example sphere instead of rosenbrock. 

Configuraition script parameters
optGoal = "max" deprecated
# PSO configuration
F = 10 the number of fitness functions the optimizer should evaluate intially. If it wont be able to build a model it is going to include extra evaluations. Setting F>M has no implication.  F should be bigger then 1. 
M = 1 How often to evaluate gbest using actuall fitness function
N = 20 Number of particles
maxFitness = 250.0 Maximum number of fitness function
maxIter = 10000.0 Maximum number of iterations
maxspeed = 0.025 deprecated
phi1 = 2.0 PSO parameter, c1
phi2 = 2.0 PSO parameter, c2
##Classifier
addClassifier = True’not sure if it is still working...
#Restart PSO every M iterations? (randomize particles)
restart = False If turned on every M iterations PSO is going to be restared. Training sets use old models
reevaluate = True Should we reevalute particles using the models, even if they have fitness which was evaluated using actuall fitness function
discardNegative = False Should values predicted as negative by the surrogate model be roundedto 0?
reevaluteModels = False Sho
trialName = "base" 
purePSO = False deprecated
randomization = False dont remmeber
alwaysEvaluate = False dont remmeber
maxEvaluations = 1 dont remmeber
evalCorrect = False dont remmeber
#Adjustment mode fitness,iter
admode = "iter" velocity clamping of parcitles can be based on the number of ffitness function evaluations or  current iteration

#Speed clamping - vp,norm,exp,no
mode = "exp" PSO speed velocity clamping, depending which we use followingparameters become significant.
# apply K coeff, any norm.. KK is the K parameter
applyK = False velocity clamping mechanism
KK = 0.73 velocity clamping mechanism
## for vp
K = 0.75 velocity clamping mechanism
p = 3 velocity clamping mechanism
## for exp
exp = 2.0 velocity clamping mechanism
#norm,linear,

weightMode = "norm" particle previous velocity adjustment mechanism... same a
#norm
weight = 1.0
#linear
minWeight = 0.4
maxWeight = 1.0
trials = 3 number of trials per run. 

# GP configuration GP surrogate model settings... this is a bit complicated.  The following parameters are used to set up  kernel functions and the GP log likelihood maximizer. It is going to become important later on, but at the moment is irrelevant to you. 
maxR = F
nClosest = N
maxstdv = 0.05
theta0=0.01
thetaL=0.001
thetaU=3.0
random_start=20
nugget=0.0
regr = "linear"
corr = "squared_exponential"
# isotropic or anisotropic
type = "anisotropic"
# Results Formatting configuration
maxNoise = -3
maxOther = 1.5
minOther = -3

svm_kernel = "rbf" svm kernel
logFolder = "/data/log/" + type + "contours/fast0.1" specify where the results aregoing to be stored
trialName = ""
visEveryXsteps = 1 how often do you want to render images? 
visOn = True Turn visualization on/off
AddLocalModel = False 
bestPert = False should we evaluate a perturbation of currently best found solution in case of model stagnation
maxevaluate = 1 how many particles can we evaluate at most without rebuilding the surrogate model? 
gp="gpr" there are two gp libraries you can use. 
makeVideo = False do you want to create a video using the rendered images . 


************************************************************************************************************************
MLO v2.0

An extension to the rudimentary MLO v1.0 implementation of the MLO tool based on Python. This tool has been redeveloped 
using professional software development practices.

************************************************************************************************************************
Repository Structure


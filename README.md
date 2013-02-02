***********************************************************************************************************************
#MLO Readme
************************************************************************************************************************

##About MLO
-----------------------------------------------------------------------------------------------------------------------

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

##Directory Structure

-----------------------------------------------------------------------------------------------------------------------
###doc
-----------------------------------------------------------------------------------------------------------------------

Contains all of the reports, papers and documentation related to MLO development. 

-----------------------------------------------------------------------------------------------------------------------
###publications
-----------------------------------------------------------------------------------------------------------------------

For each paper published and related to MLO please create a seperate folder containing the relevant LaTeX code. The naming
convention for the folder is as follows: conferencename_submissionnumber

-----------------------------------------------------------------------------------------------------------------------
###examples
-----------------------------------------------------------------------------------------------------------------------

Contains reconfigurable computing fitness function examples. The fitness functions in this directory should be based on
well documentated papers, and used for research purpose. Preferably csv files containing the fitness functions and scripts
used to obtain this are provided. 

Each folder within examples directory contains a fitnes and configuration script. each of the folder contains a small 
README.txt file refereing to fitness function relevant literature. 

-----------------------------------------------------------------------------------------------------------------------
###testing
-----------------------------------------------------------------------------------------------------------------------

testing contains all of the testing scripts. Currently implementing regression testing. 

-----------------------------------------------------------------------------------------------------------------------
###views
-----------------------------------------------------------------------------------------------------------------------

creates the code of VIEW part of the MVC system. All modes, currently GUI and TERMINAL, reside in this directory. All 
the possible visualizers are also containted in this directory. 

#### modes.py script contains the code that starts up relevant mode. Currently either terminal or gui. 

#### gui folder contains gui code. 

#### visualizers folder contains model visuzalization relevant scripts. Visualizers can be used to generate reports, 
images, plots and others. Currently only plot visuzalizer of the PSOTrial is avaiable (MLOImageViewer). 

all the viewers are notified of changes within the model by the update method call. 

-----------------------------------------------------------------------------------------------------------------------
###controller
-----------------------------------------------------------------------------------------------------------------------

Contains code of the controller of the applications. MODEL and VIEW use it to communiate with each other. 
It also contains code of visualizer, which is used to generate any plots and reports.  

#### controller.py 
it is a script used to define top level of the controller. It contains methods to provide comunication
between the VIEW and MODEL.

#### visualizer.py
It is a pool of workers that recieves visualization jobs from the controller/viewer. It recieves
a snapshot of the model and a reference to a visualization function. currently it is a multi-threaded process, it could
be ported onto a computer cluster. 

-----------------------------------------------------------------------------------------------------------------------
###model
-----------------------------------------------------------------------------------------------------------------------

contains code of the optimization algorithms, runs, trials and surrogate models. It contains code to back them up
based on save data. 

#### surrogatemodels 
contains code of the avaiable regressors, classifiers and surrogate model frameworks. Currently there are two flavours
of GPR available for regression , and one flavour of SVM for classification. Feel free to extend if neccesary. 

#### trials/trial 
contains the code of all possible trial types. Other scritps within the trials directory are specific
to trial types. Currently only one flavour of MLO is implemented the PSOTrial, which with the GP/SVM surrogate 
model is the MLO algorithm presented in the FPT'12/ARC'13 papers. 

#### run.py 
scripts prepares a bunch of trials of the same type, using a configuration script, and runs them. 

-----------------------------------------------------------------------------------------------------------------------
###scripts within the directory
-----------------------------------------------------------------------------------------------------------------------

#### optimizer.py 
the optimizer initialization script

#### utils.py
some utility functions used throught the whole application. 
************************************************************************************************************************

##Version convention vA.B.C.

-----------------------------------------------------------------------------------------------------------------------
###A
-----------------------------------------------------------------------------------------------------------------------

Within a revision starting with number A the fitness and configuration scripts should be compatible. 

-----------------------------------------------------------------------------------------------------------------------
###B
-----------------------------------------------------------------------------------------------------------------------

Minor revisions can introduce changes to the methodology of upkeeping of application state. Before updating a minor revison
please ensure that all of the current runs and trials have finished. 

-----------------------------------------------------------------------------------------------------------------------
###C
-----------------------------------------------------------------------------------------------------------------------

Extra features added with no implication on the state of the application. 

************************************************************************************************************************

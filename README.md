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

-----------------------------------------------------------------------------------------------------------------------
###scripts within the directory
-----------------------------------------------------------------------------------------------------------------------
For detailed architecture of the application please refer to doc/groupproj-rep13.pdf


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

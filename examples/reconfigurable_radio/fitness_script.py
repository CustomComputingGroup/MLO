import math
import operator
import csv

from deap import base
from deap import creator
from deap import tools

from numpy import *
from copy import deepcopy 
from numpy.random import uniform, seed,rand

cost_maxVal = 1.0
cost_minVal = 0.0

initMin = -1
initMax = 1

global reconfiguration_time_set
rotate = False

#aecc or execution time
functionType = "execution_time"

reconfiguration_time_set = 0

designSpace = [ #[min,max,stepSize]#
                    {"min":1.0,"max":112.0,"step":1.0,"type":"discrete","smin":-35.0,"smax":35.0},
                    {"min":200.0,"max":300.0,"step":1.0,"type":"discrete","smin":-45.0,"smax":45.0},
                    #{"min":0.0,"max":1.0,"step":1.0,"type":"discrete","smin":-1.0,"smax":1.0}
                    #[0,1,1]
              ]
              
              
##minimization
def is_better(a, b):
    return a < b
minVal = 0.0
if reconfiguration_time_set == 0:
    maxvalue = 10.0
    
else:
    maxvalue = 100.0
maxVal = maxvalue
worst_value = maxVal
def name():
    return "radio" + str(designSpace[1]["min"]) + "_" + str(reconfiguration_time_set)
        
error_labels = {0:'Valid',1:'Timing',2:'Overmap'}

def get_z_axis_name():
    return "$t_{total}$ (ms)"
    
def get_x_axis_name():
    return "p"
     
def get_y_axis_name():
    return "freq. (MHz)"

def termCond(best):
    global reconfiguration_time_set
    if reconfiguration_time_set == 0:   
        return best < 0.526             
    else:
        return best < 8.38
                       
def fitnessFunc(particle):
    global reconfiguration_time_set,maxvalue
    # Dimensions dynamically rescalled
    # All 
    ############Dimensions
    p = int(particle[0])
    frequency = int(particle[1])
    #particle[1]#int((particle[2]+1)/1.5) ## 0 or 1 value
    #######################
    # optimial setting for level of parallelism for fast reconfiguration is 18 cores. 
    ####
    steps = 112.0
    
    if reconfiguration_time_set == 0:
        reconfiguration_time = 0.011368403 * p + 0.080535773 ### ms min - [  20.06060606  259.09090909]   0.524120049216
    else:
        reconfiguration_time = 0.681703026 * p + 4.843653703 ### ms min - [   3.24242424  246.96969697]   8.40637795715
        
    proccesing_time = 1000.0/frequency ### ns
    proccesing_time = proccesing_time * 1e-6 ### ms
    n = 10000.0    
    
    ## check area constraints
    max_slices = 7000.0
    slices = 400.0 + 59.0 * p
    
    ## check if the frequency is achievable
    maxfreq = 262 + -0.13 * p
    
    ## check application throughput
    minimal_app_throughput = 4.93 ## ms
    
    #print "p ",p," frequency ",frequency," reconfiguration_time ",reconfiguration_time," t_total ",t_total," proccesing_time ",( n * (steps * proccesing_time) / p)
    cost = 0.5
    if frequency > maxfreq:
        return (array([maxvalue]),array([1]),array([1]),array([cost]))
        
    if slices > max_slices:
        return (array([maxvalue]),array([2]),array([1]),array([cost]))
    
    t_total = ( n * (steps * proccesing_time) / p) + reconfiguration_time
    
    #if t_total > minimal_app_throughput:
    #    return array([10.0])
        
    if functionType == "execution_time":
        return (array([t_total]),array([0]),array([0]),array([cost]))

    elif functionType == "aecc":
    
        apcc = p / steps * frequency
        aecc = ( n * (steps * proccesing_time) / p) * apcc
        return (array([aecc]),array([0]),array([0]),array([cost]))
        
def changeParams(params):
    global designSpace,reconfiguration_time_set,maxvalue,maxVal,minVal
    try:
        reconfiguration_time_set = params["reconfiguration_time_set"]
    except Exception,e:
        print "[changeParams] ",e
        pass
    try:
        designSpace = [ #[min,max,stepSize]#
                        {"min":1.0,"max":112.0,"step":1.0,"type":"discrete","smin":-35.0,"smax":35.0},
                        {"min":params["min"],"max":300.0,"step":1.0,"type":"discrete","smin":-45.0,"smax":45.0},
                        #{"min":0.0,"max":1.0,"step":1.0,"type":"discrete","smin":-1.0,"smax":1.0}
                        #[0,1,1]
                  ]
        minVal = 0.0          
        if reconfiguration_time_set == 0:
            maxvalue = 10.0    
        else:
            maxvalue = 100.0
            
        maxVal = maxvalue
        worst_value = maxVal
    except Exception,e:
        print "[changeParams] ",e
        pass
        
def trialgen():
    params = {}
    global reconfiguration_time_set
    for reconfiguration_time_set in [0,1]:
        for min in [200.0]:
            params["min"] = min
            params["reconfiguration_time_set"] = reconfiguration_time_set
            changeParams(params)
            yield params

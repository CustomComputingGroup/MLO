import math
import operator
import csv

from deap import base
from deap import creator
from deap import tools

from numpy import *
from copy import deepcopy 
from numpy.random import uniform, seed,rand

from sklearn.gaussian_process import GaussianProcess
from sklearn import svm
from sklearn import preprocessing


import traceback

from matplotlib import pyplot
from matplotlib import cm
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_pdf import PdfPages

import os
import traceback
import sys

import subprocess
import threading, os
from time import gmtime, strftime
from copy import deepcopy    
from matplotlib.ticker import MaxNLocator
    
enableTracebacks = True
from numpy import * 
initMin = -1
initMax = 1

#aecc or execution time -- when error correction applied
#### [ 15.   6.]   75.3885409418 0.1
#### [ 20.  11.]   11.1996923818 0.05
#### [ 19.  22.]   1.44387061283 0.01
#### [ 20.  32.]   0.469583069715 0.001

#### without error correction
#### [ 14.  11.   4.]   259.663179761 all
#### [ 13.  15.   7.]   46.8546137378 0.001

##maximization
def is_better(a, b):
    return a > b

cost_maxVal = 15000.0
cost_minVal = 0.0
    
doMw = True
doCores = False
doDf = True
rotate = True
designSpace = []
maxError = 0.01
errorCorrection=True

minVal = 0.0
maxVal = 300.0
worst_value = 0.0
optVal = {0.1:75.37,0.01:1.44,0.05:11.1,0.001:0.469 }[maxError]


## always do mw
designSpace.append({"min":11.0,"max":53.0,"step":1.0,"type":"discrete","smin":-5.0,"smax":5.0, "set":"h"}) #a
designSpace.append({"min":11.0,"max":53.0,"step":1.0,"type":"discrete","smin":-5.0,"smax":5.0, "set":"h"}) #b 
designSpace.append({"min":11.0,"max":53.0,"step":1.0,"type":"discrete","smin":-5.0,"smax":5.0, "set":"h"}) #B
designSpace.append({"min":11.0,"max":53.0,"step":1.0,"type":"discrete","smin":-5.0,"smax":5.0, "set":"h"}) #T
if doFrequency:
    designSpace.append({"min":1.0,"max":16.0,"step":1.0,"type":"discrete","smin":-1.0,"smax":1.0, "set":"h"})
designSpace.append({"min":11.0,"max":53.0,"step":1.0,"type":"discrete","smin":-5.0,"smax":5.0, "set":"h"}) #Ob
designSpace.append({"min":11.0,"max":53.0,"step":1.0,"type":"discrete","smin":-5.0,"smax":5.0, "set":"h"}) #Ot
if doParallelism:
    designSpace.append({"min":11.0,"max":53.0,"step":1.0,"type":"discrete","smin":-5.0,"smax":5.0, "set":"h"}) #Pdp
    designSpace.append({"min":11.0,"max":53.0,"step":1.0,"type":"discrete","smin":-5.0,"smax":5.0, "set":"h"}) #Pknl
    designSpace.append({"min":11.0,"max":53.0,"step":1.0,"type":"discrete","smin":-5.0,"smax":5.0, "set":"h"}) #Ptl
designSpace.append({"min":11.0,"max":53.0,"step":1.0,"type":"discrete","smin":-5.0,"smax":5.0, "set":"h"}) #Bs
designSpace.append({"min":11.0,"max":53.0,"step":1.0,"type":"discrete","smin":-5.0,"smax":5.0, "set":"h"}) #Ds
designSpace.append({"min":11.0,"max":53.0,"step":1.0,"type":"discrete","smin":-5.0,"smax":5.0, "set":"h"}) #Ls
designSpace.append({"min":11.0,"max":53.0,"step":1.0,"type":"discrete","smin":-5.0,"smax":5.0, "set":"h"}) #Fs

maxvalue = 0.0

error_labels = {0:'Valid',1:'Overmap',3:'Inaccuracy'}

def termCond(best):
    global optVal
    return best > optVal

def name():
    return "rtm_" + str(maxError) + "_doCores" + str(doCores) + "_doDf" + str(doDf) + "_errorCorrection" + str(errorCorrection) 
    
def fitnessFunc(particle, state):
    # Dimensions dynamically rescalled
    ############Dimensions
    if doFrequency:
        cores = int(particle[0])
        mw = int(particle[1])
    else:
        cores = 1.0
        mw = int(particle[0])
                
    if doParallelism:
        df = int(particle[-1])
        if not doCores:
            cores = allData[11][mw][df][0]
            
        error = allData[11][mw][df][1]
        #accuracy = array([allData[11][mw][i+1][1] for i in xrange(32)])[::-1]
                
        if(errorCorrection):
            ###error correction
            #for dff,acc in enumerate(accuracy):
            #    if ((32-dff) > df) and  (acc > error):
            #        error = acc 
                    
            for mww in range(mw,54):
                accuracy = array([allData[11][mww][i+1][1] for i in xrange(32)])[::-1]
                for dff,acc in enumerate(accuracy):
                    if ((32-dff) >= df) and  (acc > error):
                        error = acc 
            #print "mww ",mww," dff ",(32-dff)," acc ",acc," df ",df," mw ",mw," ",error
            #print " df ",df," mw ",mw," ",error
        
    else:
        accuracy = array([allData[11][mw][i+1][1] for i in xrange(32)])[::-1]
        error = accuracy[0]
        df = 0
        for dff,acc in enumerate(accuracy):
            if acc > maxError:
                break
            error = acc
            df = 32 - dff  
        
    frequency = 100
    #######################
    executionTime = array([cores/allData[11][mw][df][2]])
    #executionTime = array([error])
    #print " df ",df," mw ",mw," ",error," ",allData[11][mw][df][2]
    cost, state = getCost(df, float(mw),float(cores), state) ## we need to cast to float
    if error > maxError:
        return ((executionTime, array([3]),array([0]), cost) , state)##!!!! zmien na 0.0 
    ### accuracy error
        
    ### overmapping error
    if allData[11][mw][df][0] < cores :
        return ((array([maxvalue]), array([1]),array([1]), cost) , state)
    
    ### ok values execution time
    return ((executionTime, array([0]),array([0]), cost), state)
    
def calcMax():    
    if len(designSpace)==2:
        # make up data.
        x,y = mgrid[designSpace[0]["min"]:designSpace[0]["max"]:(int(designSpace[0]["max"]-designSpace[0]["min"])+1)*1.0j,designSpace[1]["min"]:designSpace[1]["max"]:(int(designSpace[1]["max"]-designSpace[1]["min"])+1)*1.0j]
        x=reshape(x,-1)
        y=reshape(y,-1)       
        z = array([[a,b] for (a,b) in zip(x,y)])
    else:
        x,y,v = mgrid[designSpace[0]["min"]:designSpace[0]["max"]:(int(designSpace[0]["max"]-designSpace[0]["min"])+1)*1.0j,designSpace[1]["min"]:designSpace[1]["max"]:(int(designSpace[1]["max"]-designSpace[1]["min"])+1)*1.0j , designSpace[2]["min"]:designSpace[2]["max"]:(int(designSpace[2]["max"]-designSpace[2]["min"])+1)*1.0j]
        x=reshape(x,-1)
        y=reshape(y,-1)
        v=reshape(v,-1)
        z = array([[a,b,c] for (a,b,c) in zip(x,y,v)])
        
    zReal = array([fitnessFunc(a)[0][0] for a in z])
    zRealClass = array([fitnessFunc(a)[1][0] for a in z])
    minn = argmin(zReal)
    filteredminn = []
    filteredZ=[]
    for i,zreal in enumerate(zReal):
        if zRealClass[i]==0.0:
            filteredminn.append(zreal)
            filteredZ.append(z[i])
            
    if filteredminn:
        doFor=5
        argsortedFiltered = argsort(filteredminn,axis=0)
        print "[returnMaxS2] ====================================="
        for kk in range(0,doFor):
            minn = argsortedFiltered[kk]
            maxx = argsortedFiltered[-(kk+1)]
            print "[returnMaxS2] Real min :",minn," ",filteredZ[minn]," ",filteredminn[minn]
            print "[returnMaxS2] Real max :",maxx," ",filteredZ[maxx]," ",filteredminn[maxx]

        print "[returnMaxS2] ====================================="

def calcMaxs():    
    global maxError
    for maxError in [0.001]:
        print "maxError: ",maxError
        calcMax()
    
##add state saving, we use p and thread id 
def getCost(df, wF, cores, bit_stream_repo):
    global costModel, costModelInputScaler, costModelOutputScaler
    bit_stream_repo_copy = deepcopy(bit_stream_repo) 
    if bit_stream_repo is None:
        bit_stream_repo_copy = {}
       
    if bit_stream_repo_copy.has_key((wF,cores)): ## bit_stream evalauted
        return array([df*5.0*(0.5 + random.random())]), bit_stream_repo_copy
    else:
        bit_stream_repo_copy[(wF,cores)] = True
        return costModelOutputScaler.inverse_transform(costModel.predict(costModelInputScaler.transform(array([[wF,cores]])))) + (df*5.0*(0.8 + random.random()/5.0)), bit_stream_repo_copy  ##software cost, very small fraction and linear
                
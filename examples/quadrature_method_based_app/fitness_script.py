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

from time import gmtime, strftime
    
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

if doCores:
    designSpace.append({"min":1.0,"max":16.0,"step":1.0,"type":"discrete","smin":-1.0,"smax":1.0})
## always do mw
designSpace.append({"min":11.0,"max":53.0,"step":1.0,"type":"discrete","smin":-5.0,"smax":5.0})

if doDf:
    designSpace.append( {"min":4.0,"max":32.0,"step":1.0,"type":"discrete","smin":-2.0,"smax":2.0})
              
maxvalue = 0.0
## if  [ 14.  11.   4.]   0.000275081632653

#errorLabels = {0:'Valid',1:'Overmap',3:'Inaccuracy'}
error_labels = {0:'Valid',3:'Inaccuracy'}

def get_z_axis_name():
    return "Throughput ($\phi_{int}$)"
    
def get_x_axis_name():
    if doCores:
        return "Cores"
    else:
        return "$m_w$"
        
def get_y_axis_name():
    if doDf:
        return "$d_f$"
    else:
        return "$m_w$"

def termCond(best):
    global optVal
    return best > optVal

def name():
    return "anson_" + str(maxError) + "_doCores" + str(doCores) + "_doDf" + str(doDf) + "_errorCorrection" + str(errorCorrection) 
    
def alwaysCorrect():
    if doCores:
        return array([1.0,53.0,32.0])
    else :
        return array([53.0,32.0]) 
    
def fitnessFunc(particle):
    allData = getAllData()
    # Dimensions dynamically rescalled
    ############Dimensions
    if doCores:
        cores = int(particle[0])
        mw = int(particle[1])
    else:
        cores = 1.0
        mw = int(particle[0])
                
    if doDf:
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
    if error > maxError:
        return (executionTime,array([3.0]),array([1])) ##!!!! zmien na 0.0 
    ### accuracy error
        
    ### overmapping error
    if allData[11][mw][df][0] < cores :
        return (array([maxvalue]),array([1.0]),array([1])) 
    
    ### ok values execution time
    return (executionTime,array([0.0]),array([0])) 
    
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
        
def changeParams(params):
    global designSpace,optVal,errorCorrection,maxError,doCores,maxVal
    try:
        doCores = params["doCores"]
    except Exception,e:
        print "[changeParams] ",e
        pass
    
    try:
        maxError = params["maxError"]
    except Exception,e:
        print "[changeParams] ",e
        pass
        
    try:
        errorCorrection = params["errorCorrection"]
    except Exception,e:
        print "[changeParams] ",e
        pass

    optVal = {0.1:75.37,0.01:1.44,0.05:11.1,0.001:0.469 }[maxError]
    maxVal = {0.1:260.0,0.01:5.0,0.05:25.0,0.001:5.0 }[maxError]

    designSpace=[]
    if doCores:
        designSpace.append({"min":1.0,"max":16.0,"step":1.0,"type":"discrete","smin":-1.0,"smax":1.0})
    ## always do mw
    designSpace.append({"min":11.0,"max":53.0,"step":1.0,"type":"discrete","smin":-5.0,"smax":5.0})

    if doDf:
        designSpace.append( {"min":4.0,"max":32.0,"step":1.0,"type":"discrete","smin":-2.0,"smax":2.0})
                  
def trialgen():
    params = {}
    global maxError,optVal,errorCorrection,doCores
    for maxError in [0.1]:
        for errorCorrection in [True]:
            for doCores in [False]:
                params["maxError"] = maxError
                params["errorCorrection"] = errorCorrection
                params["doCores"] = doCores
                changeParams(params)
                yield params
               
#python data dict creator
#
#AnsonExec.csv
#text delimiter " field delimiter  ,
#
#to get number of cores
#AnsonCores.csv


global allData ## Todo - very bad practice!!! only other thing i could come up with was inlining here...
allData = None

def getAllData():
    global allData
    if allData is None:
        path = os.path.abspath(__file__)
        dir_path = os.path.dirname(path)
        spamReader = csv.reader(open(dir_path + '/AnsonCores.csv', 'rb'), delimiter=',', quotechar='"')
        cores = {11:{}}
        for row in spamReader:
            cores[11][int(row[1])] = int(row[0])

        maxcores = cores
        spamReader = csv.reader(open(dir_path +'/AnsonExec.csv', 'rb'), delimiter=';', quotechar='"')

        allData = {}
        for row in spamReader:
            row_0 = int(row[0])
            row_1 = int(row[1])
            row_2 = int(row[2])
            row_3 = float(row[3])
            row_4 = float(row[4])
            data = [cores[row_0][row_1],row_3,row_4]
            
            try:
                try:
                    allData[row_0][row_1][row_2] = data
                except:
                    allData[row_0][row_1] = {row_2:data}
            except:
                allData[row_0] = {row_1:{row_2:data}}
    #spamReader.close()
#print allData
    return allData
#print allData
#print cores
            
import math
import operator
import csv

from deap import base
from deap import creator
from deap import tools

from numpy import *
from copy import deepcopy 
from numpy.random import uniform, seed,rand

import traceback

from matplotlib import pyplot
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_pdf import PdfPages

import os
import traceback
import sys

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
    return a < b

cost_maxVal = 15000.0
cost_minVal = 0.0
    

designSpace = []
maxError = 0.01

minVal = 0.0
maxVal = 300.0
worst_value = 10000.0

designSpace = []

designSpace.append({"min":0.05,"max":1.0,"step":1.0,"type":"continuous", "set":"h"}) #alpha
designSpace.append({"min":0.05,"max":1.0,"step":1.0,"type":"continuous", "set":"h"}) #beta
designSpace.append({"min":0.05,"max":1.0,"step":1.0,"type":"continuous", "set":"h"}) #B
designSpace.append({"min":0.05,"max":1.0,"step":1.0,"type":"continuous", "set":"h"}) #T

designSpace.append({"min":1.0,"max":20.0,"step":1.0,"type":"discrete", "set":"h"}) #Pdp
designSpace.append({"min":1.0,"max":20.0,"step":1.0,"type":"discrete", "set":"h"}) #Pknl
designSpace.append({"min":1.0,"max":20.0,"step":1.0,"type":"discrete", "set":"h"}) #Ptl

maxvalue = worst_value
error_labels = {0:'Valid',1:'Overmap',2:'Inaccuracy', 3:'Memory'}

def termCond(best):
    return False
    
def fitnessFunc(particle, state):

    # Dimensions dynamically rescalled
    ############Dimensions
    fknl = 50000000.0 ## 100 MHz
    
    alpha = (particle[0])
    beta = (particle[1])
    
    B = (particle[2])
    T = (particle[3])
    
    Pknl = (particle[4])
    Pt = (particle[5])
    Pdp = (particle[6])
        

    ######################
    Rcc = 1.0
    x = 100.0
    y = 100.0
    Nop = 700000.0
    Nc = 10.0
    S=10.0
    D=10.0**9
    
    Wdp = 32.0 # 8 bytes
    N = 1.0
    Bw = 1.0
    Wm = N * (Wdp*Bw*Pdp)
    BWm = 32000000000.0 * 8 ## GB / s
    
    ## not sure
    Bd = 1.0
    Bf = 1.0
    Bl = 1.1
    #epislon = sqrt()
    
    theta = 300000000.0*8.0 # 300 MB /s
    psi = 10000000000.0 *8 # 10 GB/s
    gamma = 5.0 
    Ad = 1000000.0
    Af = 1000000.0
    Al = 1000000.0
    Ab = 8.0 * 64000000.0 ## 64 MB
    
    If = Af * 0.2
    Il = Al * 0.1
        
    ######################
    nx = (((x-2*S)/alpha) + 2*S)
    #print "nx " + str(nx)
    ny = (((y-2*S)/beta) + 2*S)
    #print "ny " + str(ny)
    Ob = ((((x-2*S)/(nx-2*S)) * ((y-2*S)/(ny-2*S)) * nx * ny)/(x*y))
    #print "Ob " + str(Ob)            
    ######################
    if alpha*beta == 1:
        Ot = 1
    else:
        Ot = ((nx+(Pt-1)*2*S)/(nx)) * ((ny+(Pt-1)*2*S)/(ny))
                    
    ######################
    mdb = ((nx + (Pt-1)*2*S) / (Pdp)) * (ny + (Pt-1)*S)
    #print "1 " + str(mdb)
    #print "2 " + str(Pknl*Pt*Pdp)
    Bs = (Pknl*Pt*Pdp*(S*(2+Nc)+1)*mdb) / (Ab/(Wdp*Bw))
    #Bs = (1.0*(S*(2+Nc)+1)*mdb) / (Ab/(Wdp*Bw))
    #print "Bs " + str(Bs)
    Ds = (Nop * T * Bd / Ad)
    #print "DS " + str(Ds)
    Ls = (Nop * T * Bl + Il)/ Al
    #print "Ls " + str(Ls)
    Fs = (Nop * T * Bf + If)/ Af
    #print "Fs " + str(Fs)
    overmapped = (Ds >= 1) or (Ls >= 1) or (Fs >= 1) or (Bs >=1)
    cost, state = getCost(Bs, Ds, Ls, Fs , state) ## we need to cast to float
    #print "cost " + str(cost)
    #######################
    Ct = Rcc * ((D*Ob*Ot)/(fknl*Pknl*Pdp*Pt))
    Cre = gamma * max(Bs,Ds,Ls,Fs) / theta
    Cm = (2 * D * Wdp * (1+Nc)* Bw ) / psi  ### is constant
    #######################
    executionTime = array([Ct + Cre])    

    ######################
    ## mem bandwith exceeded
    #if BWm >= (Wdp*Bw*Pdp)*Pknl*fknl :
    #    return ((executionTime, array([4]),array([0]), cost) , state)
    
    ### accuracy error
    error = 0.0
    if error > maxError:
        return ((executionTime, array([1]),array([0]), cost) , state)
    
    ### overmapping error
    if overmapped:
        return ((array([maxvalue]), array([2]),array([1]), cost) , state)
    
    ### ok values execution time
    return ((executionTime, array([0]),array([0]), cost), state)
    
##add state saving, we use p and thread id 
def getCost(Bs, Ds, Ls, Fs , bit_stream_repo):
    return 4000 + (max(Bs,Ds,Ls,Fs) * 10000), {}
    

def calcMax():    
    import scipy.optimize as optimize
    def funcWrapper(x):
        xx = fitnessFunc(x,None)
        return xx[0][0][0]

    def funcWrapper2(x):
        xx = fitnessFunc(x,None)
        return xx[0][1][0]
    '''
    designSpace.append({"min":0.05,"max":1.0,"step":1.0,"type":"continuous", "set":"h"}) #alpha
    designSpace.append({"min":0.05,"max":1.0,"step":1.0,"type":"continuous", "set":"h"}) #beta
    designSpace.append({"min":0.05,"max":1.0,"step":1.0,"type":"continuous", "set":"h"}) #B
    designSpace.append({"min":0.05,"max":1.0,"step":1.0,"type":"continuous", "set":"h"}) #T

    designSpace.append({"min":1.0,"max":20.0,"step":1.0,"type":"discrete", "set":"h"}) #Pdp
    designSpace.append({"min":1.0,"max":20.0,"step":1.0,"type":"discrete", "set":"h"}) #Pknl
    designSpace.append({"min":1.0,"max":20.0,"step":1.0,"type":"discrete", "set":"h"}) #Ptl
    '''
        
    x0 = [0.83635750288312605, 0.37822062215746499, 0.13905958646985159, 0.58215055232421087, 2.0, 6.0, 18.0]
    x0 = [0.98874672302918809, 0.77342903511390226, 0.82587267298704492, 0.99183074578544894, 17.0, 3.0, 13.0]
    #x0 = [  0.24367197,   0.16558312,   0.82587267,   0.99183069, 4,   4,  4]
    xopt = optimize.minimize(fun = funcWrapper, method = "COBYLA", x0 =  x0, tol=1e-15, bounds = [(d["min"],d["max"]) for d in designSpace])
    print str(xopt)
    print str(fitnessFunc([  0.24367197,   0.16558312,   0.82587267,   0.99183069, 17,   4,  14],None))
    print str(fitnessFunc([  0.24367197,   0.16558312,   0.82587267,   0.99183069, 17,   3,  13],None))
    
    print str(fitnessFunc([  0.23313723,   0.17056786,   0.82590206,   0.04877084, 18.27665482,   4.36760638,  14.35909523],None))
    print str(fitnessFunc([  0.23313723,   0.17056786,   0.82590206,   0.04877084, 20,   6,  17],None))
    
        
    '''
    npts = 15
    D = len(designSpace)
    n_bins =  npts*ones(D)
    bounds = [(d["min"],d["max"]) for d in designSpace]
    print "bounds" + str(bounds)
    print "calculating mgrid"
    result = mgrid[[slice(row[0], row[1], npts*1.0j) for row, n in zip(bounds, n_bins)]]
    print "calculating mgrid done"
    Z = result.reshape(D,-1).T
    print "f(z)"
    filteredminn = []
    filteredZ=[]
    print "f(z)"
    for z in Z:
        fz = fitnessFunc(z,None)
        if fz[0][1][0]==0.0:
            filteredminn.append(fz[0][0][0])
            filteredZ.append(z)
            
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
    '''
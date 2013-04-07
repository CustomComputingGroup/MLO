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
maxVal = 20.0
worst_value = 20.0

designSpace = []

designSpace.append({"min":1.0,"max":20.0,"step":1.0,"type":"discrete", "set":"h"}) #alpha
designSpace.append({"min":1.0,"max":20.0,"step":1.0,"type":"discrete", "set":"h"}) #beta
designSpace.append({"min":4.0,"max":24.0,"step":1.0,"type":"discrete", "set":"h"}) #B
designSpace.append({"min":0.05,"max":1.0,"step":1.0,"type":"continuous", "set":"h"}) #T

designSpace.append({"min":1.0,"max":10.0,"step":1.0,"type":"discrete", "set":"h"}) #Pknl
designSpace.append({"min":1.0,"max":10.0,"step":1.0,"type":"discrete", "set":"h"}) #Ptl
designSpace.append({"min":1.0,"max":32.0,"step":1.0,"type":"discrete", "set":"h"}) #Pdp

maxvalue = worst_value
error_labels = {0:'Valid',1:'Overmap',2:'Inaccuracy', 3:'Memory'}

#[  6.,1.,5,1,2., 2.,32.04800588] around this point
def termCond(best):
    return best < 0.1
'''
n1*n2*n3:
128*128*128
256*256*256
512*512*512

coefficients range:
Pknl = (1~10)
Pdp  = (1~32)
Pt     =(1~10)

T = (1,2,3)
T<B> = 1
T<D> = (0~1)
T<L>  = (1.82~1)
T<F> =  (1.68~1)

B= (4~24)
B<w> = (12~32)
B<D> = (0.45~1)
B<L>  = (0.434~1)
B<F>  = (0.39~1)

optimal configuration:
128*128*128: alpha:1.00, beta:1.00, T=3, B=12, Pknl=1, Pdp =12, Pt =2,
Ot*Ob= 1.12
256*256*256: alpha:2.00, beta:1.00, T=3, B=12, Pknl=1, Pdp =12, Pt =2,
Ot*Ob= 1.16
512*512*512: alpha:4.00, beta:2.00, T=3, B=10, Pknl=1, Pdp =12, Pt =2,
Ot*Ob= 1.20
'''
    
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
    x = 256.0
    y = 256.0
    Nop = 700000.0
    Nc = 10.0
    S=4.0
    D=10.0**9
    
    Wdp = 32.0 # 8 bytes
    N = 1.0
    Bw = 12.0
    Wm = N * (Wdp*Bw*Pdp)
    BWm = 32000000000.0 * 8 ## GB / s
    
    ## not sure
    Bd = 0.45
    Bl = 0.39
    Bf = 0.434
    
    #epislon = sqrt()
    
    theta = 300000000.0*8.0 # 300 MB /s
    psi = 10000000000.0 *8 # 10 GB/s
    gamma = 5.0 
    Ad = 1000000.0
    Af = 1000000.0
    Al = 1000000.0
    Ab = 8.0 * 220000000.0 ## 300 MB, need to change it...
    
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
    if alpha*beta == 1.0:
        Ot = 1.0
    else:
        Ot = ((nx+(Pt-1)*2*S)/(nx)) * ((ny+(Pt-1)*2*S)/(ny))
                    
    ######################
    mdb = ((nx + (Pt-1)*2*S) / (Pdp)) * (ny + (Pt-1)*S)
    #print "2 " + str(Pknl*Pt*Pdp)
    Bs = (Pknl*Pt*Pdp*(S*(2+Nc)+1)*mdb) / (Ab/(Wdp*Bw))
    #Bs = (1.0*(S*(2+Nc)+1)*mdb) / (Ab/(Wdp*Bw))
    
    Ds = (Nop * T * Bd / Ad)
    #print "DS " + str(Ds)
    Ls = (Nop * (1.82-T) * Bl + Il)/ Al
    #print "Ls " + str(Ls)
    Fs = (Nop * (1.68-T) * Bf + If)/ Af
    #print "Fs " + str(Fs)
    overmapped = (Ds >= 1) or (Ls >= 1) or (Fs >= 1) or (Bs >=1)
    cost, state = getCost(Bs, Ds, Ls, Fs , state) ## we need to cast to float
    #print "cost " + str(cost)
    #######################
    Ct = Rcc * ((D*Ob*Ot)/(fknl*Pknl*Pdp*Pt))
    Cre = gamma * max(Bs,Ds,Ls,Fs) / theta
    Cm = (2 * D * Wdp * (1+Nc)* Bw ) / psi  ### is constant
    #######################
    executionTime = array([Ct])    

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
designSpace.append({"min":1.0,"max":20.0,"step":1.0,"type":"discrete", "set":"h"}) #alpha
designSpace.append({"min":1.0,"max":20.0,"step":1.0,"type":"discrete", "set":"h"}) #beta
designSpace.append({"min":4.0,"max":24.0,"step":1.0,"type":"discrete", "set":"h"}) #B
designSpace.append({"min":0.05,"max":1.0,"step":1.0,"type":"continuous", "set":"h"}) #T

designSpace.append({"min":1.0,"max":10.0,"step":1.0,"type":"discrete", "set":"h"}) #Pknl
designSpace.append({"min":1.0,"max":10.0,"step":1.0,"type":"discrete", "set":"h"}) #Ptl
designSpace.append({"min":1.0,"max":32.0,"step":1.0,"type":"discrete", "set":"h"}) #Pdp
    '''
    x0 = [6.0, 3.0, 5.0, 0.56117378588844269, 4.0, 3.0, 29.0]
    for kk in range(0,15):
        x0 = optimize.minimize(fun = funcWrapper, method = "COBYLA", x0 =  x0, tol=1e-15, bounds = [(d["min"],d["max"]) for d in designSpace])
        x0 = x0.x
    print str(fitnessFunc(x0,None))
    print str(fitnessFunc(x0,None))
    print str(fitnessFunc([6.0, 3.0, 5.0, 1.0, 4.0, 3.0, 32.0],None))
    print str(fitnessFunc([6.0, 3.0, 5.0, 0.56117378588844269, 4.0, 3.0, 32.0],None))
    print str(fitnessFunc([  20.,1.,15,1,4., 2.,32.04800588],None))
    print str(x0)
    print str([(d["min"],d["max"]) for d in designSpace])
    '''
    print str(fitnessFunc([  4.0,   2.0,   10.0,   0.1, 2,   2,  8],None))
    print str(fitnessFunc([  4.0,   4.0,   10.0,   0.1, 2,   2,  5],None))
    print str(fitnessFunc([  1.0,   1.0,   10.0,   0.1, 2,   2,  5],None))
    print str(fitnessFunc([  4,   3,  10.0,   0.53298803, 4.0,   3.0 ,  10.0],None))
    
        
    
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
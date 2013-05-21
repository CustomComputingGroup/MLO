import math
import operator

from numpy import *
from numpy.random import uniform, seed,rand

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
    
#always_valid= ## IMPORTANT TO ADD THIS!!!!

designSpace = []
maxError = 0.01

minVal = 0.0
maxVal = 20.0
worst_value = 20.0

designSpace = []
ratios = False
if ratios:
    designSpace.append({"min":1.0,"max":10.0,"step":1.0,"type":"discrete", "set":"h"}) #alpha
    designSpace.append({"min":1.0,"max":10.0,"step":1.0,"type":"discrete", "set":"h"}) #beta
width = False
if width:    
    designSpace.append({"min":4.0,"max":24.0,"step":1.0,"type":"discrete", "set":"h"}) #B
    designSpace.append({"min":1.0,"max":3.0,"step":1.0,"type":"discrete", "set":"h"}) #T
P = True
if P:
    designSpace.append({"min":1.0,"max":10.0,"step":1.0,"type":"discrete", "set":"h"}) #Pknl
    designSpace.append({"min":1.0,"max":10.0,"step":1.0,"type":"discrete", "set":"h"}) #Ptl
    designSpace.append({"min":1.0,"max":32.0,"step":1.0,"type":"discrete", "set":"h"}) #Pdp


maxvalue = worst_value
error_labels = {0:'Valid',1:'Overmap',2:'Inaccuracy', 3:'Memory'}

#[  6.,1.,5,1,2., 2.,32.04800588] around this point
#(4.0, 5.0, 4.0, 0.050000000000000003, 9.0, 2.0, 31.0) 0.0559685819892
def termCond(best):
    return best < 0.1
    
optimial_x = [1.0, 1.0, 4.0, 3.0, 1.0, 10.0, 6.0]
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
    fknl = 100000000.0 ## 100 MHz
    index = 0 
    if ratios:
        alpha = (particle[index])
        beta = (particle[index+1])
        index = index + 2
    else:
        alpha = 1.
        beta = 1.
        
    if width: 
        B = (particle[index])
        T = (particle[index+1])
        index = index + 2
    else:
        B = 4.
        T = 1.
        
    if P:
        Pknl = (particle[index])
        Pt = (particle[index+1])
        Pdp = (particle[index+2])
    else:
        Pknl = 1.
        Pt = 1.
        Pdp = 1.

    ######################
    Rcc = 1.0
    x = 256.0
    y = x
    Nc = 10.0
    S=4.0
    D=10.0**9
    
    Wdp = 32.0 # 8 bytes
    N = 1.0
    
    ## not sure
    Bw = 8.0 + B
    Bd = 0.45 + (0.55 * (B-4.0)/20.)
    Bl = 0.434 + (0.566 * (B-4.0)/20.)
    Bf = 0.39 + (0.61 * (B-4.0)/20.)
    
    Wm = N * (Wdp*Bw*Pdp)
    BWm = 32000000000.0 * 8 ## GB/s
        
    theta = 300000000.0*8.0 # 300 MB/s
    psi = 100000000.0*8 # 1 GB/s
    gamma = 80.0

    Ad = 2016
    Af  = 595200
    Al  = 297600
    Ab = 37.5 * 100000000*8 ## 
     
    If = Af * 0.1
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
    #print Bs
    ## should be 81 for T
    Ds = Bd * Pknl * Pknl * Pdp * (100 * ((T-1)/2)) / Ad
    Ls = Bl * Pknl * Pknl * Pdp * (16665 * (1.68 - 0.82*(T-1)/2) + Il)/ Al
    #print "Fs " + str(Fs)
    Fs = Bf * Pknl * Pknl * Pdp * (24492 * (1.68 - 0.68*(T-1)/2) + If)/ Af
    #print "####"
    #print Ds
    #print Ls
    #print Fs
    #print "####"
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
    #    return ((array([maxvalue]), array([4]),array([0]), cost) , state)
    
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
    return array([4000 + (max(Bs,Ds,Ls,Fs) * 10000)]), {}
    
if __name__ == '__main__':

    import itertools 
    maxEI = maxVal
    maxEIcord = None
    space_def = []
    counter = 0
    '''
    xx = (2.0, 2.0, 4.0, 1.0, 1.0, 2.0, 2.0)
    print fitnessFunc(xx,None)[0][0][0]
    '''
    for d in designSpace:
        space_def.append(arange(d["min"],d["max"]+1.0,d["step"]))
    print space_def
    
    for z in itertools.product(*space_def):
        counter = counter + 1
        EI = fitnessFunc(z,None)[0][0][0]
        print EI 
        print z
        if counter % 10000 == 0:
            print str(counter) + " " +  str(maxEIcord) + " " +  str(maxEI)
        
        if maxEI > EI: ## no need for None checking
            maxEI = EI
            maxEIcord = z
    ### 
    print "DONE!"
    print maxEIcord
    print maxEI
    
    
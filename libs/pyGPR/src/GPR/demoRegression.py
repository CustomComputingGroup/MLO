from gp import gp
from UTIL.solve_chol import solve_chol
import Tools.general
import Tools.min_wrapper 
import Tools.nearPD
import numpy as np
import matplotlib.pyplot as plt

from UTIL.utils import convert_to_array, hyperParameters, plotter, FITCplotter

if __name__ == '__main__':
    ## GENERATE data from a noisy GP
    n = 20 # number of labeled/training data
    D = 1 # Dimension of input data
    x = np.array([2.083970427750732,  -0.821018066101379,  -0.617870699182597,  -1.183822608860694,\
                  0.274087442277144,   0.599441729295593,   1.768897919204435,  -0.465645549031928,\
                  0.588852784375935,  -0.832982214438054,  -0.512106527960363,   0.277883144210116,\
                  -0.065870426922211,  -0.821412363806325,   0.185399443778088,  -0.858296174995998,\
                   0.370786630037059,  -1.409869162416639,-0.144668412325022,-0.553299615220374]);
    x = np.reshape(x,(n,D))
    ### GENERATE sample observations from the GP
    y = np.array([4.549203746331698,   0.371985574437271,   0.711307965514790,  -0.013212893618430,   2.255473255338191,\
                  1.009915749295733,   3.744675937965029,   0.424592771793202,   1.322833652295811,   0.278298293510020,\
                  0.267229130945574,   2.200112286723833,   1.200609983308969,   0.439971697236094,   2.628580433511255,\
                  0.503774817336353,   1.942525313820564,   0.579133950013327,   0.670874423968554,   0.377353755100965]);
    y = np.reshape(y,(n,D))

    plt.plot(x,y,'b+',markersize=12)
    plt.axis([-1.9,1.9,-0.9,3.9])
    plt.grid()
    plt.xlabel('input x')
    plt.ylabel('output y')
    plt.show()

    z = np.array([np.linspace(-1.9,1.9,101)]).T # u test points evenly distributed in the interval [-7.5, 7.5]
    ## DEFINE parameterized covariance function
    meanfunc = [ ['means.meanSum'], [ ['means.meanLinear'] , ['means.meanConst'] ] ]
    covfunc  = [ ['kernels.covMatern'] ]
    inffunc  = ['inf.infExact']
    likfunc  = ['lik.likGauss']

    ## SET (hyper)parameters
    hyp = hyperParameters()

    hyp.cov = np.array([np.log(0.25),np.log(1.0),np.log(3.0)])
    hyp.mean = np.array([0.5,1.0])
    sn = 0.1; hyp.lik = np.array([np.log(sn)])

    #_________________________________
    # STANDARD GP:
    ## PREDICTION 
    vargout = gp(hyp,inffunc,meanfunc,covfunc,likfunc,x,y,None,None,False)
    print "nlml = ",vargout[0]
    vargout = gp(hyp,inffunc,meanfunc,covfunc,likfunc,x,y,z)
    ym = vargout[0]; ys2 = vargout[1]
    m  = vargout[2]; s2 = vargout[3]
    ## Plot results
    plotter(z,ym,ys2,x,y,[-1.9, 1.9, -0.9, 3.9])
    ###########################################################
    covfunc = [ ['kernels.covSEiso'] ]
    ## SET (hyper)parameters
    hyp2 = hyperParameters()

    hyp2.cov = np.array([0.0,0.0])
    hyp2.lik = np.array([np.log(0.1)])
    #vargout = min_wrapper(hyp2,gp,'CG',inffunc,[],covfunc,likfunc,x,y,None,None,True)
    #hyp2 = vargout[0]
    hyp2.cov = np.array([-0.993396880620537,0.685943441677086])
    hyp2.lik = np.array([-1.902546786026883])
    vargout = gp(hyp2,inffunc,[],covfunc,likfunc,x,y,None,None,False)
    print "nlml2 = ",vargout[0]

    vargout = gp(hyp2,inffunc,[],covfunc,likfunc,x,y,z)
    ym = vargout[0]; ys2 = vargout[1]
    m  = vargout[2]; s2  = vargout[3]
    ## Plot results
    plotter(z,ym,ys2,x,y,[-1.9, 1.9, -0.9, 3.9])
    ###########################################################
    covfunc = [ ['kernels.covSEiso'] ]
    hyp = hyperParameters()

    hyp.cov = np.array([0.0,0.0])
    hyp.mean = np.array([0.0,0.0])
    hyp.lik = np.array([np.log(0.1)])

    #vargout = min_wrapper(hyp,gp,'BFGS',inffunc,meanfunc,covfunc,likfunc,x,y,None,None,True)
    #hyp = vargout[0]
    hyp.mean = np.array([1.1919,1.4625])
    hyp.cov = np.array([-1.1513,-0.4559])
    hyp.lik = np.array([-1.9122])
    vargout = gp(hyp,inffunc,meanfunc,covfunc,likfunc,x,y,z)
    ym = vargout[2]; ys2 = vargout[3]
    m  = vargout[2]; s2  = vargout[3]

    ## Plot results
    plotter(z,ym,ys2,x,y,[-1.9, 1.9, -0.9, 3.9])
    ###########################################################
    covfunc = [ ['kernels.covSEiso'] ]
    hyp = hyperParameters()
    
    nu = np.fix(n/2); u = np.linspace(-1.3,1.3,nu).T
    u  = np.reshape(u,(nu,1))
    covfuncF = [['kernels.covFITC'], covfunc, u]
    inffunc  = ['inf.infFITC']
    hyp.mean = np.array([1.191883198464308,1.462453407472655])
    hyp.cov = np.array([-1.513469525593061,-0.455900718308796])
    hyp.lik = np.array([-1.912235134009549])
    vargout = gp(hyp, inffunc, meanfunc, covfuncF, likfunc, x, y, z);
    ymF = vargout[0]; y2F = vargout[1] 
    mF  = vargout[2];  s2F = vargout[3] 

    FITCplotter(u,z,ymF,y2F,x,y,[-1.9, 1.9, -0.9, 3.9])

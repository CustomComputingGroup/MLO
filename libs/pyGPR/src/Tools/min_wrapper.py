import numpy as np
from copy import deepcopy
from scipy.optimize import fmin_bfgs as bfgs
from scipy.optimize import fmin_cg as cg

from GPR.UTIL.utils import convert_to_array, convert_to_class

def min_wrapper(hyp, F, Flag, *varargin):
    # Utilize scipy.optimize functions to minimize the negative log marginal liklihood.  This is REALLY inefficient!
    x = convert_to_array(hyp)

    if Flag == 'CG':
        aa = cg(nlml, x, dnlml, (F,hyp,varargin), maxiter=100, disp=False, full_output=True)
        x = aa[0]; fx = aa[1]; funcCalls = aa[2]; gradcalls = aa[3]
        if aa[4] == 1:
            print "Maximum number of iterations exceeded."
        elif aa[4] ==  2:
            print "Gradient and/or function calls not changing."
        gvals = dnlml(x,F,hyp,varargin)
        return convert_to_class(x,hyp), fx, gvals, funcCalls

    elif Flag == 'BFGS':
        # Use BFGS
        aa = bfgs(nlml, x, dnlml, (F,hyp,varargin), maxiter=100, disp=False, full_output=True)
        x = aa[0]; fvals = aa[1]; gvals = aa[2]; Bopt = aa[3]; funcCalls = aa[4]; gradcalls = aa[5]
        if aa[6] == 1:
            print "Maximum number of iterations exceeded."
        elif aa[6] ==  2:
            print "Gradient and/or function calls not changing."
        return convert_to_class(x,hyp), fvals, gvals, funcCalls

    else:
        raise Exception('Incorrect usage of optimization flag in min_wrapper')

def nlml(x,F,*varargin):
    hyp = varargin[0]
    temp = list(varargin[1:][0])
    temp[-1] = False

    f = lambda z: F(z,*temp)
    X = convert_to_class(x,hyp)
    vargout = f(X)
    return vargout[0]

def dnlml(x,F,*varargin):
    hyp = varargin[0]
    temp = list(varargin[1:][0])
    temp[-1] = True

    f = lambda z: F(z,*temp)
    X = convert_to_class(x,hyp)
    vargout = f(X)
    return convert_to_array( vargout[1] )

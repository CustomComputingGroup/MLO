#    Copyright (C) 2009  
#    Marion Neumann [marion dot neumann at iais dot fraunhofer dot de]
#    Zhao Xu [zhao dot xu at iais dot fraunhofer dot de]
#    Supervisor: Kristian Kersting [kristian dot kersting at iais dot fraunhofer dot de]
# 
#    Fraunhofer IAIS, STREAM Project, Sankt Augustin, Germany
# 
#    This file is part of pyXGPR.
# 
#    pyXGPR is free software; you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation; either version 2 of the License, or
#    (at your option) any later version.
# 
#    pyXGPR is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
# 
#    You should have received a copy of the GNU General Public License 
#    along with this program; if not, see <http://www.gnu.org/licenses/>.
#===============================================================================
'''
    Created on 31/08/2009
    
    
    This implementation (partly) follows the matlab covFunctions implementation by Rasmussen, 
    which is Copyright (c) 2005 - 2007 by Carl Edward Rasmussen and Chris Williams.
    
    
    covariance functions/kernels to be used by Gaussian process functions. 
    Beside the graph kernels based on the regularized Laplacian 
    
    regLapKernel  - returns covariance matrix of regularized Laplacian Kernel
    
    there are two different kinds of covariance functions: simple and composite:
    
    simple covariance functions:
    
    covNoise      - independent covariance function (ie white noise)
    covSEard      - squared exponential covariance function with ard
    covSEiso      - isotropic squared exponential covariance function
    
    simple covariance matices
    
    covMatrix     - non parameterized covariance (ie kernel matrix -> no (hyper)parameters)
    
    composite covariance functions (see explanation at the bottom):
    
    covSum        - sums of (parameterized) covariance functions
    covSumMat     - sums of (parameterized) covariance functions and ONE kernel matrix
    TODO: extend this to sum of more than one kernel matices
    
    Naming convention: all covariance functions start with "cov". A trailing
    "iso" means isotropic, "ard" means Automatic Relevance Determination, and
    "one" means that the distance measure is parameterized by a single parameter.
    
    The covariance functions are written according to a special convention where
    the exact behaviour depends on the number of input and output arguments
    passed to the function. If you want to add new covariance functions, you 
    should follow this convention if you want them to work with the function
    gpr. There are four different ways of calling
    the covariance functions:
    
    1) With no input arguments:
    
    p = covNAME
    
    The covariance function returns a string telling how many hyperparameters it
    expects, using the convention that "D" is the dimension of the input space.
    For example, calling "covSEard" returns the string 'D + 1'.
    
    2) With two input arguments:
    
    K = covNAME(logtheta, x) 
    
    The function computes and returns the covariance matrix where logtheta are
    the log og the hyperparameters and x is an n by D matrix of cases, where
    D is the dimension of the input space. The returned covariance matrix is of
    size n by n.
    
    3) With three input arguments and two output arguments:
    
    [v, B] = covNAME(hyp, x, z)
    
    The function computes test set covariances; v is a vector of self covariances
    for the test cases in z (of length nn) and B is a (n by nn) matrix of cross
    covariances between training cases x and test cases z.
    
    4) With three input arguments and a single output:
    
    D = covNAME(logtheta, x, z)
    
    The function computes and returns the n by n matrix of partial derivatives
    of the training set covariance matrix with respect to logtheta(z), ie with
    respect to the log of hyperparameter number z.
    
    The functions may retain a local copy of the covariance matrix for computing
    derivatives, which is cleared as the last derivative is returned.
    
    About the specification of simple and composite covariance functions to be
    used by the Gaussian process function gpr:
    
    covfunc = 'kernels.covSEard'
    
    Composite covariance functions can be specified as list. For example:
    
    covfunc = ['kernels.covSum', ['kernels.covSEard','kernels.covNoise']]
    
    
    To find out how many hyperparameters this covariance function requires, we do:
    
    Tools.general.feval(covfunc)
    
    which returns the list of strings ['D + 1', 1] 
    (ie the 'covSEard' uses D+1 and 'covNoise' a single parameter).
    
    
    @author: Marion Neumann (last update 08/01/10)
    
    Substantial updates by Daniel Marthaler Fall 2012.
'''
import Tools
import numpy as np
import math

def covFITC(covfunc, xu=None, hyp=None, x=None, z=None, der=None):
    ''' Covariance function to be used together with the FITC approximation.
    #
    # The function allows for more than one output argument and does not respect the
    # interface of a proper covariance function. In fact, it wraps a proper
    # covariance function such that it can be used together with infFITC.m.
    # Instead of outputing the full covariance, it returns cross-covariances between
    # the inputs x, z and the inducing inputs xu as needed by infFITC.m
    #
    # Copyright (c) by Ed Snelson, Carl Edward Rasmussen
    #                                               and Hannes Nickisch, 2010-12-21.
    #
    # See also COVFUNCTIONS.M, INFFITC.M.
    # NOTE: The first element of cov should be ['kernels.covFITC']
    '''
    
    if hyp == None: # report number of parameters
        A = [Tools.general.feval(covfunc)]
        return A

    try:
        assert(xu.shape[1]==x.shape[1])
    except AssertionError:
        raise Exception('Dimensionality of inducing inputs must match training inputs')

    n,D = x.shape
    if der == None:                        # compute covariance matrices for dataset x
        if z == None:
            K   = Tools.general.feval(covfunc,hyp,x,'diag')
            Kuu = Tools.general.feval(covfunc,hyp,xu)
            Ku  = Tools.general.feval(covfunc,hyp,xu,x)
        elif z == 'diag':
            K = Tools.general.feval(covfunc,hyp,x,z)
            return K
        else:
            K = Tools.general.feval(covfunc,hyp,xu,z)
            return K
    else:                                  # compute derivative matrices
        if z == None:
            K   = Tools.general.feval(covfunc,hyp,x,'diag')
            Kuu = Tools.general.feval(covfunc,hyp,xu)
            Ku  = Tools.general.feval(covfunc,hyp,xu,x)
        elif z == 'diag':
            K = Tools.general.feval(covfunc,hyp,x,z)
            return K
        else:
            K = Tools.general.feval(covfunc,hyp,xu,z)
            return K
    return K, Kuu, Ku

def covMask(covfunc, hyp=None, x=None, z=None, der=None):
    '''covMask - compose a covariance function as another covariance
        function (covfunc), but with only a subset of dimensions of x. hyp here contains
        the hyperparameters of covfunc. This function doesn't actually compute very much on its own, it
        merely does some bookkeeping, and calls other covariance functions to do the
        actual work. '''

    mask = covfunc[0] # The indicies to be masked (should be a list of integers)
    cov  = covfunc[1]                                 # covariance function to be masked

    if hyp == None: # report number of parameters
        A = [Tools.general.feval(covfunc[1])]
        return A

    n, D = x.shape

    assert(len(mask) < D)
    assert(max(mask) < D)
    assert(min(mask) >= 0)

    if der == None:      # compute covariance matix for dataset x
        if z == None:
            A = Tools.general.feval(cov, hyp, x[:,mask])
        else:                                          # compute covariance between data sets x and z
            if z == 'diag':
                A = Tools.general.feval(cov,hyp,x[:,mask],z)
            else:
                A = Tools.general.feval(cov,hyp,x[:,mask],z[:,mask])       # cross covariances
    else:                # compute derivatives
        if z == None:
            A = Tools.general.feval(cov, hyp, x[:,mask],z,der)
        elif z == 'diag':
            A = Tools.general.feval(cov,hyp,x[:,mask],z,der)
        else:
            A = Tools.general.feval(cov,hyp,x[:,mask],z[:,mask],der)
    
    return A

def covPoly(hyp=None, x=None, z=None,der=None):
    '''Polynomial covariance function 
    The covariance function is parameterized as:
     k(x^p,x^q) = sf2 * ( c +  (x^p)'*(x^q) ) ** d

    The hyperparameters of the function are:
    hyp = [ log(c)
                log(sqrt(sf2)) 
                log(d) ]

    '''
    if hyp == None:                  # report number of parameters
        return [3]

    c   = np.exp(hyp[0])          # inhomogeneous offset
    sf2 = np.exp(2.*hyp[1])        # signal variance
    ord = np.exp(hyp[2])          # ord of polynomical

    if np.abs(ord-np.round(ord)) < 1e-8:  # remove numerical error from format of parameter
        ord = int(round(ord))

    assert(ord == max(1.,np.fix(ord))) # only nonzero integers for d              
    ord = int(ord)

    n, D = x.shape

    if z == 'diag':
        A = (x*x).sum(axis=1)
    elif z==None:
        A = np.dot(x,x.T)
    else:                                  # compute covariance between data sets x and z
        A = np.dot(x,z.T)           # cross covariances
        
    if der == None:                        # compute covariance matix for dataset x
        A = sf2 * (c + A)**ord
    else:
        if der == 0:  # compute derivative matrix wrt 1st parameter
            A = c * ord * sf2 * (c+A)**(ord-1)
        elif der == 1:  # compute derivative matrix wrt 2nd parameter
            A = 2. * sf2 * (c + A)**ord
        elif der == 2:  # Wants to compute derivative wrt order
            A = np.zeros_like(A)
        else:
            raise Exception("Wrong derivative entry in covPoly")

    return A

def covPPiso(hyp=None, x=None, z=None, der=None):
    '''Piecewise polynomial covariance function with compact support
    The covariance function is:
    
     k(x^p,x^q) = s2f * (1-r)_+.^j * f(r,j)
    
    where r is the distance sqrt((x^p-x^q)'*inv(P)*(x^p-x^q)), P is ell^2 times
    the unit matrix and sf2 is the signal variance. 
    The hyperparameters are:
    
     hyp = [ log(ell)
             log(sqrt(sf2)) 
             log(v) ]
    '''
    def ppmax(A,B):
        return np.maximum(A,B*np.ones_like(A))

    def func(v,r,j):
        if v == 0:
            return 1
        elif v == 1:
            return ( 1. + (j+1) * r )
        elif v == 2:
            return ( 1. + (j+2)*r + (j*j + 4.*j+ 3)/3.*r*r )
        elif v == 3:
            return ( 1. + (j+3)*r + (6.*j*j+36.*j+45.)/15.*r*r + (j*j*j+9.*j*j+23.*j+15.)/15.*r*r*r )
        else:
             raise Exception (["Wrong degree in covPPiso.  Should be 0,1,2 or 3, is " + str(v)])

    def dfunc(v,r,j):
        if v == 0:
            return 0
        elif v == 1:
            return ( j+1 )
        elif v == 2:
            return ( (j+2) + 2.*(j*j+ 4.*j+ 3.)/3.*r )
        elif v == 3:
            return ( (j+3) + 2.*(6.*j*j+36.*j+45.)/15.*r + (j*j*j+9.*j*j+23.*j+15.)/5.*r*r )
        else:
            raise Exception (["Wrong degree in covPPiso.  Should be 0,1,2 or 3, is " + str(v)])

    def pp(r,j,v,func):
        return func(v,r,j)*(ppmax(1-r,0)**(j+v))

    def dpp(r,j,v,func,dfunc):
        return ppmax(1-r,0)**(j+v-1) * r * ( (j+v)*func(v,r,j) - ppmax(1-r,0) * dfunc(v,r,j) )

    if hyp == None:                 # report number of parameters
        return [3]

    ell = np.exp(hyp[0])            # characteristic length scale
    sf2 = np.exp(2.*hyp[1])       # signal variance
    v   = np.exp(hyp[2])         # degree (v = 0,1,2 or 3 only)

    if np.abs(v-np.round(v)) < 1e-8:     # remove numerical error from format of parameter
        v = int(round(v))

    assert(int(v) in range(4))           # Only allowed degrees: 0,1,2 or 3
    v = int(v)
    
    n, D = x.shape

    j = math.floor(0.5*D) + v + 1

    if z == 'diag':
        A = np.zeros((n,1))
    elif z == None:
        A = np.sqrt( sq_dist(x/ell) )
    else:                                          # compute covariance between data sets x and z
        A = np.sqrt( sq_dist(x/ell,z/ell) )         # cross covariances


    if der == None:                        # compute covariance matix for dataset x
        A = sf2 * pp(A,j,v,func)
    else:
        if der == 0:  # compute derivative matrix wrt 1st parameter
            A = sf2 * dpp(A,j,v,func,dfunc)

        elif der == 1:  # compute derivative matrix wrt 2nd parameter
            A = 2. * sf2 * pp(A,j,v,func)

        elif der == 2:  # Wants to compute derivative wrt order
            A = np.zeros_like(A)
        else:
            raise Exception("Wrong derivative entry in covPPiso")

    return A

def covConst(hyp=None, x=None, z=None, der=None):
    '''Covariance function for a constant function.
    The covariance function is parameterized as:
    k(x^p,x^q) = sf2 

    The scalar hyperparameter is:
    hyp = [ log(sqrt(sf2)) ]
    '''

    if hyp == None:                 # report number of parameters
        return [1]

    sf2 = np.exp(2.*hyp[0])       # s2

    n,m = x.shape
    if z == 'diag':
        A = sf2*np.ones((n,1))
    elif z == None:
        A = sf2 * np.ones((n,n))
    else:
        A = sf2*np.ones((n,z.shape[0]))

    if der == 0:  # compute derivative matrix wrt sf2
        A = 2. * A
    elif der:
        raise Exception("Wrong derivative entry in covConst")
    return A

def covScale(covfunc, hyp=None, x=None, z=None, der=None):
    '''Compose a covariance function as a scaled version of another one
    k(x^p,x^q) = sf2 * k0(x^p,x^q)
    
    The hyperparameter is :
    
    hyp = [ log(sf2) ]

    This function doesn't actually compute very much on its own. it merely does
    some bookkeeping, and calls another covariance function to do the actual work.
    '''

    if hyp == None:    # report number of parameters
        A = [1]
        A.append( Tools.general.feval(covfunc[0]) )
        return A

    sf2 = np.exp(2.*hyp[0])    # scale parameter
    n,D = x.shape

    if der == None:                           # compute covariance matrix
        A = sf2 * Tools.general.feval(covfunc[0], hyp[1:], x,z)  # accumulate covariances

    else:
        if der == 0:                # compute derivative w.r.t. sf2
            A = 2. * sf2 * Tools.general.feval(covfunc[0], hyp[1:], x, z)
        else:                # compute derivative w.r.t. scaled covFunction
            A = sf2 * Tools.general.feval(covfunc[0], hyp[1:], x, z, der-1)

    return A

def covLIN(hyp=None, x=None, z=None, der=None):
    '''Linear Covariance function.
    The covariance function is parameterized as:
    k(x^p,x^q) = sf2 + x^p'*x^q

    There are no hyperparameters:

    hyp = []
 
    Note that there is no bias or scale term; use covConst and covScale to add these

    '''

    if hyp == None:                       # report number of parameters
        return [0]
    n,m = x.shape

    if z == 'diag':
        A = (x*x).sum(axis=1)
    elif z == None:
        A = np.dot(x,x.T) + np.eye(n)*1e-16 #required for numerical accuracy
    else:                                         # compute covariance between data sets x and z
        A = np.dot(x,z.T)                      # cross covariances

    if der:
        raise Exception("No derivative available in covLIN")

    return A

def covLINard(hyp=None, x=None, z=None, der=None):
    '''Linear covariance function with Automatic Relevance Detemination
    (ARD) distance measure. The covariance function is parameterized as:
    k(x^p,x^q) = x^p' * inv(P) * x^q
    
    where the P matrix is diagonal with ARD parameters ell_1^2,...,ell_D^2, where
    D is the dimension of the input space and sf2 is the signal variance. The
    hyperparameters are:
    
    hyp = [ log(ell_1)
                 log(ell_2)
                   .
                   .
                 log(ell_D) ]

    Note that there is no bias term; use covConst to add a bias.
    '''

    if hyp == None:                  # report number of parameters
        return ['D + 0']                    # USAGE: integer OR D_+_int (spaces are SIGNIFICANT)

    n, D = x.shape
    ell = np.exp(hyp) # characteristic length scales
    x_ = np.dot(x,np.diag(1./ell))

    if z == 'diag':
        A = (x_*x_).sum(axis=1)
    elif z == None:
        A = np.dot(x_,x_.T)
    else:                                       # compute covariance between data sets x and z
        z = np.dot(z,np.diag(1./ell))
        A = np.dot(x_,z.T)                   # cross covariances

    if der and der < D:
        if z == 'diag':
            A = -2.*x_[:,der]*x_[:,der]
        elif z == None:
            A = -2.*np.dot(x_[:,der],x_[:,der].T)
        else:
            A = -2.*np.dot(x_[:,der],z[:,der].T)                   # cross covariances
    elif der:
        raise Exception("Wrong derivative index in covLINard")
    
    return A

def covMatern(hyp=None, x=None, z=None, der=None):
    ''' Matern covariance function with nu = d/2 and isotropic distance measure. For d=1 
        the function is also known as the exponential covariance function or the 
        Ornstein-Uhlenbeck covariance in 1d. The covariance function is: 
            k(x^p,x^q) = s2f * f( sqrt(d)*r ) * exp(-sqrt(d)*r) 
        with f(t)=1 for d=1, f(t)=1+t for d=3 and f(t)=1+t+(t*t)/3 for d=5. 
        Here, r is the distance sqrt( (x^p-x^q)'*inv(P)*(x^p-x^q)), 
        where P is ell times the unit matrix and sf2 is the signal variance. 
        The hyperparameters of the function are: 
    hyp = [ log(ell) 
                 log(sqrt(sf2)) 
                 log(d) ]
    '''
    def func(d,t):
        if d == 1:
            return 1
        elif d == 3:
            return 1 + t
        elif d == 5:
            return 1 + t*(1+t/3.)
        else:
            raise Exception("Wrong value for d in covMatern")

    def dfunc(d,t):
        if d == 1:
            return 1
        elif d == 3:
            return t
        elif d == 5:
            return t*(1+t/3.)
        else:
            raise Exception("Wrong value for d in covMatern")

    def mfunc(d,t):
        return func(d,t)*np.exp(-1.*t)

    def dmfunc(d,t):
        return dfunc(d,t)*t*np.exp(-1.*t)

    if hyp == None:                 # report number of parameters
        return [3]

    ell = np.exp(hyp[0])            # characteristic length scale
    sf2 = np.exp(2.*hyp[1])       # signal variance
    d   = np.exp(hyp[2])         # 2 times nu
    
    if np.abs(d-np.round(d)) < 1e-8:     # remove numerical error from format of parameter
        d = int(round(d))

    try:
        assert(int(d) in [1,3,5])            # Check for valid values of d
    except AssertionError:
        d = 3

    d = int(d)

    if z == 'diag':
        A = np.zeros((x.shape[0],1))
    elif z == None:
        x = np.sqrt(d)*x/ell
        A = np.sqrt(sq_dist(x))
    else:
        x = np.sqrt(d)*x/ell
        z = np.sqrt(d)*z/ell
        A = np.sqrt(sq_dist(x,z))

    if der == None:                        # compute covariance matix for dataset x
        A = sf2 * mfunc(d,A)
    else:
        if der == 0:  # compute derivative matrix wrt 1st parameter
            A = sf2 * dmfunc(d,A)

        elif der == 1:  # compute derivative matrix wrt 2nd parameter
            A = 2 * sf2 * mfunc(d,A)
        elif der == 2:  # Wants to compute derivative wrt nu
            A = np.zeros_like(A) # Do nothing
        else:
            raise Exception("Wrong derivative value in covMatern")

    return A

def covSEiso(hyp=None, x=None, z=None, der=None):
    '''Squared Exponential covariance function with isotropic distance measure.
    The covariance function is parameterized as:
    k(x^p,x^q) = sf2 * exp(-(x^p - x^q)'*inv(P)*(x^p - x^q)/2)
    where the P matrix is ell^2 times the unit matrix and
    sf2 is the signal variance  

    The hyperparameters of the function are:
    hyp = [ log(ell)
                log(sqrt(sf2)) ]
    a column vector  
    each row of x/z is a data point'''


    if hyp == None:                 # report number of parameters
        return [2]

    ell = np.exp(hyp[0])          # characteristic length scale
    sf2 = np.exp(2.*hyp[1])       # signal variance
    n,D = x.shape

    if z == 'diag':
        A = np.zeros((n,1))
    elif z == None:
        A = sq_dist(x/ell)
    else:                                          # compute covariance between data sets x and z
        A = sq_dist(x/ell,z/ell)         # self covariances (needed for GPR)

    if der == None:                        # compute covariance matix for dataset x
        A = sf2 * np.exp(-0.5*A)
    else:
        if der == 0:  # compute derivative matrix wrt 1st parameter
            A = sf2 * np.exp(-0.5*A) * A

        elif der == 1:  # compute derivative matrix wrt 2nd parameter
            A = 2. * sf2 * np.exp(-0.5*A)
        else:
            raise Exception("Calling for a derivative in covSEiso that does not exist")
    
    return A

def covSEard(hyp=None, x=None, z=None, der=None):

    '''Squared Exponential covariance function with Automatic Relevance Detemination
    (ARD) distance measure. The covariance function is parameterized as:
    k(x^p,x^q) = sf2 * exp(-(x^p - x^q)'*inv(P)*(x^p - x^q)/2)
    
    where the P matrix is diagonal with ARD parameters ell_1^2,...,ell_D^2, where
    D is the dimension of the input space and sf2 is the signal variance. The
    hyperparameters are:
    
    hyp = [ log(ell_1)
            log(ell_2)
               .
            log(ell_D)
            log(sqrt(sf2)) ]'''
    
    if hyp == None:                # report number of parameters
        return ['D + 1']                  # USAGE: integer OR D_+_int (spaces are SIGNIFICANT)
    
    [n, D] = x.shape
    ell = 1/np.exp(hyp[0:D])       # characteristic length scale

    sf2 = np.exp(2.*hyp[D])      # signal variance

    if z == 'diag':
        A = np.zeros((n,1))
    elif z == None:
        A = sq_dist(np.dot(np.diag(ell),x.T).T)
    else:                                           # compute covariance between data sets x and z
        A = sq_dist(np.dot(np.diag(ell),x.T).T,np.dot(np.diag(ell),z.T).T)       # cross covariances
 
    A = sf2*np.exp(-0.5*A)
    if der:
        if der < D:      # compute derivative matrix wrt length scale parameters
            if z == 'diag':
                A = A*0
            elif z == None:
                A = A * sq_dist(x[:,der].T/ell[der])
            else:
                A = A * sq_dist(x[:,der].T/ell[der],z[:,der].T/ell[der])
            # NOTE: ell = 1/exp(hyp) AND sq_dist is written for the transposed input!!!!
        elif der==D:      # compute derivative matrix wrt magnitude parameter
            A = 2.*A
        else:
            raise Exception("Wrong derivative index in covSEard")
                
    return A

def covSEisoU(hyp=None, x=None, z=None, der=None):
    '''Squared Exponential covariance function with isotropic distance measure with
    unit magnitude. The covariance function is parameterized as:
    k(x^p,x^q) = exp( -(x^p - x^q)' * inv(P) * (x^p - x^q) / 2 )
    where the P matrix is ell^2 times the unit matrix 

    The hyperparameters of the function are:
    hyp = [ log(ell) ]
    '''

    if hyp == None:                 # report number of parameters
        return [1]

    ell = np.exp(hyp[0])            # characteristic length scale
    n,D = x.shape

    if z == 'diag':
        A = np.zeros((n,1))
    elif z == None:
        A = sq_dist(x/ell)
    else:                                      # compute covariance between data sets x and z
        A = sq_dist(x/ell,z/ell)         # self covariances (needed for GPR)

    if der == None:                        # compute covariance matix for dataset x
        A = np.exp(-0.5*A)
    else:
        if der == 0:  # compute derivative matrix wrt 1st parameter
            A = np.exp(-0.5*A) * A
        else:
            raise Exception("Wrong derivative index in covSEisoU")

    return A

def covPeriodic(hyp=None, x=None, z=None, der=None):
    '''Stationary covariance function for a smooth periodic function,'
    with period p:
    k(x^p,x^q) = sf2 * exp( -2*sin^2( pi*||x^p - x^q)||/p )/ell**2 )

    The hyperparameters of the function are:
    hyp = [ log(ell)
                log(p)
                log(sqrt(sf2)) ]
    '''

    if hyp == None:                 # report number of parameters
        return [3]

    ell = np.exp(hyp[0])            # characteristic length scale
    p   = np.exp(hyp[1])            # period
    sf2 = np.exp(2.*hyp[2])      # signal variance

    n,D = x.shape

    if z == 'diag':
        A = np.zeros((n,1))
    elif z == None:
        A = np.sqrt(sq_dist(x))
    else:
        A = np.sqrt(sq_dist(x,z))

    A = np.pi*A/p

    if der == None:                        # compute covariance matix for dataset x
        A = np.sin(A)/ell
        A = A * A
        A = sf2 *np.exp(-2.*A)
    else:
        if der == 0:  # compute derivative matrix wrt 1st parameter
            A = np.sin(A)/ell
            A = A * A
            A = 4. *sf2 *np.exp(-2.*A) * A

        elif der == 1:  # compute derivative matrix wrt 2nd parameter
            R = np.sin(A)/ell
            A = 4 * sf2/ell * np.exp(-2.*R*R)*R*np.cos(A)*A

        elif der == 2:  # compute derivative matrix wrt 3rd parameter
            A = np.sin(A)/ell
            A = A * A
            A = 2. * sf2 * np.exp(-2.*A)
        else:
            raise Exception("Wrong derivative index in covPeriodic")
                
    return A

def covRQiso(hyp=None, x=None, z=None, der=None):
    '''Rational Quadratic covariance function with isotropic distance measure.
    The covariance function is parameterized as:
    k(x^p,x^q) = sf2 * [1 + (x^p - x^q)'*inv(P)*(x^p - x^q)/(2*alpha)]^(-alpha)
    where the P matrix is ell^2 times the unit matrix,
    sf2 is the signal variance, and alpha is the shape parameter for the RQ
    covariance.  

    The hyperparameters of the function are:
    hyp = [ log(ell)
                 log(sqrt(sf2)) 
                 log(alpha) ]
    each row of x/z is a data point'''

    if hyp == None:                   # report number of parameters
        return [3]

    ell   = np.exp(hyp[0])            # characteristic length scale
    sf2   = np.exp(2.*hyp[1])         # signal variance
    alpha = np.exp(hyp[2])            #

    n,D = x.shape

    if z == 'diag':
        D2 = np.zeros((n,1))
    elif z == None:
        D2 = sq_dist(x/ell)
    else:
        D2 = sq_dist(x/ell,z/ell)

    if der == None:                        # compute covariance matix for dataset x
        A = sf2 * ( ( 1.0 + 0.5*D2/alpha )**(-alpha) )
    else:
        if der == 0:  # compute derivative matrix wrt 1st parameter
            A = sf2 * ( 1.0 + 0.5*D2/alpha )**(-alpha-1) * D2

        elif der == 1:  # compute derivative matrix wrt 2nd parameter
            A = 2.* sf2 * ( ( 1.0 + 0.5*D2/alpha )**(-alpha) )

        elif der == 2:  # compute derivative matrix wrt 3rd parameter
            K = ( 1.0 + 0.5*D2/alpha )
            A = sf2 * K**(-alpha) * (0.5*D2/K - alpha*np.log(K) )
        else:
            raise Exception("Wrong derivative index in covRQiso")
    
    return A

def covRQard(hyp=None, x=None, z=None, der=None):
    '''Rational Quadratic covariance function with Automatic Relevance Detemination
    (ARD) distance measure. The covariance function is parameterized as:
    k(x^p,x^q) = sf2 * [1 + (x^p - x^q)'*inv(P)*(x^p - x^q)/(2*alpha)]^(-alpha)
    
    where the P matrix is diagonal with ARD parameters ell_1^2,...,ell_D^2, where
    D is the dimension of the input space, sf2 is the signal variance and alpha is 
    the shape parameter for the RQ covariance. The hyperparameters are:
    
    hyp = [ log(ell_1)
                  log(ell_2)
                   .
                  log(ell_D)
                  log(sqrt(sf2)) 
                  log(alpha)]'''
    
    if hyp == None:                # report number of parameters
        return ['D + 2']                  # USAGE: integer OR D_+_int (spaces are SIGNIFICANT)
    
    [n, D] = x.shape
    ell = 1/np.exp(hyp[0:D])       # characteristic length scale
    sf2 = np.exp(2.*hyp[D])        # signal variance
    alpha = np.exp(hyp[D+1])

    if z == 'diag':
        D2 = np.zeros((n,1))
    elif z == None:
        D2 = sq_dist(np.dot(np.diag(ell),x.T).T)
    else:
        D2 = sq_dist(np.dot(np.diag(ell),x.T).T,np.dot(np.diag(ell),z.T).T)


    if der == None:                          # compute covariance matix for dataset x
        A = sf2 * ( ( 1.0 + 0.5*D2/alpha )**(-alpha) )
    else:
        if der < D:      # compute derivative matrix wrt length scale parameters
            if z == 'diag':
                A = D2*0
            elif z == None:
                A = sf2 * ( 1.0 + 0.5*D2/alpha )**(-alpha-1) * sq_dist(x[:,der].T/ell[der])
            else:
                A = sf2 * ( 1.0 + 0.5*D2/alpha )**(-alpha-1) * sq_dist(x[:,der].T/ell[der],z[:,der].T/ell[der])
        elif der==D:      # compute derivative matrix wrt magnitude parameter
            A = 2. * sf2 * ( ( 1.0 + 0.5*D2/alpha )**(-alpha) )

        elif der==(D+1):      # compute derivative matrix wrt magnitude parameter
            K = ( 1.0 + 0.5*D2/alpha )
            A = sf2 * K**(-alpha) * ( 0.5*D2/K - alpha*np.log(K) )
        else:
            raise Exception("Wrong derivative index in covRQard")
    
    return A

def covNoise(hyp=None, x=None, z=None, der=None):
    '''Independent covariance function, ie "white noise", with specified variance.
    The covariance function is specified as:
    k(x^p,x^q) = s2 * \delta(p,q)

    where s2 is the noise variance and \delta(p,q) is a Kronecker delta function
    which is 1 iff p=q and zero otherwise. The hyperparameter is

    hyp = [ log(sqrt(s2)) ]

    NOTE: Calling this function with z = x does NOT produce the correct result!
    '''
    tol = 1.e-9       # Tolerance for declaring two vectors "equal"
    if hyp == None:                             # report number of parameters
        return [1]
    
    s2 = np.exp(2.*hyp[0])                   # noise variance
    n,D = x.shape

    if z == 'diag':
        A = np.ones((n,1))
    elif z == None:
        A = np.eye(n)
    else:                                        # compute covariance between data sets x and z
        M = sq_dist(x,z)
        A = np.zeros_like(M,dtype=np.float)
        A[M < tol] = 1.

    if der == None:
        A = s2*A
    else: # compute derivative matrix
        if der == 0:
            A = 2.*s2*A
        else:
            raise Exception("Wrong derivative index in covNoise")

    return A
    
def covMatrix(R_=None, Rstar_=None):
    '''This function allows for a non-paramtreised covariance.
    input:  R_:        training set covariance matrix (train by train)
            Rstar_:    cross covariances train by test
                      last row: self covariances (diagonal of test by test)
    -> no hyperparameters have to be optimised. '''
    
    if R_ == None:                                 # report number of parameters
        return 0
    
    A = None
    if Rstar_==None:                               # trainings set covariances
        A = R_
    elif isinstance(Rstar_, int):  # derivative matrix (not needed here!!)                             
        raise Exception("Error: NO optimization to be made in covfunc (CV is done seperatly)")
    else:                                          # test set covariances  
        A = np.array([Rstar_[-1,]]).transpose() # self covariances for the test cases (last row) 
        B = Rstar_[0:Rstar_.shape[0]-1,:]          # cross covariances for trainings and test cases
        A = [A,B]
    return A    

def covSum(covfunc, hyp=None, x=None, z=None, der=None):
    '''covSum - compose a covariance function as the sum of other covariance
        functions. This function doesn't actually compute very much on its own, it
        merely does some bookkeeping, and calls other covariance functions to do the
        actual work. '''

    def DetermineNumberOfParameters(v,no_param):
        if isinstance(no_param, int):
            v.append(no_param)
        elif isinstance(no_param,str): # no_param is a string
            pram_str = no_param.split(' ')
            if pram_str[0]=='D': temp = int(D)
            if pram_str[1]=='+': temp += int(pram_str[2])
            elif pram_str[1]=='-': temp -= int(pram_str[2])
            else:
                raise Exception(["Error: number of parameters of "+covfunc[i] +" unknown!"])
            v.append(temp)
        elif isinstance(no_param, list):
            # The number of hyperparameters for this piece of covfunc is the sum
            # of all of them in this composition
            temp = [0]
            for jj in xrange(len(no_param)):
                DetermineNumberOfParameters(temp,no_param[jj])
            v.append(sum(temp))
        else:
            # This is an error, we should never be here
            raise Exception("Error in return of number of parameters")
        return v
    
    if hyp == None: # report number of parameters
        A = [Tools.general.feval(covfunc[0])]
        for i in range(1,len(covfunc)):
            A.append(Tools.general.feval(covfunc[i]))
        return A

    [n, D] = x.shape
    
    # SET vector v (v indicates how many parameters each covfunc has
    # (NOTE : v[i]=number of parameters + 1 -> this is because of the indexing of python!))
    
    v = [0] # needed for technical reasons
    for ii in range(1,len(covfunc)+1):
        no_param = Tools.general.feval(covfunc[ii-1])
        DetermineNumberOfParameters(v,no_param)

    AT = Tools.general.feval(covfunc[0], hyp[:v[1]], x, z)
    A = np.zeros_like(AT)         # Allocate covariance Matrix

    if der == None: # compute covariance matrix
        for ii in range(1,len(covfunc)+1): # iteration over summand functions
            f = covfunc[ii-1]
            s = sum(v[0:ii])
            A = A + Tools.general.feval(f, hyp[s:(s+v[ii])], x, z) # accumulate covariances
    else: # compute derivative matrices
        tmp = 0
        for ii in range(1,len(covfunc)+1):
            tmp += v[ii]
            if der<tmp:
                j = der-(tmp-v[ii]); break # j: which parameter in that covariance
        f = covfunc[ii-1] # i: which covariance function
        # compute derivative
        A = Tools.general.feval(f, hyp[sum(v[0:ii]):sum(v[0:ii])+v[ii]], x, z, int(j))

    return A

def covProd(covfunc, hyp=None, x=None, z=None, der=None):
    '''covProd - compose a covariance function as the product of other covariance
    functions. This function doesn't actually compute very much on its own, it
    merely does some bookkeeping, and calls other covariance functions to do the
    actual work. '''

    def DetermineNumberOfParameters(v,no_param):
        if isinstance(no_param, int):
            v.append(no_param)
        elif isinstance(no_param,str): # no_param is a string
            pram_str = no_param.split(' ')
            if pram_str[0]=='D': temp = int(D)
            if pram_str[1]=='+': temp += int(pram_str[2])
            elif pram_str[1]=='-': temp -= int(pram_str[2])
            else:
                raise Exception(["Error: number of parameters of "+covfunc[i] +" unknown!"])
            v.append(temp)
        elif isinstance(no_param, list):
            # The number of hyperparameters for this piece of covfunc is the sum
            # of all of them in this composition
            temp = [0]
            for jj in xrange(len(no_param)):
                DetermineNumberOfParameters(temp,no_param[jj])
            v.append(sum(temp))
        else:
            # This is an error, we should never be here
            raise Exception("Error in return of number of parameters")
        return v

    if hyp == None:    # report number of parameters
        A = [Tools.general.feval(covfunc[0])]
        for ii in range(1,len(covfunc)):
            A.append(Tools.general.feval(covfunc[ii]))
        return A

    [n, D] = x.shape
        
    # SET vector v (v indicates how many parameters each covfunc has 
    # (NOTE : v[i]=number of parameters + 1 -> this is because of the indexing of python!))
    v = [0]
    for ii in range(1,len(covfunc)+1):
        no_param = Tools.general.feval(covfunc[ii-1])
        DetermineNumberOfParameters(v,no_param)  

    AT = Tools.general.feval(covfunc[0], hyp[:v[1]], x, z)
    A = np.ones_like(AT)         # Allocate covariance Matrix

    if der == None:                          # compute covariance matrix
        for ii in range(1,len(covfunc)+1):  # iteration over multiplicand functions
            f = covfunc[ii-1]
            s = sum(v[0:ii])
            A *= Tools.general.feval(f, hyp[s:(s+v[ii])], x, z)  # accumulate covariances
    else:
        tmp = 0                                                                                  
        flag = True
        for ii in range(1,len(covfunc)+1): 
            tmp += v[ii]
            if der<tmp and flag:
                flag = False                
                jj = der-(tmp-v[ii])                    # j: which parameter in that covariance
                f = covfunc[ii-1]                    # i: which covariance function
                # compute derivative
                s = sum(v[0:ii])
                A *= Tools.general.feval(f, hyp[s:(s+v[ii])], x, z, int(jj))
            else:                
                f = covfunc[ii-1]                    # ii: which covariance function
                s = sum(v[0:ii])
                A *= Tools.general.feval(f, hyp[s:(s+v[ii])], x, z)            
    return A

def regLapKernel(R, beta, s2):
    '''Covariance/kernel matrix calculated via regluarized Laplacian.'''

    v = R.sum(axis=0)     # sum of each column
    D = np.diag(v)   
    
    K_R = np.linalg.inv(beta*(np.eye(R.shape[0])/s2+D-R)) # cov matrix for ALL the data
    
    ## NORMALISATION = scale to [0,1]
    ma = K_R.max(); mi = K_R.min()
    K_R = (K_R-mi)/(ma-mi)
    
    return K_R

def sq_dist(a, b=None):
    '''Compute a matrix of all pairwise squared distances
    between two sets of vectors, stored in the row of the two matrices:
    a (of size n by D) and b (of size m by D). '''

    n = a.shape[0]
    D = a.shape[1]
    m = n    

    if b == None:
        b = a.transpose()
    else:
        m = b.shape[0]
        b = b.transpose()

    C = np.zeros((n,m))

    for d in range(0,D):
        tt = a[:,d]
        tt = tt.reshape(n,1)
        tem = np.kron(np.ones((1,m)), tt)
        tem = tem - np.kron(np.ones((n,1)), b[d,:])
        C = C + tem * tem  
    return C

#===============================================================================
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
'''Created on 31/08/2009

    This implementation follows the matlab GP implementation by Rasmussen, 
    which is Copyright (c) 2005 - 2007 by Carl Edward Rasmussen and Chris Williams. 
    It invokes minimize.py which is implemented in python by Roland Memisevic 2008, 
    following minimize.m which is copyright (C) 1999 - 2006, Carl Edward Rasmussen.

    This module provides functions for 
    - training the hyperparameters of a Standard GP (parameterised covariance function)                                 
    - prediction in the Standard GP framework
    
    (note: these also work for the XGP framework but mixture weights have to be adjusted separately!)
    
    INPUT:
    gp             a dictionary containing
        meantheta     column vector of hyperparameters of the mean function  
        meanfunc      is the mean function (string)
        meantheta     column vector of log hyperparameters (of the covariance function)  
        covfunc       is the covariance function (string)
    X              is the n by D matrix of training inputs
    y              is the column vector of targets (size n)
                   NOTE: y must have zero mean!!
    Xstar          is the nn by D matrix of test inputs
    
    R, w, Rstar    are inputs for XGP regression
    
    OUTPUT:
    gp             the structure containing the learnt hyperparameters for the given mean and cov funcs, resp.
    val            the final value of the objective function
    iters          the number of iterations used by the method
    
    
    TODOS:
    ------
    - provide a large number of random starting locations to minimize,
      and pick the best final answer.
    - box-bound the search space, so that minimize doesn't head off to
      crazily large or small values. Even better, do MAP rather than ML.
    
@author: Marion Neumann (last update 08/01/10)
'''
import numpy as np
from Tools.general import feval
from Tools.nearPD import nearPD
from scipy.optimize import fmin_bfgs as bfgs
from scipy.optimize import fmin_cg as cg
import minimize, scg

import matplotlib.pyplot as plt

sn2 = 1e-6 #Hardcoded noise for Gaussian inference

def gp_train(gp, X, y, R=None, w=None, Flag = 'CG'):
    ''' gp_train() returns the learnt hyperparameters.
    Following chapter 5.4.1 in Rasmussen and Williams: GPs for ML (2006).
    The original version (MATLAB implementation) of used optimizer minimize.m 
    is copyright (C) 1999 - 2006, Carl Edward Rasmussen.
    The used python versions are in scipy.optimize
    
    Input R and w is needed for XGP regression! '''

    # Build the parameter list that we will optimize
    theta = np.concatenate((gp['meantheta'],gp['covtheta']))
    if Flag == 'CG':
        aa = cg(nlml, theta, dnlml, [gp,X,y,R,w], maxiter=100, disp=False, full_output=True)
        theta = aa[0]; fvals = aa[1]; funcCalls = aa[2]; gradcalls = aa[3]
        gvals = dnlml(theta, gp, X, y, R, w)
        if aa[4] == 1:
            print "Maximum number of iterations exceeded." 
        elif aa[4] ==  2:
            print "Gradient and/or function calls not changing."
        mt = len(gp['meantheta'])
        gp['meantheta'] = theta[:mt]
        gp['covtheta']  = theta[mt:]
        return gp, fvals, gvals, funcCalls
    elif Flag == 'BFGS':
        # Use BFGS
        #aa = bfgs(nlml, theta, dnlml, [gp,X,y,R,w], maxiter=100, disp=False, full_output=True)
        aa = bfgs(nlml, theta, dnlml, [gp,X,y,R,w], maxiter=100, disp=True, full_output=True)
        theta = aa[0]; fvals = aa[1]; gvals = aa[2]; Bopt = aa[3]; funcCalls = aa[4]; gradcalls = aa[5]
        if aa[6] == 1:
            print "Maximum number of iterations exceeded." 
        elif aa[6] ==  2:
            print "Gradient and/or function calls not changing."
        mt = len(gp['meantheta'])
        gp['meantheta'] = theta[:mt]
        gp['covtheta']  = theta[mt:]
        return gp, fvals, gvals, funcCalls
    elif Flag == 'SCG':
        theta, listF = scg.scg(theta, nlml, dnlml, [gp,X,y,R,w], niters = 100)
        mt = len(gp['meantheta'])
        gp['meantheta'] = theta[:mt]
        gp['covtheta']  = theta[mt:]
        return gp, listF 
    else:
        raise Exception("Need to specify a method for optimization in gp_train")

def infExact(gp, X, y, Xstar, R=None, w=None, Rstar=None):
    # Exact inference for a GP with Gaussian likelihood. Compute a parametrization
    # of the posterior, the negative log marginal likelihood and its derivatives
    # w.r.t. the hyperparameters. See also "help infMethods".
    #
    if R==None:
        K     = feval(gp['covfunc'], gp['covtheta'], X)             # training covariances
        Kss   = feval(gp['covfunc'], gp['covtheta'], Xstar, 'diag')  # test covariances (Kss = self covariances, Kstar = cov between train and test cases)
        Kstar = feval(gp['covfunc'], gp['covtheta'], X, Xstar)      # test covariances (Kss = self covariances, Kstar = cov between train and test cases)
    else:
        K     = feval(gp['covfunc'], gp['covtheta'], X, R, w)                    # training covariances
        Kss   = feval(gp['covfunc'],gp['covtheta'], Xstar, R, w, 'diag', Rstar)   # test covariances
        Kstar = feval(gp['covfunc'],gp['covtheta'], X, R, w, Xstar, Rstar)       # test covariances
    
    ms     = feval(gp['meanfunc'], gp['meantheta'], Xstar)
    mean_y = feval(gp['meanfunc'], gp['meantheta'], X)

    K += sn2*np.eye(X.shape[0])            # Hardcoded noise

    try:
        L = np.linalg.cholesky(K)          # cholesky factorization of cov (lower triangular matrix)
    except linalg.linalg.LinAlgError:
        L = np.linalg.cholesky(nearPD(K))  # Find the "Nearest" covariance mattrix to K and do cholesky on that'

    alpha = solve_chol(L.T,y-mean_y)       # compute inv(K)*(y-mean(y))
    fmu   = ms + np.dot(Kstar.T,alpha)     # predicted means
    v     = np.linalg.solve(L, Kstar)
    tmp   = v*v
    fs2   = Kss - np.array([tmp.sum(axis=0)]).T  # predicted variances
    fs2[fs2 < 0.] = 0.                                # Remove numerical noise i.e. negative variances
    return [fmu, fs2]

def infFITC(gp, X, y, Xstar, R=None, w=None, Rstar=None):
    # FITC approximation to the posterior Gaussian process. The function is
    # equivalent to infExact with the covariance function:
    #   Kt = Q + G;   G = diag(diag(K-Q);   Q = Ku'*inv(Kuu + snu2*eye(nu))*Ku;
    # where Ku and Kuu are covariances w.r.t. to inducing inputs xu and
    # snu2 = sn2/1e6 is the noise of the inducing inputs. We fixed the standard
    # deviation of the inducing inputs to be a one per mil of the measurement noise
    # standard deviation.
    # The implementation exploits the Woodbury matrix identity
    #   inv(Kt) = inv(G) - inv(G)*Ku'*inv(Kuu+Ku*inv(G)*Ku')*Ku*inv(G)
    # in order to be applicable to large datasets. The computational complexity
    # is O(n nu^2) where n is the number of data points x and nu the number of
    # inducing inputs in xu.
    cov     = gp['covfunc']
    assert( isinstance(cov[-1],np.ndarray) )
    xu      = cov[-1]
    covfunc = cov[:-1]

    m  = feval(gp['meanfunc'], gp['meantheta'], X)          # evaluate mean vector
    ms = feval(gp['meanfunc'], gp['meantheta'], Xstar)      # evaluate mean vector
    n,D = X.shape
    nu = len(xu)

    [diagK,Kuu,Ku] = feval(covfunc, xu, gp['covtheta'], X)  # evaluate covariance matrix
    snu2           = 1.e-6 * sn2
    Kuu           += snu2*np.eye(nu)

    try:
        Luu  = np.linalg.cholesky(Kuu)                         # Kuu = Luu'*Luu
    except np.linalg.linalg.LinAlgError:
        Luu  = np.linalg.cholesky(nearPD(Kuu))                 # Kuu = Luu'*Luu, or at least closest SDP Kuu

    V     = np.linalg.solve(Luu.T,Ku)                          # V = inv(Luu')*Ku => V'*V = Q
    M     = np.reshape( np.diag(np.dot(V.T,V)), (diagK.shape[0],1) )
    g_sn20 = diagK + sn2 - (V*V).sum(axis=0).T                  # g = diag(K) + sn2 - diag(Q)
    g_sn2 = diagK + sn2 - M                  # g = diag(K) + sn2 - diag(Q)
    print "Values = ",g_sn20, g_sn2

    diagG = np.reshape( np.diag(g_sn2),(g_sn2.shape[0],1))
    V     = V/np.tile(diagG.T,(nu,1))  
    Lu    = np.linalg.cholesky( np.eye(nu) + np.dot(V,V.T) )   # Lu'*Lu = I + V*diag(1/g_sn2)*V'
    r     = (y-m)/np.sqrt(diagG)
    #be    = np.linalg.solve(Lu.T,np.dot(V,(r/np.sqrt(g_sn2))))
    be    = np.linalg.solve(Lu.T,np.dot(V,r))
    iKuu  = solve_chol(Luu,np.eye(nu))                       # inv(Kuu + snu2*I) = iKuu
    LuBe  = np.linalg.solve(Lu,be)
    alpha = np.linalg.solve(Luu,LuBe)                      # return the posterior parameters
    L     = solve_chol(np.dot(Lu,Luu),np.eye(nu)) - iKuu                    # Sigma-inv(Kuu)

    #Q = np.dot(Ku.T,np.linalg.solve(Kuu,Ku))
    #Lam = diagK - np.reshape( np.diag(Q), (diagK.shape[0],1) ) # diag(K) - diag(Q), Q = Kxu * Kuu^-1 * Kux
    #K = Q + Lam* np.eye(len(Lam))

    if R==None:
        Kss    = feval(covfunc[1], gp['covtheta'], Xstar, 'diag')   
        Kus    = feval(covfunc[1], gp['covtheta'], xu, Xstar)      
        Kuf    = feval(covfunc[1], gp['covtheta'], xu, X)      
    else:
        Kss   = feval(covfunc[1], gp['covtheta'], Xstar, R, w, 'diag', Rstar)   # test covariances
        Ku    = feval(covfunc[1], gp['covtheta'], Xstar, R, w, xu, Rstar)   # test covariances

    #Qsf = np.dot(Kus.T,np.linalg.solve(Kuu, Kuf))
    #fmu = ms + np.dot(Qsf,np.linalg.solve(K,y-m))  # predicted means
    fmu = ms + np.dot(Kus.T,alpha)  # predicted means

    #QQ = np.dot(Qsf,np.linalg.solve(K,Qsf.T))
    
    #fs2   = Kss - np.reshape( np.diag(QQ), (Kss.shape[0],1) ) # predicted variances

    vv    = np.linalg.solve(L.T, Kuf)
    tmp   = vv*vv
    fs2   = Kss - np.array([tmp.sum(axis=0)]).T  # predicted variances
    fs2[fs2 < 0.] = 0.                                # Remove numerical noise i.e. negative variances
    return [fmu, fs2]

def gp_pred(gp, X, y, Xstar, R=None, w=None, Rstar=None):
    # compute training set covariance matrix (K) and
    # (marginal) test predictions (Kss = self-cov; Kstar = corss-cov)
    covfunc = gp['covfunc']
    if covfunc[0][0] == 'kernels.covFITC':
        #Use FITC approximation
        fmu,fs2 = infFITC(gp, X, y, Xstar)
    else:
        fmu,fs2 = infExact(gp, X, y, Xstar, R, w, Rstar)
    return fmu, fs2

def solve_chol(A,B):
    return np.linalg.solve(A,np.linalg.solve(A.T,B))

def nlml(theta, gp, X, y, R=None, w=None):
    mt = len(gp['meantheta'])
    ct = len(gp['covtheta'])
    out = np.zeros(mt+ct)
    meantheta = theta[:mt]
    covtheta  = theta[mt:]
    n,D = X.shape
    
    if gp['covfunc'][0][0] == 'kernels.covFITC':
    # Do FITC Approximation
        cov = gp['covfunc']
        xu      = cov[-1]
        covfunc = cov[:-1]

        [diagK,Kuu,Ku] = feval(covfunc, xu, covtheta, X)  # evaluate covariance matrix

        m  = feval(gp['meanfunc'], meantheta, X)          # evaluate mean vector
        nu = Ku.shape[0]

        snu2 = 1.e-6 * sn2
        Kuu += snu2*np.eye(nu)

        try:
            Luu  = np.linalg.cholesky(Kuu)                         # Kuu = Luu'*Luu
        except np.linalg.linalg.LinAlgError:
            Luu  = np.linalg.cholesky(nearPD(Kuu))                 # Kuu = Luu'*Luu, or at least closest SDP Kuu

        V     = np.linalg.solve(Luu.T,Ku)                              # V = inv(Luu')*Ku => V'*V = Q
        g_sn2 = diagK - (V*V).sum(axis=0).T                            # g = diag(K) - diag(Q)
        diagG = np.reshape( np.diag(g_sn2),(g_sn2.shape[0],1)) 
        V     = V/np.tile(diagG.T,(nu,1))
        Lu    = np.linalg.cholesky( np.eye(nu) + np.dot(V,V.T) )  # Lu'*Lu = I + V*diag(1/g_sn2)*V'

        r     = (y-m)/np.sqrt(diagG)
        V     = np.linalg.solve(Luu.T,Ku)                              # V = inv(Luu')*Ku => V'*V = Q
        #be    = np.linalg.solve(Lu.T,np.dot(V,(r/np.sqrt(g_sn2))))
        be    = np.linalg.solve(Lu.T,np.dot(V,r))
        iKuu  = solve_chol(Luu,np.eye(nu))                       # inv(Kuu + snu2*I) = iKuu
        LuBe  = np.linalg.solve(Lu,be)
        alpha = np.linalg.solve(Luu,LuBe)                      # return the posterior parameters
        L     = solve_chol(np.dot(Lu,Luu),np.eye(nu)) - iKuu                    # Sigma-inv(Kuu)

        #Q = np.dot(Ku.T,np.linalg.solve(Kuu,Ku))
        #Lam = diagK - np.reshape( np.diag(Q), (diagK.shape[0],1) ) # diag(K) - diag(Q), Q = Kxu * Kuu^-1 * Kux
        #K = Q + Lam* np.eye(len(Lam))

        ms = feval(gp['meanfunc'], meantheta, X)

        # cholesky factorization of the covariance
        #try:
        #    L = np.linalg.cholesky(K)      # lower triangular matrix
        #except np.linalg.linalg.LinAlgError:
        #    L = np.linalg.cholesky(nearPD(K))                 # Find the "Nearest" covariance mattrix to K and do cholesky on that
        # compute inv(K)*y
        #alpha = solve_chol(L.T,y-m)
        #alpha = np.linalg.solve(K,y-m)
        # compute the negative log marginal likelihood
        aa = ( np.log(np.diag(Lu)).sum() + np.log(g_sn2).sum() + n*np.log(2.*np.pi) + np.dot(r.T,r) - np.dot(be.T,be) )/2.
        #aa =  ( 0.5*np.dot((y-ms).T,alpha) + (np.log(np.diag(L))).sum(axis=0) + 0.5*n*np.log(2.*np.pi) )
        #aa =  ( 0.5*np.dot((y-ms).T,alpha) + (np.log(np.linalg.det(K))) + 0.5*n*np.log(2.*np.pi) )
        #t1 = 0.5*np.dot((y-ms).T,alpha)
        #t2 = (np.log(np.linalg.det(K)))
        #t3 = 0.5*n*np.log(2.*np.pi) 
        #t4 = np.linalg.det(K)
        #print t1, t2, t3, t4
        #aa = t1 + t2 + t3 

    else:
        # Do Exact inference
        # compute training set covariance matrix
        if R==None:
            K = feval(gp['covfunc'], covtheta, X)
        else:
            K = feval(gp['covfunc'], covtheta, X, R, w)     

        K += sn2*np.eye(X.shape[0])
        m = feval(gp['meanfunc'], meantheta, X)
    
        # cholesky factorization of the covariance
        try:
            L = np.linalg.cholesky(K)      # lower triangular matrix
        except np.linalg.linalg.LinAlgError:
            L = np.linalg.cholesky(nearPD(K))                 # Find the "Nearest" covariance mattrix to K and do cholesky on that
        # compute inv(K)*y
        #alpha = solve_chol(L.T,y-m)
        alpha = np.linalg.solve(K,y-m) 
        # compute the negative log marginal likelihood
        aa =  ( 0.5*np.dot((y-m).T,alpha) + (np.log(np.diag(L))).sum(axis=0) + 0.5*n*np.log(2.*np.pi) )
    return aa[0]
    
def dnlml(theta, gp, X, y, R=None, w=None):
    mt = len(gp['meantheta'])
    ct = len(gp['covtheta'])
    out = np.zeros(mt+ct)
    meantheta = theta[:mt]
    covtheta  = theta[mt:]

    if gp['covfunc'][0][0] == 'kernels.covFITC':
        # Do FITC Approximation
        cov = gp['covfunc']
        xu      = cov[-1]
        covfunc = cov[1:-1][0]
        nu = len(xu)
        [diagK,Kuu,Ku] = feval(cov[:-1], xu, covtheta, X)  # evaluate covariance matrix

        snu2 = 1.e-6 * sn2
        Kuu += snu2*np.eye(nu)

        m  = feval(gp['meanfunc'], meantheta, X)          # evaluate mean vector
        n,D = X.shape
        nu = Ku.shape[0]


        try:
            Luu  = np.linalg.cholesky(Kuu)                         # Kuu = Luu'*Luu
        except np.linalg.linalg.LinAlgError:
            Luu  = np.linalg.cholesky(nearPD(Kuu))                 # Kuu = Luu'*Luu, or at least closest SDP Kuu

        V     = np.linalg.solve(Luu.T,Ku)                              # V = inv(Luu')*Ku => V'*V = Q
        g_sn2 = diagK - (V*V).sum(axis=0).T                            # g = diag(K) - diag(Q)
        diagG = np.reshape( np.diag(g_sn2),(g_sn2.shape[0],1)) 
        V     = V/np.tile(diagG.T,(nu,1))
        Lu    = np.linalg.cholesky( np.eye(nu) + np.dot(V,V.T) )  # Lu'*Lu = I + V*diag(1/g_sn2)*V'
        r     = (y-m)/np.sqrt(diagG)
        #be    = np.linalg.solve(Lu.T,np.dot(V,(r/np.sqrt(g_sn2))))
        be    = np.linalg.solve(Lu.T,np.dot(V,r))
        V     = np.linalg.solve(Luu.T,Ku)                              # V = inv(Luu')*Ku => V'*V = Q
        iKuu  = solve_chol(Luu,np.eye(nu))                       # inv(Kuu + snu2*I) = iKuu
        LuBe  = np.linalg.solve(Lu,be)
        alpha = np.linalg.solve(Luu,LuBe)                      # return the posterior parameters
        L     = solve_chol(np.dot(Lu,Luu),np.eye(nu)) - iKuu                    # Sigma-inv(Kuu)'''

        #Q = np.dot(Ku.T,np.linalg.solve(Kuu,Ku))
        #Lam = diagK - np.reshape( np.diag(Q), (diagK.shape[0],1) ) # diag(K) - diag(Q), Q = Kxu * Kuu^-1 * Kux
        #K = Q + Lam* np.eye(len(Lam))

        # cholesky factorization of the covariance
        #try:
        #    L = np.linalg.cholesky(K) # lower triangular matrix
        #except np.linalg.linalg.LinAlgError:
        #    L = np.linalg.cholesky(nearPD(K)) # Find the "Nearest" covariance mattrix to K and do cholesky on that
        #alpha = solve_chol(L.T,y-m)
        #alpha = np.linalg.solve(K,y-m)
        #W = np.linalg.solve(L.T,np.linalg.solve(L,np.eye(n)))-np.dot(alpha,alpha.T)
        #W = np.linalg.solve(K,np.eye(n))-np.dot(alpha,alpha.T)

        W     = Ku/np.tile(diagG.T,(nu,1))
        W     = np.linalg.solve((Kuu + np.dot(W,W.T) + snu2*np.eye(nu)).T, Ku)
        al    = ( (y-m) - np.dot(W.T,np.dot(W,(y-m)/diagG)) )/diagG
        B     = np.dot(iKuu,Ku)
        Wdg   = W/np.tile(diagG.T,(nu,1))
        w     = np.dot(B,al)

        '''al = r/np.sqrt(g_sn2) - np.dot(V.T,LuBe)/g_sn2          # al = (Kt+sn2*eye(n))\y
        B = np.dot(iKuu,Ku)
        w = np.dot(B,al)
        W = np.linalg.solve(Lu.T, (V/np.tile(g_sn2.T,(nu,1))))'''

        if R==None:
            for ii in range(len(meantheta)):
                out[ii] = -1.*np.dot(feval(gp['meanfunc'], meantheta, X, ii).T,al)

            kk = len(gp['meantheta'])
            for ii in range(len(covtheta)):
                [ddiagKi,dKuui,dKui] = feval(cov[:-1], covtheta, X, None, ii)  # eval cov deriv
                R = 2.*dKui - np.dot(dKuui,B)
                v = ddiagKi - (R*B).sum(axis=0).T   # diag part of cov deriv
                out[ii+kk] = ( np.dot(ddiagKi.T,(1./g_sn2)) + np.dot(w.T,(np.dot(dKuui,w)-2.*np.dot(dKui,al))-np.dot(al.T,(v*al) \
                               - np.dot((Wdg*Wdg).sum(axis=0),v) - np.dot(R,Wdg.T)*np.dot(B,Wdg.T))) )/2.
                #A = feval(covfunc, covtheta, X, None, ii) 
                #out[ii+kk] = ( W * feval(covfunc, covtheta, X, None, ii) ).sum()/2.
        else:
            for ii in range(len(meantheta)):
                out[ii] = (W*feval(gp['meanfunc'], meantheta, X, R, w, ii)).sum()/2.
            kk = len(meantheta)
            for ii in range(len(covtheta)):
                out[ii+kk] = (W*feval(covfunc, covtheta, X, R, w, ii)).sum()/2.
    else:
        # Do Exact inference
        W = get_W(theta, gp, X, y, R, w)

        if R==None:
            for ii in range(len(meantheta)):
                out[ii] = (W*feval(gp['meanfunc'], meantheta, X, ii)).sum()/2.
            kk = len(meantheta)
            for ii in range(len(covtheta)):
                out[ii+kk] = (W*feval(gp['covfunc'], covtheta, X, None, ii)).sum()/2.
        else:
            for ii in range(len(meantheta)):
                out[ii] = (W*feval(gp['meanfunc'], meantheta, X, R, w, ii)).sum()/2.
            kk = len(meantheta)
            for ii in range(len(gpcovtheta)):
                out[ii+kk] = (W*feval(gp['covfunc'], covtheta, X, R, w, ii)).sum()/2. 
    return out
         
def get_W(theta, gp, X, y, R=None, w=None):
    '''Precompute W for convenience.'''
    n  = X.shape[0]
    mt = len(gp['meantheta'])
    meantheta = theta[:mt]
    covtheta  = theta[mt:]
    # compute training set covariance matrix
    if R==None:
        K = feval(gp['covfunc'], covtheta, X)
    else:
        K = feval(gp['covfunc'], covtheta, X, R, w)
    K += sn2*np.eye(X.shape[0])

    m = feval(gp['meanfunc'], meantheta, X)
    # cholesky factorization of the covariance
    try:
        L = np.linalg.cholesky(K) # lower triangular matrix
    except np.linalg.linalg.LinAlgError:
        L = np.linalg.cholesky(nearPD(K)) # Find the "Nearest" covariance mattrix to K and do cholesky on that
    alpha = solve_chol(L.T,y-m)
    W = np.linalg.solve(L.T,np.linalg.solve(L,np.eye(n)))-np.dot(alpha,alpha.T)
    return W

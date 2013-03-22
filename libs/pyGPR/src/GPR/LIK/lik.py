import numpy as np
import Tools.general
from scipy.special import erf

def likErf(hyp=None, y=None, mu=None, s2=None, inffunc=None, der=None, nargout=1):
#function [varargout] = likErf(hyp, y, mu, s2, inf, i)
    # likErf - Error function or cumulative Gaussian likelihood function for binary
    # classification or probit regression. The expression for the likelihood is 
    #   likErf(t) = (1+erf(t/sqrt(2)))/2 = normcdf(t).
    #
    # Several modes are provided, for computing likelihoods, derivatives and moments
    # respectively, see likFunctions.m for the details. In general, care is taken
    # to avoid numerical issues when the arguments are extreme.
    # 
    # Copyright (c) by Carl Edward Rasmussen and Hannes Nickisch, 2010-07-22.
    #
    # See also LIKFUNCTIONS.M.

    if mu == None: 
        return [0] #end   # report number of hyperparameters
    if not y == None:
        y = np.sign(y)
        y[y==0] = 1
    else:
         y = 1; # allow only +/- 1 values
    #end 

    if inffunc == None:                              # prediction mode if inf is not present
        y = y*np.ones_like(mu)                                       # make y a vector
        s2zero = True; 
        if not s2 == None: 
            if np.linalg.norm(s2)>0:
                 s2zero = False # s2==0 ?
            #end
        #end         
        if s2zero:                                         # log probability evaluation
            [p,lp] = cumGauss(y,mu,2)
        else:                                                              # prediction
            lp = Tools.general.feval(['lik.likErf'],hyp, y, mu, s2, 'inf.infEP')
            p = np.exp(lp)
        #end
        if nargout>1:
            ymu = 2*p-1                                                # first y moment
            if nargout>2:
                ys2 = 4*p*(1-p)                                        # second y moment
                varargout = [lp,ymu,ys2]
            else:
                varargout = [lp,ymu]
            #end
        else:
            varargout = lp
        #end
    else:                                                   # inference mode
        if inffunc == 'inf.infLaplace':
            if der == None:                                  # no derivative mode
                f = mu; yf = y*f                             # product latents and labels
                [p,lp] = cumGauss(y,f,2)
                if nargout>1:                                # derivative of log likelihood
                    n_p = gauOverCumGauss(yf,p)
                    dlp = y*n_p                              # derivative of log likelihood
                    if nargout>2:                            # 2nd derivative of log likelihood
                        d2lp = -n_p**2 - yf*n_p
                        if nargout>3:                        # 3rd derivative of log likelihood
                            d3lp = 2*y*n_p**3 + 3*f*n_p**2 + y*(f**2-1)*n_p 
                            varargout = [lp,dlp,d2lp,d3lp]
                        else:
                            varargout = [lp,dlp,d2lp]
                        #end
                    else:
                        varargout = [lp,dlp]
                    #end
                else:
                    varargout = lp
                #end
            else:                                              # derivative mode
                varargout = nargout*[]                         # derivative w.r.t. hypers
            #end

        if inffunc == 'inf.infEP':
            if der == None:                                            # no derivative mode
                #print "s2 = ",s2
                z = mu/np.sqrt(1+s2) 
                [junk,lZ] = cumGauss(y,z,2)                            # log part function
                if not y == None:
                     z = z*y
                if nargout>1:
                    if y == None: y = 1
                    n_p = gauOverCumGauss(z,np.exp(lZ))
                    dlZ = y*n_p/np.sqrt(1.+s2)                      # 1st derivative wrt mean
                    if nargout>2:
                        d2lZ = -n_p*(z+n_p)/(1.+s2)                 # 2nd derivative wrt mean
                        varargout = [lZ,dlZ,d2lZ]
                    else:
                        varargout = [lZ,dlZ]
                    #end
                else:
                    varargout = lZ
                #end
            else:                                                   # derivative mode
                varargout = []                                     # deriv. wrt hyp.lik
            #end
  
        if inffunc == 'inf.infVB':
            if der == None:                                           # no derivative mode
                # naive variational lower bound based on asymptotical properties of lik
                # normcdf(t) -> -(t*A_hat^2-2dt+c)/2 for t->-np.inf (tight lower bound)
                d =  0.158482605320942;
                c = -1.785873318175113;
                ga = s2; n = len(ga); b = d*y*np.ones((n,1)); db = np.zeros((n,1)); d2b = db
                h = -2.*c*np.ones((n,1)); h[ga>1] = np.inf; dh = np.zeros((n,1)); d2h = dh   
                varargout = [h,b,dh,db,d2h,d2b]
            else:                                                  # derivative mode
                varargout = []                                     # deriv. wrt hyp.lik
            #end
        #end
    #end
    return varargout

def cumGauss(y=None,f=None,nargout=1):
    #function [p,lp] = cumGauss(y,f)
    if not y == None: 
        yf = y*f 
    else:
        yf = f 
    #end     # product of latents and labels
    p  = (1. + erf(yf/np.sqrt(2.)))/2.                                       # likelihood
    if nargout>1: 
        lp = logphi(yf,p)
        return p,lp 
    else:
        return p
    #end                          # log likelihood

    # safe implementation of the log of phi(x) = \int_{-\infty}^x N(f|0,1) df
    # logphi(z) = log(normcdf(z))

def logphi(z,p):
    #function lp = logphi(z,p)
    lp = np.zeros_like(z)                              # allocate memory
    zmin = -6.2; zmax = -5.5;
    ok = z>zmax                                        # safe evaluation for large values
    bd = z<zmin                                        # use asymptotics
    nok = np.logical_not(ok)
    ip = np.logical_and(nok,np.logical_not(bd)) # interpolate between both of them
    lam = 1/(1.+np.exp( 25.*(0.5-(z[ip]-zmin)/(zmax-zmin)) ))  # interp. weights
    lp[ok] = np.log(p[ok])
    # use lower and upper bound acoording to Abramowitz&Stegun 7.1.13 for z<0
    # lower -log(pi)/2 -z.^2/2 -log( sqrt(z.^2/2+2   ) -z/sqrt(2) )
    # upper -log(pi)/2 -z.^2/2 -log( sqrt(z.^2/2+4/pi) -z/sqrt(2) )
    # the lower bound captures the asymptotics
    lp[nok] = -np.log(np.pi)/2. -z[nok]**2/2. - np.log( np.sqrt(z[nok]**2/2.+2.) - z[nok]/np.sqrt(2.) )
    lp[ip] = (1-lam)*lp[ip] + lam*np.log( p[ip] )
    return lp
  
def gauOverCumGauss(f,p):
#function n_p = gauOverCumGauss(f,p)
    n_p = np.zeros_like(f)       # safely compute Gaussian over cumulative Gaussian
    ok = f>-5                    # naive evaluation for large values of f
    n_p[ok] = (np.exp(-f[ok]**2/2)/np.sqrt(2*np.pi)) / p[ok] 

    bd = f<-6                    # tight upper bound evaluation
    n_p[bd] = np.sqrt(f[bd]**2/4+1)-f[bd]/2

    interp = np.logical_and(np.logical_not(ok),np.logical_not(bd)) # linearly interpolate between both of them
    tmp = f[interp]
    lam = -5. - f[interp]
    n_p[interp] = (1-lam)*(np.exp(-tmp**2/2)/np.sqrt(2*np.pi))/p[interp] + \
                                                 lam *(np.sqrt(tmp**2/4+1)-tmp/2);
    return n_p

def likGauss(hyp=None, y=None, mu=None, s2=None, inffunc=None, der=None, nargout=1):
    # likGauss - Gaussian likelihood function for regression. The expression for the 
    # likelihood is 
    #   likGauss(t) = exp(-(t-y)^2/2*sn^2) / sqrt(2*pi*sn^2),
    # where y is the mean and sn is the standard deviation.
    #
    # The hyperparameters are:
    #
    # hyp = [  log(sn)  ]
    #
    # Several modes are provided, for computing likelihoods, derivatives and moments
    # respectively, see likFunctions.m for the details. In general, care is taken
    # to avoid numerical issues when the arguments are extreme.
    #
    # Copyright (c) by Carl Edward Rasmussen and Hannes Nickisch, 2013-01-21
    #
    # See also LIKFUNCTIONS.M.

    if mu == None:
        return [1]    # report number of hyperparameters

    sn2 = np.exp(2.*hyp)

    if inffunc == None:                                  # prediction mode if inffunc is not present
        if y == None:
            y = np.zeros_like(mu)
        #end
        s2zero = True
        if not (s2 == None):
            if np.linalg.norm(s2) > 0:
                s2zero = False 
            #end
        #end             # s2==0 ?
        if s2zero:                                                        # log probability
            lp = -(y-mu)**2 /sn2/2 - np.log(2.*np.pi*sn2)/2. 
            s2 = 0.
        else:
            lp = Tools.general.feval(['lik.likGauss'],hyp, y, mu, s2, 'inf.infEP')                    # prediction
        #end
        if nargout>1:
            ymu = mu;                                                       # first y moment
            if nargout>2:
                ys2 = s2 + sn2;                                              # second y moment
                varargout = [lp,ymu,ys2]
            else:
                varargout = [lp,ymu]
            #end
        else:
            varargout = lp
        #end
    else:
        if inffunc == 'inf.infLaplace':
            if der == None:                                                 # no derivative mode
                if not y: y=0 #end
                ymmu = y-mu
                lp = -ymmu**2/(2*sn2) - np.log(2*np.pi*sn2)/2. 
                if nargout>1:
                    dlp = ymmu/sn2                          # dlp, derivative of log likelihood
                    if nargout>2:                        # d2lp, 2nd derivative of log likelihood
                        d2lp = -np.ones_like(ymmu)/sn2
                        if nargout>3:                      # d3lp, 3rd derivative of log likelihood
                            d3lp = np.zeros_like(ymmu)
                            varargout = [lp,dlp,d2lp,d3lp]
                        else:
                            varargout = [lp,dlp,d2lp]
                        #end
                    else:
                        varargout = [lp,dlp]
                    #end
                else:
                    varargout = lp
                #end
            else:                                         # derivative mode
                lp_dhyp   = (y-mu)**2/sn2 - 1             # derivative of log likelihood w.r.t. hypers
                dlp_dhyp  = 2*(mu-y)/sn2                  # first derivative,
                d2lp_dhyp = 2*np.ones_like(mu)/sn2        # and also of the second mu derivative
                varargout = [lp_dhyp,dlp_dhyp,d2lp_dhyp]
            #end

        elif inffunc == 'inf.infEP':
            if der == None:                                                 # no derivative mode
                lZ = -(y-mu)**2/(sn2+s2)/2. - np.log(2*np.pi*(sn2+s2))/2.   # log part function
                if nargout>1:
                    dlZ  = (y-mu)/(sn2+s2)                        # 1st derivative w.r.t. mean
                    if nargout>2:
                        d2lZ = -1/(sn2+s2)                          # 2nd derivative w.r.t. mean
                        varargout = [lZ,dlZ,d2lZ]
                    else:
                        varargout = [lZ,dlZ]
                    #end
                else:
                    varargout = lZ
                #end
            else:                                                           # derivative mode
                dlZhyp = ((y-mu)**2/(sn2+s2)-1) / (1+s2/sn2)       # deriv. w.r.t. hyp.lik
                varargout = dlZhyp
            #end

        elif inffunc ==  'inf.infVB':
            if der == None:
                # variational lower site bound
                # t(s) = exp(-(y-s)^2/2sn2)/sqrt(2*pi*sn2)
                # the bound has the form: b*s - s.^2/(2*ga) - h(ga)/2 with b=y/ga
                ga = s2; n = len(ga); b = y/ga; y = y*np.ones((n,1))
                db  = -y/ga**2 
                d2b = 2*y/ga**3
                h    = np.zeros((n,1)); dh = h; d2h = h            # allocate memory for return args
                id = (ga <= sn2 + 1e-8)                            # OK below noise variance
                h[id]   = y[id]**2/ga[id] + np.log(2*np.pi*sn2); h[np.logical_not(id)] = np.inf
                dh[id]  = -y[id]**2/ga[id]**2
                d2h[id] = 2*y[id]**2/ga[id]**3
                id = ga < 0; h[id] = np.inf; dh[id] = 0; d2h[id] = 0         # neg. var. treatment
                varargout = [h,b,dh,db,d2h,d2b]
            else:
                ga = s2; n = len(ga); 
                dhhyp = np.zeros((n,1)); dhhyp[ga<=sn2] = 2
                dhhyp[ga<0] = 0                  # negative variances get a special treatment
                varargout = dhhyp                # deriv. w.r.t. hyp.lik
            #end
        else:
            raise Exception('Incorrect inference in lik.Gauss\n');
       #end
    return varargout

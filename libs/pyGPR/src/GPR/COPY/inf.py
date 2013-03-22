import numpy as np
import Tools.general
from solve_chol import solve_chol
from copy import copy, deepcopy
from utils import randperm

class postStruct:
    def __init__(self):
        self.alpha = np.array([])
        self.L     = np.array([])
        self.sW    = np.array([])

def infFITC_EP(hyp, meanfunc, covfunc, likfunc, x, y, nargout=1):
#function [post nlZ dnlZ] = infFITC_EP(hyp, mean, cov, lik, x, y)

    # FITC-EP approximation to the posterior Gaussian process. The function is
    # equivalent to infEP with the covariance function:
    #   Kt = Q + G; G = diag(g); g = diag(K-Q);  Q = Ku'*inv(Kuu + snu2*eye(nu))*Ku;
    # where Ku and Kuu are covariances w.r.t. to inducing inputs xu and
    # snu2 = sn2/1e6 is the noise of the inducing inputs. We fixed the standard
    # deviation of the inducing inputs snu to be a one per mil of the measurement 
    # noise's standard deviation sn. In case of a likelihood without noise
    # parameter sn2, we simply use snu2 = 1e-6.
    # For details, see The Generalized FITC Approximation, Andrew Naish-Guzman and
    #                  Sean Holden, NIPS, 2007.
    #
    # The implementation exploits the Woodbury matrix identity
    #   inv(Kt) = inv(G) - inv(G)*Ku'*inv(Kuu+Ku*inv(G)*Ku')*Ku*inv(G)
    # in order to be applicable to large datasets. The computational complexity
    # is O(n nu^2) where n is the number of data points x and nu the number of
    # inducing inputs in xu.
    # The posterior N(f|h,Sigma) is given by h = m+mu with mu = nn + P'*gg and
    # Sigma = inv(inv(K)+diag(W)) = diag(d) + P'*R0'*R'*R*R0*P. Here, we use the
    # site parameters: b,w=$b,\pi$=tnu,ttau, P=$P'$, nn=$\nu$, gg=$\gamma$
    #             
    # The function takes a specified covariance function (see covFunction.m) and
    # likelihood function (see likFunction.m), and is designed to be used with
    # gp.m and in conjunction with covFITC. 
    #
    # Copyright (c) by Hannes Nickisch, 2012-11-09.
    #
    # See also INFMETHODS.M, COVFITC.M.

    cov1 = cov{1}; if isa(cov1, 'function_handle'), cov1 = func2str(cov1); end
    if ~strcmp(cov1,'covFITC'); error('Only covFITC supported.'), end    # check cov

    persistent last_ttau last_tnu              # keep tilde parameters between calls
    tol = 1e-4; max_sweep = 20; min_sweep = 2;     # tolerance to stop EP iterations

    inf = 'infEP';
    n = size(x,1);
    [diagK,Kuu,Ku] = feval(cov{:}, hyp.cov, x);         # evaluate covariance matrix
    if ~isempty(hyp.lik)                          # hard coded inducing inputs noise
        sn2 = exp(2*hyp.lik(end)); snu2 = 1e-6*sn2;               # similar to infFITC
    else
        snu2 = 1e-6;
    end
    nu = size(Kuu,1);
    m = feval(mean{:}, hyp.mean, x);                      # evaluate the mean vector

    rot180   = @(A)   rot90(rot90(A));                     # little helper functions
    chol_inv = @(A) rot180(chol(rot180(A))')\eye(nu);                 # chol(inv(A))
    R0 = chol_inv(Kuu+snu2*eye(nu));           # initial R, used for refresh O(nu^3)
    V = R0*Ku; d0 = diagK-sum(V.*V,1)';    # initial d, needed for refresh O(n*nu^2)

    # A note on naming: variables are given short but descriptive names in 
    # accordance with Rasmussen & Williams "GPs for Machine Learning" (2006): mu
    # and s2 are mean and variance, nu and tau are natural parameters. A leading t
    # means tilde, a subscript _ni means "not i" (for cavity parameters), or _n
    # for a vector of cavity parameters.

    # marginal likelihood for ttau = tnu = zeros(n,1); equals n*log(2) for likCum*
    nlZ0 = -sum(feval(lik{:}, hyp.lik, y, m, diagK, inf));
    if any(size(last_ttau) ~= [n 1])      # find starting point for tilde parameters
        ttau = zeros(n,1);             # initialize to zero if we have no better guess
        tnu  = zeros(n,1);
        [d,P,R,nn,gg] = epfitcRefresh(d0,Ku,R0,V, ttau,tnu); # compute initial repres.
        nlZ = nlZ0;
    else:
        ttau = last_ttau;                    # try the tilde values from previous call
        tnu  = last_tnu;
        [d,P,R,nn,gg] = epfitcRefresh(d0,Ku,R0,V, ttau,tnu); # compute initial repres.
        nlZ = epfitcZ(d,P,R,nn,gg,ttau,tnu,d0,R0,Ku,y,lik,hyp,m,inf);
        if nlZ > nlZ0                                           # if zero is better ..
            ttau = zeros(n,1);                    # .. then initialize with zero instead
            tnu  = zeros(n,1);
            [d,P,R,nn,gg] = epfitcRefresh(d0,Ku,R0,V, ttau,tnu);       # initial repres.
            nlZ = nlZ0;
        #end
    #end

    nlZ_old = Inf; sweep = 0;               # converged, max. sweeps or min. sweeps?
    while (abs(nlZ-nlZ_old) > tol && sweep < max_sweep) || sweep<min_sweep
        nlZ_old = nlZ; sweep = sweep+1;
        for i = randperm(n)       # iterate EP updates (in random order) over examples
            pi = P(:,i); t = R*(R0*pi);                            # temporary variables
            sigmai = d(i) + t'*t; mui = nn(i) + pi'*gg;           # post moments O(nu^2)
    
            tau_ni = 1/sigmai-ttau(i);          #  first find the cavity distribution ..
            nu_ni = mui/sigmai+m(i)*tau_ni-tnu(i);          # .. params tau_ni and nu_ni
   
            # compute the desired derivatives of the indivdual log partition function
            [lZ, dlZ, d2lZ] = feval(lik{:}, hyp.lik, y(i), nu_ni/tau_ni, 1/tau_ni, inf);
            ttaui =                            -d2lZ  /(1+d2lZ/tau_ni);
            ttaui = max(ttaui,0);     # enforce positivity i.e. lower bound ttau by zero
            tnui  = ( dlZ + (m(i)-nu_ni/tau_ni)*d2lZ )/(1+d2lZ/tau_ni);
            [d,P(:,i),R,nn,gg,ttau,tnu] = ...                    # update representation
                epfitcUpdate(d,P(:,i),R,nn,gg, ttau,tnu,i,ttaui,tnui, m,d0,Ku,R0);
        #end
        # recompute since repeated rank-one updates can destroy numerical precision
        [d,P,R,nn,gg] = epfitcRefresh(d0,Ku,R0,V, ttau,tnu);
        [nlZ,nu_n,tau_n] = epfitcZ(d,P,R,nn,gg,ttau,tnu,d0,R0,Ku,y,lik,hyp,m,inf);
    #end

    if sweep == max_sweep
        warning('maximum number of sweeps reached in function infEP')
    end
    last_ttau = ttau; last_tnu = tnu;                       # remember for next call

    post.sW = sqrt(ttau);                  # unused for FITC_EP prediction with gp.m
    dd = 1./(d0+1./ttau);
    alpha = tnu./ttau.*dd;
    RV = R*V; R0tV = R0'*V;
    alpha = alpha - (RV'*(RV*alpha)).*dd;     # long alpha vector for ordinary infEP
    post.alpha = R0tV*alpha;       # alpha = R0'*V*inv(Kt+diag(1./ttau))*(tnu./ttau)
    B = R0tV.*repmat(dd',nu,1); L = B*R0tV'; B = B*RV';
    post.L = B*B' - L;                      # L = -R0'*V*inv(Kt+diag(1./ttau))*V'*R0

    if nargout>2                                           # do we want derivatives?
        dnlZ = hyp;                                   # allocate space for derivatives
        RVdd = RV.*repmat(dd',nu,1);
        for i=1:length(hyp.cov)
            [ddiagK,dKuu,dKu] = feval(cov{:}, hyp.cov, x, [], i); # eval cov derivatives
            dA = 2*dKu'-R0tV'*dKuu;                                       # dQ = dA*R0tV
            w = sum(dA.*R0tV',2); v = ddiagK-w;   # w = diag(dQ); v = diag(dK)-diag(dQ);
            z = dd'*(v+w) - sum(RVdd.*RVdd,1)*v - sum(sum( (RVdd*dA)'.*(R0tV*RVdd') ));
            dnlZ.cov(i) = (z - alpha'*(alpha.*v) - (alpha'*dA)*(R0tV*alpha))/2;
        #end
        for i = 1:numel(hyp.lik)                                   # likelihood hypers
            dlik = feval(lik{:}, hyp.lik, y, nu_n./tau_n, 1./tau_n, inf, i);
            dnlZ.lik(i) = -sum(dlik);
            if i==numel(hyp.lik)
                # since snu2 is a fixed fraction of sn2, there is a covariance-like term
                # in the derivative as well
                v = sum(R0tV.*R0tV,1)';
                z = sum(sum( (RVdd*R0tV').^2 )) - sum(RVdd.*RVdd,1)*v;
                z = z + post.alpha'*post.alpha - alpha'*(v.*alpha);
                dnlZ.lik(i) = dnlZ.lik(i) + snu2*z;
            #end
        #end
        [junk,dlZ] = feval(lik{:}, hyp.lik, y, nu_n./tau_n, 1./tau_n, inf);# mean hyps
        for i = 1:numel(hyp.mean)
            dm = feval(mean{:}, hyp.mean, x, i);
            dnlZ.mean(i) = -dlZ'*dm;
        #end
    #end

# refresh the representation of the posterior from initial and site parameters
# to prevent possible loss of numerical precision after many epfitcUpdates
# effort is O(n*nu^2) provided that nu<n
def epfitcRefresh(d0,P0,R0,R0P0, w,b):
    #function [d,P,R,nn,gg] = epfitcRefresh(d0,P0,R0,R0P0, w,b)
    nu = size(R0,1);                                   # number of inducing points
    rot180   = @(A)   rot90(rot90(A));                   # little helper functions
    chol_inv = @(A) rot180(chol(rot180(A))')\eye(nu);               # chol(inv(A))
    t  = 1./(1+d0.*w);                                   # temporary variable O(n)
    d  = d0.*t;                                                             # O(n)
    P  = repmat(t',nu,1).*P0;                                            # O(n*nu)
    T  = repmat((w.*t)',nu,1).*R0P0;                # temporary variable O(n*nu^2)
    R  = chol_inv(eye(nu)+R0P0*T');                                    # O(n*nu^3)
    nn = d.*b;                                                              # O(n)
    gg = R0'*(R'*(R*(R0P0*(t.*b))));                                     # O(n*nu)

# compute the marginal likelihood approximation
# effort is O(n*nu^2) provided that nu<n
def epfitcZ(d,P,R,nn,gg,ttau,tnu, d0,R0,P0, y,lik,hyp,m,inf):
#function [nlZ,nu_n,tau_n] = epfitcZ(d,P,R,nn,gg,ttau,tnu, d0,R0,P0, y,lik,hyp,m,inf)
    T = (R*R0)*P;                                             # temporary variable
    diag_sigma = d + sum(T.*T,1)'; mu = nn + P'*gg;       # post moments O(n*nu^2)
    tau_n = 1./diag_sigma-ttau;              # compute the log marginal likelihood
    nu_n  = mu./diag_sigma-tnu+m.*tau_n;            # vectors of cavity parameters
    lZ = feval(lik{:}, hyp.lik, y, nu_n./tau_n, 1./tau_n, inf);
    nu = size(gg,1);
    U = (R0*P0)'.*repmat(1./sqrt(d0+1./ttau),1,nu);
    L = chol(eye(nu)+U'*U);
    ld = 2*sum(log(diag(L))) + sum(log(d0+1./ttau)) + sum(log(ttau));
    t = T*tnu; tnu_Sigma_tnu = tnu'*(d.*tnu) + t'*t;
    nlZ = ld/2 -sum(lZ) -tnu_Sigma_tnu/2  ...
        -(nu_n-m.*tau_n)'*((ttau./tau_n.*(nu_n-m.*tau_n)-2*tnu)./(ttau+tau_n))/2 ...
        +sum(tnu.^2./(tau_n+ttau))/2-sum(log(1+ttau./tau_n))/2;

# update the representation of the posterior to reflect modification of the site 
# parameters by w(i) <- wi and b(i) <- bi
# effort is O(nu^2)
# Pi = P(:,i) is passed instead of P to prevent allocation of a new array
def epfitcUpdate(d,Pi,R,nn,gg, w,b, i,wi,bi, m,d0,P0,R0):
    #function [d,Pi,R,nn,gg,w,b] = epfitcUpdate(d,Pi,R,nn,gg, w,b, i,wi,bi, m,d0,P0,R0)
    dwi = wi-w(i); dbi = bi-b(i);
    hi = nn(i) + m(i) + Pi'*gg;                   # posterior mean of site i O(nu)
    t = 1+dwi*d(i);
    d(i) = d(i)/t;                                                          # O(1)
    nn(i) = d(i)*bi;                                                        # O(1)
    r = 1+d0(i)*w(i);
    r = r*r/dwi + r*d0(i);
    v = R*(R0*P0(:,i));
    r = 1/(r+v'*v);
    if r>0:
        R = cholupdate(R,sqrt( r)*R'*v,'-');
    else:
        R = cholupdate(R,sqrt(-r)*R'*v,'+');
    #end
    gg = gg + ((dbi-dwi*(hi-m(i)))/t)*(R0'*(R'*(R*(R0*Pi))));            # O(nu^2)
    w(i) = wi; b(i) = bi;                            # update site parameters O(1)
    Pi = Pi/t;                                                             # O(nu)

def infEP(hyp, meanfunc, covfunc, likfunc, x, y, nargout=1):
    #function [post nlZ dnlZ] = infEP(hyp, mean, cov, lik, x, y)

    # Expectation Propagation approximation to the posterior Gaussian Process.
    # The function takes a specified covariance function (see covFunction.m) and
    # likelihood function (see likFunction.m), and is designed to be used with
    # gp.m. See also infFunctions.m. In the EP algorithm, the sites are 
    # updated in random order, for better performance when cases are ordered
    # according to the targets.
    #
    # Copyright (c) by Carl Edward Rasmussen and Hannes Nickisch 2010-02-25.
    #
    # See also INFMETHODS.M.

    tol = 1e-4; max_sweep = 10; min_sweep = 2 # tolerance to stop EP iterations

    inffunc = 'inf.infEP'
    n = x.shape[0]
    K = Tools.general.feval(covfunc, hyp.cov, x)    # evaluate the covariance matrix
    m = Tools.general.feval(meanfunc, hyp.mean, x)  # evaluate the mean vector

    # A note on naming: variables are given short but descriptive names in 
    # accordance with Rasmussen & Williams "GPs for Machine Learning" (2006): mu
    # and s2 are mean and variance, nu and tau are natural parameters. A leading t
    # means tilde, a subscript _ni means "not i" (for cavity parameters), or _n
    # for a vector of cavity parameters.

    # marginal likelihood for ttau = tnu = zeros(n,1); equals n*log(2) for likCum*
    nlZ0 = -Tools.general.feval(likfunc, hyp.lik, y, m, np.reshape(np.diag(K),(np.diag(K).shape[0],1)), inffunc).sum()
    if "last_ttau" not in infEP.__dict__:   # find starting point for tilde parameters
        ttau  = np.zeros((n,1))             # initialize to zero if we have no better guess
        tnu   = np.zeros((n,1))
        Sigma = K                           # initialize Sigma and mu, the parameters of ..
        mu    = np.zeros((n,1))             # .. the Gaussian posterior approximation
        nlZ   = nlZ0
    else:
        ttau = infEP.last_ttau              # try the tilde values from previous call
        tnu  = infEP.last_tnu
        [Sigma, mu, nlZ, L] = epComputeParams(K, y, ttau, tnu, likfunc, hyp, m, inffunc)
        if nlZ > nlZ0:                                # if zero is better ..
            ttau = np.zeros((n,1))                    # .. then initialize with zero instead
            tnu  = np.zeros((n,1)) 
            Sigma = K                              # initialize Sigma and mu, the parameters of ..
            mu = np.zeros((n,1))                   # .. the Gaussian posterior approximation
            nlZ = nlZ0
        #end
    #end

    nlZ_old = np.inf; sweep = 0               # converged, max. sweeps or min. sweeps?
    while (np.abs(nlZ-nlZ_old) > tol and sweep < max_sweep) or (sweep < min_sweep):
        nlZ_old = nlZ; sweep += 1
        rperm = range(n)#randperm(n)
        for ii in rperm:       # iterate EP updates (in random order) over examples
            tau_ni = 1/Sigma[ii,ii] - ttau[ii]      #  first find the cavity distribution ..
            nu_ni  = mu[ii]/Sigma[ii,ii] + m[ii]*tau_ni - tnu[ii]    # .. params tau_ni and nu_ni
            # compute the desired derivatives of the indivdual log partition function
            vargout = Tools.general.feval(likfunc, hyp.lik, y[ii], nu_ni/tau_ni, 1/tau_ni, inffunc, None, 3)
            lZ = vargout[0]; dlZ = vargout[1]; d2lZ = vargout[2] 
            ttau_old = copy(ttau[ii])   # then find the new tilde parameters, keep copy of old
    
            ttau[ii] = -d2lZ  /(1.+d2lZ/tau_ni)
            ttau[ii] = max(ttau[ii],0) # enforce positivity i.e. lower bound ttau by zero
            tnu[ii]  = ( dlZ + (m[ii]-nu_ni/tau_ni)*d2lZ )/(1.+d2lZ/tau_ni)
    
            ds2 = ttau[ii] - ttau_old                   # finally rank-1 update Sigma ..
            si  = np.reshape(Sigma[:,ii],(Sigma.shape[0],1))
            Sigma = Sigma - ds2/(1.+ds2*si[ii])*np.dot(si,si.T)   # takes 70# of total time
            mu = np.dot(Sigma,tnu)                                # .. and recompute mu
        #end
        # recompute since repeated rank-one updates can destroy numerical precision
        [Sigma, mu, nlZ, L] = epComputeParams(K, y, ttau, tnu, likfunc, hyp, m, inffunc)
    #end

    if sweep == max_sweep:
        raise Exception('maximum number of sweeps reached in function infEP')
    #end

    infEP.last_ttau = ttau; infEP.last_tnu = tnu      # remember for next call

    sW = np.sqrt(ttau); alpha = tnu-sW*solve_chol(L,sW*np.dot(K,tnu))

    post = postStruct()

    post.alpha = alpha                                # return the posterior params
    post.sW    = sW
    post.L     = L

    if nargout>2:                                           # do we want derivatives?
        dnlZ = deepcopy(hyp)                               # allocate space for derivatives
        ssi  = np.sqrt(ttau)
        V = np.linalg.solve(L.T,np.tile(ssi,(1,n))*K)
        Sigma = K - np.dot(V.T,V)
        mu = np.dot(Sigma,tnu)
        Dsigma = np.reshape(np.diag(Sigma),(np.diag(Sigma).shape[0],1))
        tau_n = 1/Dsigma-ttau                    # compute the log marginal likelihood
        nu_n  = mu/Dsigma-tnu                    # vectors of cavity parameters

        F = np.dot(alpha,alpha.T) - np.tile(sW,(1,n))* \
            solve_chol(L,np.reshape(np.diag(sW),(np.diag(sW).shape[0],1)))   # covariance hypers
        for jj in range(len(hyp.cov)):
            dK = Tools.general.feval(covfunc, hyp.cov, x, None, jj)
            dnlZ.cov[jj] = -(F*dK).sum()/2.
        #end
        for ii in range(len(hyp.lik)):
            dlik = Tools.general.feval(likfunc, hyp.lik, y, nu_n/tau_n+m, 1/tau_n, inffunc, ii)
            dnlZ.lik[ii] = -dlik.sum()
        #end
        [junk,dlZ] = Tools.general.feval(likfunc, hyp.lik, y, nu_n/tau_n+m, 1/tau_n, inffunc) # mean hyps
        for ii in range(len(hyp.mean)):
            dm = Tools.general.feval(meanfunc, hyp.mean, x, ii)
            dnlZ.mean[ii] = -np.dot(dlZ.T,dm)
        #end
        vargout = [post, nlZ, dnlZ]
    else:
        vargout = [post, nlZ]
    #end
    return vargout

# functions to compute the parameters of the Gaussian approximation, Sigma and
# mu, and the negative log marginal likelihood, nlZ, from the current site
# parameters, ttau and tnu. Also returns L (useful for predictions).

def epComputeParams(K, y, ttau, tnu, likfunc, hyp, m, inffunc):
    #function [Sigma mu nlZ L] = epComputeParams(K, y, ttau, tnu, lik, hyp, m, inf)
    n     = len(y)                                                # number of training cases
    ssi   = np.sqrt(ttau)                                         # compute Sigma and mu
    L     = np.linalg.cholesky(np.eye(n)+np.dot(ssi,ssi.T)*K).T   # L'*L=B=eye(n)+sW*K*sW
    V     = np.linalg.solve(L.T,np.tile(ssi,(1,n))*K)
    Sigma = K - np.dot(V.T,V)
    mu    = np.dot(Sigma,tnu)

    Dsigma = np.reshape(np.diag(Sigma),(np.diag(Sigma).shape[0],1)) 

    tau_n = 1/Dsigma - ttau # compute the log marginal likelihood
    nu_n  = mu/Dsigma-tnu + m*tau_n       # vectors of cavity parameters
    lZ    = Tools.general.feval(likfunc, hyp.lik, y, nu_n/tau_n, 1/tau_n, inffunc)

    nlZ   = np.log(np.diag(L)).sum() - lZ.sum() - np.dot(tnu.T,np.dot(Sigma,tnu))/2  \
            - np.dot((nu_n-m*tau_n).T,((ttau/tau_n*(nu_n-m*tau_n)-2*tnu) / (ttau+tau_n)))/2 \
            + (tnu**2/(tau_n+ttau)).sum()/2.- np.log(1.+ttau/tau_n).sum()/2.
    return Sigma, mu, nlZ[0], L

def infExact(hyp, meanfunc, covfunc, likfunc, x, y, nargout=1):
    # Exact inference for a GP with Gaussian likelihood. Compute a parametrization
    # of the posterior, the negative log marginal likelihood and its derivatives
    # w.r.t. the hyperparameters. See also "help infMethods".
    #
    # Copyright (c) by Carl Edward Rasmussen and Hannes Nickisch, 2013-01-21
    #
    # See also INFMETHODS.M.

    if not (likfunc[0] == 'lik.likGauss'):                   # NOTE: no explicit call to likGauss
        raise Exception ('Exact inference only possible with Gaussian likelihood')
    #end
 
    n, D = x.shape
    K = Tools.general.feval(covfunc, hyp.cov, x)           # evaluate covariance matrix
    m = Tools.general.feval(meanfunc, hyp.mean, x)         # evaluate mean vector

    sn2   = np.exp(2.*hyp.lik)                            # noise variance of likGauss
    L     = np.linalg.cholesky(K/sn2+np.eye(n)).T          # Cholesky factor of covariance with noise
    alpha = solve_chol(L,y-m)/sn2

    post = postStruct()

    post.alpha = alpha                                          # return the posterior parameters
    post.sW    = np.ones((n,1))/np.sqrt(sn2)                     # sqrt of noise precision vector
    post.L     = L                                               # L = chol(eye(n)+sW*sW'.*K)

    if nargout>1:                                                # do we want the marginal likelihood?
        nlZ = np.dot((y-m).T,alpha/2) + np.log(np.diag(L)).sum() + n*np.log(2*np.pi*sn2)/2. # -log marg lik
        if nargout>2:                                            # do we want derivatives?
            dnlZ = deepcopy(hyp)                                 # allocate space for derivatives
            Q = solve_chol(L,np.eye(n))/sn2 - np.dot(alpha,alpha.T) # precompute for convenience
            for ii in range(len(hyp.cov)):
                dnlZ.cov[ii] = (Q*Tools.general.feval(covfunc, hyp.cov, x, None, ii)).sum()/2.
            #end
            dnlZ.lik = sn2*np.trace(Q)
            for ii in range(len(hyp.mean)): 
                dnlZ.mean[ii] = np.dot(-Tools.general.feval(meanfunc, hyp.mean, x, ii).T,alpha)
            #end
            return post, nlZ, dnlZ
        #end
        return post, nlZ
    #end
    return post

def infFITC(hyp, meanfunc, covfunc, likfunc, x, y, nargout=1):
    # FITC approximation to the posterior Gaussian process. The function is
    # equivalent to infExact with the covariance function:
    #   Kt = Q + G; G = diag(g); g = diag(K-Q);  Q = Ku'*inv(Quu)*Ku;
    # where Ku and Kuu are covariances w.r.t. to inducing inputs xu, snu2 = sn2/1e6
    # is the noise of the inducing inputs and Quu = Kuu + snu2*eye(nu).
    # We fixed the standard deviation of the inducing inputs snu to be a one per mil
    # of the measurement noise's standard deviation sn.
    # The implementation exploits the Woodbury matrix identity
    #   inv(Kt) = inv(G) - inv(G)*V'*inv(eye(nu)+V*inv(G)*V')*V*inv(G)
    # in order to be applicable to large datasets. The computational complexity
    # is O(n nu^2) where n is the number of data points x and nu the number of
    # inducing inputs in xu.
    # The function takes a specified covariance function (see covFunction.m) and
    # likelihood function (see likFunction.m), and is designed to be used with
    # gp.m and in conjunction with covFITC and likGauss. 
    #
    # Copyright (c) by Ed Snelson, Carl Edward Rasmussen 
    #                                               and Hannes Nickisch, 2012-11-20.
    #
    # See also INFMETHODS.M, COVFITC.M.

    if not (likfunc[0] == 'lik.likGauss'):                   # NOTE: no explicit call to likGauss
        raise Exception ('Exact inference only possible with Gaussian likelihood')
    #end

    cov1 = covfunc[0] 
    if not cov1 == ['kernels.covFITC']:
        raise Exception('Only covFITC supported.') # check cov
    #end

    diagK,Kuu,Ku = Tools.general.feval(covfunc, hyp.cov, x)    # evaluate covariance matrix
    m            = Tools.general.feval(meanfunc, hyp.mean, x)  # evaluate mean vector
    n, D = x.shape; nu = Kuu.shape[0]

    sn2   = np.exp(2*hyp.lik)                              # noise variance of likGauss
    snu2  = 1.e-6*sn2                              # hard coded inducing inputs noise
    Luu   = np.linalg.cholesky(Kuu+snu2*np.eye(nu)).T  # Kuu + snu2*I = Luu'*Luu
    V     = np.linalg.solve(Luu.T,Ku)               # V = inv(Luu')*Ku => V'*V = Q
    g_sn2 = diagK + sn2 - np.array([(V*V).sum(axis=0)]).T # g + sn2 = diag(K) + sn2 - diag(Q)
    Lu    = np.linalg.cholesky(np.eye(nu) + np.dot(V/np.tile(g_sn2.T,(nu,1)),V.T)).T  # Lu'*Lu=I+V*diag(1/g_sn2)*V'
    r     = (y-m)/np.sqrt(g_sn2)
    be    = np.linalg.solve(Lu.T,np.dot(V,r/np.sqrt(g_sn2)))
    iKuu  = solve_chol(Luu,np.eye(nu))              # inv(Kuu + snu2*I) = iKuu

    post = postStruct()

    post.alpha = np.linalg.solve(Luu,np.linalg.solve(Lu,be)) # return the posterior parameters
    post.L  = solve_chol(np.dot(Lu,Luu),np.eye(nu)) - iKuu   # Sigma-inv(Kuu)
    post.sW = np.ones((n,1))/np.sqrt(sn2)                   # unused for FITC prediction  with gp.m

    if nargout>1:                                # do we want the marginal likelihood
        nlZ = np.log(np.diag(Lu)).sum() + np.log(g_sn2).sum() + n*np.log(2*np.pi) + np.dot(r.T,r) - np.dot(be.T,be)/2. 
        if nargout>2:                                    # do we want derivatives?
            dnlZ = deepcopy(hyp)                        # allocate space for derivatives
            al = r/np.sqrt(g_sn2) - np.dot(V.T,np.linalg.solve(Lu,be))/g_sn2 # al = (Kt+sn2*eye(n))\y
            B = np.dot(iKuu,Ku); w = np.dot(B,al)
            W = np.linalg.solve(Lu.T,V/np.tile(g_sn2.T,(nu,1)))
            for ii in range(len(hyp.cov)):
                [ddiagKi,dKuui,dKui] = feval(covfunc, hyp.cov, x, None, ii)  # eval cov deriv
                R = 2.*dKui-np.dot(dKuui,B); v = ddiagKi - np.array([(R*B).sum(axis=0)]).T # diag part of cov deriv
                dnlZ.cov[ii] = ( np.dot(ddiagKi.T,1./g_sn2) + np.dot(w.T,(np.dot(dKuui,w)-2.*np.dot(dKui,al)) - np.dot(al.T,(v*al)) \
                                 - np.array([(W*W).sum(axis=0)])*v - (np.dot(R,W.T)*np.dot(B,W.T)).sum()) )/2.
            #end  
            dnlZ.lik = sn2*((1./g_sn2).sum() - (np.array([(W*W).sum(axis=0)])).sum() - np.dot(al.T,al))
            # since snu2 is a fixed fraction of sn2, there is a covariance-like term in
            # the derivative as well
            dKuui = 2*snu2; R = -np.dot(dKuui,B); v = -np.array([(R*B).sum(axis=0)]).T # diag part of cov deriv
            dnlZ.lik += (np.dot(w.T,np.dot(dKuui,w)) -np.dot(al.T,(v*al)) \
                                 - np.array([(W*W).sum(axis=0)])*v - (np.dot(R,W.T)*np.dot(B,W.T)).sum() )/2. 
            for ii in range(len(hyp.mean)):
                dnlZ.mean[ii] = np.dot(-Tools.general.feval(meanfunc, hyp.mean, x, ii).T,*al)
            #end
        return post,nlZ,dnlZ
        #end
        return post,nlZ
    #end
    return post

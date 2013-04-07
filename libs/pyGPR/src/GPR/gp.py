import numpy as np
import Tools.general
from UTIL import solve_chol
from UTIL.utils import checkParameters, unique

def gp(hyp, inffunc, meanfunc, covfunc, likfunc, x, y, xs=None, ys=None, der=None):
    # Gaussian Process inference and prediction. The gp function provides a
    # flexible framework for Bayesian inference and prediction with Gaussian
    # processes for scalar targets, i.e. both regression and binary
    # classification. The prior is Gaussian process, defined through specification
    # of its mean and covariance function. The likelihood function is also
    # specified. Both the prior and the likelihood may have hyperparameters
    # associated with them.
    #
    # Two modes are possible: training or prediction: if no test cases are
    # supplied, then the negative log marginal likelihood and its partial
    # derivatives w.r.t. the hyperparameters is computed; this mode is used to fit
    # the hyperparameters. If test cases are given, then the test set predictive
    # probabilities are returned. Usage:
    #
    #   training: [nlZ dnlZ          ] = gp(hyp, inf, mean, cov, lik, x, y, None, None, der);
    # prediction: [ymu ys2 fmu fs2   ] = gp(hyp, inf, mean, cov, lik, x, y, xs, None, None, None);
    #         or: [ymu ys2 fmu fs2 lp] = gp(hyp, inf, mean, cov, lik, x, y, xs, ys, None);
    #
    # where:
    #
    #   hyp          column vector of hyperparameters
    #   inffunc      function specifying the inference method 
    #   covfunc      prior covariance function (see below)
    #   meanfunc     prior mean function
    #   likfunc      likelihood function
    #   x            n by D matrix of training inputs
    #   y            column vector of length n of training targets
    #   xs           ns by D matrix of test inputs
    #   ys           column vector of length nn of test targets
    #   der          flag for dnlZ computation determination (when xs == None also)
    #
    #   nlZ          returned value of the negative log marginal likelihood
    #   dnlZ         column vector of partial derivatives of the negative
    #                    log marginal likelihood w.r.t. each hyperparameter
    #   ymu          column vector (of length ns) of predictive output means
    #   ys2          column vector (of length ns) of predictive output variances
    #   fmu          column vector (of length ns) of predictive latent means
    #   fs2          column vector (of length ns) of predictive latent variances
    #   lp           column vector (of length ns) of log predictive probabilities
    #
    #   post         struct representation of the (approximate) posterior
    #                3rd output in training mode and 6th output in prediction mode
    # 
    # See also covFunctions.m, infMethods.m, likFunctions.m, meanFunctions.m.
    #
    # Copyright (c) by Carl Edward Rasmussen and Hannes Nickisch, 2011-02-18

    if not inffunc:
        inffunc = ['inf.infExact']                           # set default inf
    if not meanfunc:
        meanfunc = ['means.meanZero']                     # set default mean
    if not covfunc:
        raise Exception('Covariance function cannot be empty') # no default covariance

    if covfunc[0] == 'kernels.covFITC':
        inffunc = ['inf.infFITC']                     # only one possible inference alg for covFITC
    if not likfunc:
        likfunc = ['lik.likGauss']                # set default lik

    D = np.shape(x)[1]

    if not checkParameters(meanfunc,hyp.mean,D):
        raise Exception('Number of mean function hyperparameters disagree with mean function')
    if not checkParameters(covfunc,hyp.cov,D):
        raise Exception('Number of cov function hyperparameters disagree with cov function')
    if not checkParameters(likfunc,hyp.lik,D):
        raise Exception('Number of lik function hyperparameters disagree with lik function')

    try:                                         # call the inference method
        # issue a warning if a classification likelihood is used in conjunction with
        # labels different from +1 and -1
        if likfunc[0] == ['lik.likErf'] or likfunc[0] == ['lik.likLogistic']:
            uy = unique(y)
            ind = ( uy != 1 )
            if any( uy[ind] != -1):
                raise Exception('You attempt classification using labels different from {+1,-1}\n')
            #end
        #end
        if not xs == None:   # compute marginal likelihood and its derivatives only if needed
            vargout = Tools.general.feval(inffunc,hyp, meanfunc, covfunc, likfunc, x, y, 1)
            post = vargout[0]
        else:
            if not der:
                vargout = Tools.general.feval(inffunc, hyp, meanfunc, covfunc, likfunc, x, y, 2)
                post = vargout[0]; nlZ = vargout[1] 
            else:
                vargout = Tools.general.feval(inffunc, hyp, meanfunc, covfunc, likfunc, x, y, 3)
                post = vargout[0]; nlZ = vargout[1]; dnlZ = vargout[2] 
            #end
        #end
    except Exception, e:
        raise Exception('Inference method failed ' + str(e) + '\n') 
    #end

    if xs == None:                           # if no test cases are provided
        if not der:
            varargout = [nlZ, post]          # report -log marg lik, derivatives and post
        else:
            varargout = [nlZ, dnlZ, post]    # report -log marg lik, derivatives and post
    else:
        alpha = post.alpha
        L     = post.L
        sW    = post.sW
        #if issparse(alpha)                  # handle things for sparse representations
        #    nz = alpha != 0                 # determine nonzero indices
        #    if issparse(L), L = full(L(nz,nz)); end      # convert L and sW if necessary
        #    if issparse(sW), sW = full(sW(nz)); end
        #else:
        nz = range(len(alpha[:,0]))      # non-sparse representation 
        if L == []:                      # in case L is not provided, we compute it
            K = Tools.general.feval(covfunc, hyp.cov, x[nz,:])
            L = np.linalg.cholesky( (np.eye(nz) + np.dot(sW,sW.T)*K).T )
        #end
        Ltril     = np.all( np.tril(L,-1) == 0 ) # is L an upper triangular matrix?
        ns        = xs.shape[0]                  # number of data points
        nperbatch = 1000                         # number of data points per mini batch
        nact      = 0                            # number of already processed test data points
        ymu = np.zeros((ns,1)); ys2 = np.zeros((ns,1))
        fmu = np.zeros((ns,1)); fs2 = np.zeros((ns,1)); lp  = np.zeros((ns,1))   

        while nact<ns-1:                           # process minibatches of test cases to save memory
            id  = range(nact,min(nact+nperbatch,ns))               # data points to process
            kss = Tools.general.feval(covfunc, hyp.cov, xs[id,:], 'diag')     # self-variances
            Ks  = Tools.general.feval(covfunc, hyp.cov, x[nz,:], xs[id,:])    # cross-covariances
            ms  = Tools.general.feval(meanfunc, hyp.mean, xs[id,:])
            N = (alpha.shape)[1]     # number of alphas (usually 1; more in case of sampling)
            Fmu = np.tile(ms,(1,N)) + np.dot(Ks.T,alpha[nz])         # conditional mean fs|f
            fmu[id] = np.reshape(Fmu.sum(axis=1)/N,(len(id),1))       # predictive means
            #fmu[id] = ms + np.dot(Ks.T,alpha[nz])         # conditional mean fs|f
            #
            if Ltril: # L is triangular => use Cholesky parameters (alpha,sW,L)
                V       = np.linalg.solve(L.T,np.tile(sW,(1,len(id)))*Ks)
                fs2[id] = kss - np.array([(V*V).sum(axis=0)]).T           # predictive variances
            else:     # L is not triangular => use alternative parametrization
                fs2[id] = kss + np.array([(Ks*np.dot(L,Ks)).sum(axis=0)]).T # predictive variances
            #end
            fs2[id] = np.maximum(fs2[id],0)         # remove numerical noise i.e. negative variances
            Fs2 = np.tile(fs2[id],(1,N))            # we have multiple values in case of sampling
            if ys == None:
                [Lp, Ymu, Ys2] = Tools.general.feval(likfunc,hyp.lik,None,Fmu[:],Fs2[:],None,None,3)
            else:
                [Lp, Ymu, Ys2] = Tools.general.feval(likfunc,hyp.lik, np.tile(ys[id],(1,N)), Fmu[:], Fs2[:],None,None,3)
            #end
            lp[id]  = np.reshape( np.reshape(Lp,(np.prod(Lp.shape),N)).sum(axis=1)/N , (len(id),1) )   # log probability; sample averaging
            ymu[id] = np.reshape( np.reshape(Ymu,(np.prod(Ymu.shape),N)).sum(axis=1)/N ,(len(id),1) )  # predictive mean ys|y and ...
            ys2[id] = np.reshape( np.reshape(Ys2,(np.prod(Ys2.shape),N)).sum(axis=1)/N , (len(id),1) ) # .. variance
            nact = id[-1]          # set counter to index of last processed data point
        #end
       
        if ys == None:
            varargout = [ymu, ys2, fmu, fs2, None, post]        # assign output arguments
        else:
            varargout = [ymu, ys2, fmu, fs2, lp, post]
        #end

    return varargout

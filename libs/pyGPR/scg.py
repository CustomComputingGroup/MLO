from math import sqrt
import numpy as np

#function [x, options, flog, pointlog, scalelog] = scg(f, x, options, gradf, varargin)
def scg(x, f, gradf, args, niters = 100, gradcheck = False, display = 0, flog = False, pointlog = False, scalelog = False, tolX = 1.0e-8, tolO = 1.0e-8, eval = None):
#SCG  Scaled conjugate gradient optimization.
#
#     Description
#     [X, OPTIONS] = SCG(F, X, OPTIONS, GRADF) uses a scaled conjugate
#     gradients algorithm to find a local minimum of the function F(X)
#     whose gradient is given by GRADF(X).  Here X is a row vector and F
#     returns a scalar value. The point at which F has a local minimum is
#     returned as X.  The function value at that point is returned in
#     OPTIONS(8).
#
#     [X, OPTIONS, FLOG, POINTLOG, SCALELOG] = SCG(F, X, OPTIONS, GRADF)
#     also returns (optionally) a log of the function values after each
#     cycle in FLOG, a log of the points visited in POINTLOG, and a log of
#     the scale values in the algorithm in SCALELOG.
#
#     SCG(F, X, OPTIONS, GRADF, P1, P2, ...) allows additional arguments to
#     be passed to F() and GRADF().     The optional parameters have the
#     following interpretations.
#
#     OPTIONS(1) = display is set to 1 to display error values; also logs error
#     values in the return argument ERRLOG, and the points visited in the
#     return argument POINTSLOG.  If OPTIONS(1) is set to 0, then only
#     warning messages are displayed.  If OPTIONS(1) is -1, then nothing is
#     displayed.
#
#     OPTIONS(2)= tolX is a measure of the absolute precision required for the
#     value of X at the solution.  If the absolute difference between the
#     values of X between two successive steps is less than OPTIONS(2),
#     then this condition is satisfied.
#
#     OPTIONS(3)= tolO is a measure of the precision required of the objective
#     fuction at the solution.  If the absolute difference between the
#     objective function values between two successive steps is less than
#     OPTIONS(3), then this condition is satisfied. Both this and the
#     previous condition must be satisfied for termination.
#
#     OPTIONS(9) is set to 1 to check the user defined gradient function.
#
#     OPTIONS(10) = funcCount returns the total number of function evaluations
#     (including those in any line searches).
#
#     OPTIONS(11)= gradCount returns the total number of gradient evaluations.
#
#     OPTIONS(14) = niters is the maximum number of iterations; default 100.
#
#     See also
#     CONJGRAD, QUASINEW
#
#
#     Copyright (c) Ian T Nabney (1996-2001)
 
#  Set up the options.

    # Set up strings for evaluating function and gradient
#    f = fcnchk(f, length(varargin));
#    gradf = fcnchk(gradf, length(varargin));
     
    if display: print '\n***** starting optimization (SCG) *****\n'
    nparams = len(x);
     
    #  Check gradients
    if (gradcheck):
        pass
#        feval('gradchek', x, f, gradf, varargin{:});
     
    eps = 1.0e-4
    sigma0 = 1.0e-4;
    fold = f(x, *args);            # Initial function value.
    fnow = fold;
    funcCount = 1;                # Increment function evaluation counter.
    gradnew = gradf(x, *args)      # Initial gradient.
    gradold = gradnew;
    gradCount = 1;                # Increment gradient evaluation counter.
    d = -gradnew;                  # Initial search direction.
    success = 1;                   # Force calculation of directional derivs.
    nsuccess = 0;                  # nsuccess counts number of successes.
    beta = 1.0;                    # Initial scale parameter.
    betamin = 1.0e-15;             # Lower bound on scale.
    betamax = 1.0e50;              # Upper bound on scale.
    j = 1;                         # j counts number of iterations.

    if flog:
        pass
        #flog(j, :) = fold;
    if pointlog:
        pass
        #pointlog(j, :) = x;
     
    # Main optimization loop.
    
    listF = [fold]
    if eval is not None:
        evalue, timevalue = eval(x, *args)
        evalList = [evalue]
        time = [timevalue]

    while (j <= niters):
         
        # Calculate first and second directional derivatives.
        if (success == 1):
            mu = np.dot(d, gradnew)
            if (mu >= 0):
                d = - gradnew
                mu = np.dot(d, gradnew)

            kappa = np.dot(d, d)
            if (kappa < eps):
                print "FNEW: " , fnow
                #options(8) = fnow
                if eval is not None:
                    return x, listF, evalList, time
                else:
                    return x, listF

            sigma = sigma0/sqrt(kappa)
            xplus = x + sigma*d
            gplus = gradf(xplus, *args)
            gradCount += 1
            theta = (np.dot(d, (gplus - gradnew)))/sigma;
     
        # Increase effective curvature and evaluate step size alpha.
        delta = theta + beta*kappa
        if (delta <= 0):
            delta = beta*kappa
            beta = beta - theta/kappa

        alpha = - mu/delta
         
        # Calculate the comparison ratio.
        xnew = x + alpha*d
        fnew = f(xnew, *args)
        funcCount += 1;
        Delta = 2*(fnew - fold)/(alpha*mu)
        if (Delta  >= 0):
            success = 1;
            nsuccess += 1;
            x = xnew;
            fnow = fnew;
            listF.append(fnow)
            if eval is not None:
                evalue, timevalue = eval(x, *args)
                evalList.append(evalue)
                time.append(timevalue)
                
        else:
            success = 0;
            fnow = fold;
     
        if flog:
            # Store relevant variables
            #flog(j) = fnow;          # Current function value
            pass
        if pointlog:
            #pointlog(j,:) = x;     # Current position
            pass
        if scalelog:
            #scalelog(j) = beta;     # Current scale parameter
            pass
        if display > 0:
            print('***** Cycle %4d  Error %11.6f  Scale %e' %( j, fnow, beta))
     
        if (success == 1):
        # Test for termination
#            print type (alpha), type(d), type(tolX), type(fnew), type(fold)

            if ((max(abs(alpha*d)) < tolX) & (abs(fnew-fold) < tolO)):
                #options(8) = fnew;
#                print "FNEW: " , fnew
                if eval is not None:
                    return x, listF, evalList, time
                else:
                    return x, listF
            else:
                # Update variables for new position
                fold = fnew
                gradold = gradnew
                gradnew = gradf(x, *args)
                gradCount += 1
                # If the gradient is zero then we are done.
                if (np.dot(gradnew, gradnew) == 0):
#                    print "FNEW: " , fnew
                    #options(8) = fnew;
                    if eval is not None:
                        return x, listF, evalList, time
                    else:
                        return x, listF
     
        # Adjust beta according to comparison ratio.
        if (Delta < 0.25):
            beta = min(4.0*beta, betamax);
        if (Delta > 0.75):
            beta = max(0.5*beta, betamin);
     
        # Update search direction using Polak-Ribiere formula, or re-start
        # in direction of negative gradient after nparams steps.
        if (nsuccess == nparams):
            d = -gradnew;
            nsuccess = 0;
        else:
            if (success == 1):
                gamma = np.dot((gradold - gradnew), gradnew)/(mu)
                d = gamma*d - gradnew;

        j += 1
     
    # If we get here, then we haven't terminated in the given number of
    # iterations.
    #options(8) = fold;
    if (display):
        print "maximum number of iterations reached"
    if eval is not None:
        return x, listF, evalList, time
    else:
        return x, listF

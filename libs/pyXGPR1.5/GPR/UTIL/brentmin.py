import sys
from math import sqrt

def brentmin(xlow,xupp,Nitmax,tol,f,nout=None,*args):
    #def brentmin(0,smax,Nline,thr,'Psi_line',4,dalpha,alpha,hyp,K,m,likfunc,y,inffunc):
    ## BRENTMIN: Brent's minimization method in one dimension
    # code taken from
    #    Section 10.2 Parabolic Interpolation and Brent's Method in One Dimension
    #    Press, Teukolsky, Vetterling & Flannery
    #    Numerical Recipes in C, Cambridge University Press, 2002
    #
    # [xmin,fmin,funccout,varargout] = BRENTMIN(xlow,xupp,Nit,tol,f,nout,varargin)
    #    Given a function f, and given a search interval this routine isolates 
    #    the minimum of fractional precision of about tol using Brent's method.
    # 
    # INPUT
    # -----
    # xlow,xupp:  search interval such that xlow<=xmin<=xupp
    # Nitmax:     maximum number of function evaluations made by the routine
    # tol:        fractional precision 
    # f:          [y,varargout{:}] = f(x,varargin{:}) is the function
    # nout:       no. of outputs of f (in varargout) in addition to the y value
    #
    # OUTPUT
    # ------
    # fmin:      minimal function value
    # xmin:      corresponding abscissa-value
    # funccount: number of function evaluations made
    # varargout: additional outputs of f at optimum
    #
    # Copyright (c) by Hannes Nickisch 2010-01-10.

    if nout == None:
        nout = 0
    eps = sys.float_info.epsilon

    # tolerance is no smaller than machine's floating point precision
    tol = max(tol,eps)

    # Evaluate endpoints
    vargout = f(xlow,*args); fa = vargout[0][0]
    vargout = f(xupp,*args); fb = vargout[0][0]
    funccount = 2; # number of function evaluations
    # Compute the start point
    seps = sqrt(eps);
    c = 0.5*(3.0 - sqrt(5.0)) # golden ratio
    a = xlow; b = xupp;
    v = a + c*(b-a)
    w = v; xf = v
    d = 0.; e = 0.
    x = xf; vargout = f(x,*args); fx = vargout[0][0]; varargout = vargout[1:]
    funccount += 1

    fv = fx; fw = fx
    xm = 0.5*(a+b);
    tol1 = seps*abs(xf) + tol/3.0;
    tol2 = 2.0*tol1;

    # Main loop
    while ( abs(xf-xm) > (tol2 - 0.5*(b-a)) ):
        gs = True
        # Is a parabolic fit possible
        if abs(e) > tol1:
            # Yes, so fit parabola
            gs = False
            r = (xf-w)*(fx-fv)
            q = (xf-v)*(fx-fw)
            p = (xf-v)*q-(xf-w)*r
            q = 2.0*(q-r)
            if q > 0.0:  
                p = -p
            q = abs(q)
            r = e;  e = d

            # Is the parabola acceptable
            if ( (abs(p)<abs(0.5*q*r)) and (p>q*(a-xf)) and (p<q*(b-xf)) ):
                # Yes, parabolic interpolation step
                d = p/q
                x = xf+d
                # f must not be evaluated too close to ax or bx
                if ((x-a) < tol2) or ((b-x) < tol2):
                    si = cmp(xm-xf,0)
                    if ((xm-xf) == 0): si += 1
                    d = tol1*si
                #end
            else:
                # Not acceptable, must do a golden section step
                gs = True
            #end
        #end
        if gs:
            # A golden-section step is required
            if xf >= xm: e = a-xf    
            else: 
                e = b-xf
            d = c*e
        #end

        # The function must not be evaluated too close to xf
        si = cmp(d,0)
        if (d == 0): si += 1
        x = xf + si * max(abs(d),tol1)
        vargout = f(x,*args); fu = vargout[0][0]; varargout = vargout[1:]
        funccount += 1

        # Update a, b, v, w, x, xm, tol1, tol2
        if fu <= fx:
            if x >= xf: a = xf 
            else: b = xf
            v = w; fv = fw
            w = xf; fw = fx
            xf = x; fx = fu
        else: # fu > fx
            if x < xf: 
                a = x
            else: 
                b = x 
            if ( (fu <= fw) or (w == xf) ):
                v = w; fv = fw
                w = x; fw = fu
            elif ( (fu <= fv) or ((v == xf) or (v == w)) ):
                v = x; fv = fu
        #end
        xm = 0.5*(a+b)
        tol1 = seps*abs(xf) + tol/3.0; tol2 = 2.0*tol1

        if funccount >= Nitmax:        
            # typically we should not get here
            raise Exception('Maximum number of iterations exceeded in brentmin')
    #end # while

    # check that endpoints are less than the minimum found
    if ( (fa < fx) and (fa <= fb) ):
        xf = xlow; fx = fa
    elif fb < fx:
        xf = xupp; fx = fb
    #end
    fmin = fx
    xmin = xf
    vargout = [xmin,fmin,funccount]
    for vv in varargout:
        vargout.append(vv)
    return vargout

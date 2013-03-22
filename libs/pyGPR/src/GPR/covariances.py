# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 10:29:06 2013

@author: Hannes Nickisch, Philips Research Hamburg
"""

import numpy as np

# output number of parameters
# 1) full covariance matrix
# 2) cross covariance matrix
# 3) diagonal of covariance matrix
# 4) derivatives of 1-3)

class Covariance:
    """ Covariance function base class.
    """
    def __init__(self,*args):
        self.name = 'base'
        self.diag  = False
        self.deriv = False
    
    def __str__(self):
        return "covariance "+self.name
    
    def __add__(self, other):
        """ covSum and covNoise
        """
        print "add"
        return Covariance('add')

    def __radd__(self, other):
        """ covNoise
        """
        print "radd"
        return Covariance('radd')
    
    def __mul__(self, other):
        """ covProd and covScale
        """
        print "mul"
        return Covariance('mul')

    def __rmul__(self, other):
        """ covScale
        """
        print "rmul"
        return Covariance('rmul')

    def __getitem__(self, modifier):
        """ Apply derivative or diagonal modifier.
        """
        if modifier=='diag':
            self.diag = True
        elif modifier=='d' or modifier=='deriv':
            self.deriv = True
        return self

    def __call__(self, hyp, x, z=None):
        if self.deriv:
            if self.diag:
                print "diag-d-eval"
            else:
                print "ndiag-d-eval"
        else:
            if self.diag:
                print "diag-eval"
            else:
                print "ndiag-eval"
        return 3

class covSEiso(Covariance):
    pass

class covSEard(Covariance):
    pass

if __name__ == "__main__":
    k1 = Covariance()
    k2 = Covariance()
    k3 = covSEiso()
    k4 = covSEard()
    k5 = Covariance()

    k = 2*k1*k2*4*k3 + k4 + k5 + 1
    print k

    n,d = 87,4
    hyp = None
    x = np.random.rand(n,d)
    K = k(hyp,x)
    dK = k['deriv'](hyp,x)
    diagK = k['diag'](hyp,x)

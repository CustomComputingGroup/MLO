import numpy as np
from solve_chol import solve_chol
import Tools
from copy import copy,deepcopy

from random import sample

def foo():
    foo.counter += 1
    print "Counter is %d" % foo.counter

def randperm(k):
    # return a random permutation of range(k)
    z = range(k)
    y = []
    ii = 0
    while z and ii < 2*k:
        n = sample(z,1)[0]
        y.append(n)
        z.remove(n)
        ii += 1    
    return y

def flatten(l, ltypes=(list, tuple)):
    ltype = type(l)
    l = list(l)
    i = 0
    while i < len(l):
        while isinstance(l[i], ltypes):
            if not l[i]:
                l.pop(i)
                i -= 1
                break
            else:
                l[i:i + 1] = l[i]
        i += 1
    return ltype(l)

def convert_to_array(hyp):
    y = np.concatenate((np.concatenate((hyp.mean, hyp.cov),axis=0),hyp.lik),axis=0)
    return y

def convert_to_class(x,hyp):
    y = deepcopy(hyp)
    Lm = len(hyp.mean)
    Lc = len(hyp.cov)
    y.mean = x[:Lm]
    y.cov  = x[Lm:(Lm+Lc)]
    y.lik  = x[(Lm+Lc):]
    return y

def tempfunc(x,y=None,z=0):
    print x,y,z
    '''if x == 1 and z>0:
        return 6,None,None
    elif x ==2 and z>0:
        return 6,5,None
    elif z>0:
        return 6,5,4
    else:
        return "z is not active"
    '''
def unique(x):
    # First flatten x
    y = [item for sublist in x for item in sublist]
    if isinstance(x,np.ndarray):
        n,D = x.shape
        assert(D == 1)
        y = np.array( list(set(x[:,0])) )
        y = np.reshape(y, (len(y),1))
    else:
        y = list(set(y))    
    return y

class hyperParameters:
    def __init__(self):
        self.mean = []
        self.cov  = []
        self.lik  = []

if __name__ == '__main__':
    hyp = hyperParameters()
    hyp.mean = [3.3,3.1]
    hyp.cov = [0.5]

    #x = np.array([2,3,4,2])
    #y = [2,3,4,2]
    #print np.shape(x)[0]

    #print type(x), type(unique(x))
    #print type(y), type(unique(y))

    #x = np.random.rand(10,1)
    #x[x < 0.5] = 1
    #x[x > 0] = 1
    #ux = unique(x)
    #ind = ( ux != 1 )
    #if any( ux[ind] != -1):
    #    print 'You attempt classification using labels different from {+1,-1}\n'
    
    #x = np.random.rand(10,1)
    #tempfunc(1,None,3)
    #tempfunc(1,None)
    n = 5
    x = np.random.rand(n,n)
    X = np.dot(x,x.T)
    L = np.linalg.cholesky(X).T
    #Ltril = np.all( np.tril(L,-1) == 0 )
    #print Ltril
    #M = solve_chol(L,np.eye(n))
    #print np.dot(M,X)
    #print np.dot(X,M)
    #y = convert_to_array(hyp)
    #print y
    #z = list(y)
    #z = convert_to_class(y,hyp)
    #print z.mean, z.cov, z.lik
    #x = z.pop(0)
    #print x,z
    #z.insert(0,x)
    #a = tuple(z)
    #b = list(a)
    #b[0] = -300
    #print "a = ",a
    #y = np.random.random((5,1))
    #nz = range(len(y[:,0]))
    #Fmu1 = np.dot(X.T,y[nz,:])
    #Fmu  = np.dot(X.T,y)
    #print np.allclose(Fmu1,Fmu)
    #y = np.reshape(x,(5,))
    #A = [3,4,5]
    #print A, flatten(A)
    #A = [3,[4,5]]
    #print A, flatten(A)
    #A = [3,[4,5],[3,[4,5],[3,4,5]]]
    #print A, flatten(A)
    #k = 10
    #y = randperm(k)
    #print y
    #foo.counter = 0
    #foo()
    #t1,t2 = np.meshgrid(np.arange(-4,4.1,0.1),np.arange(-4,4.1,0.1))
    #t = [t1(:) t2(:)]; n = len(t)
    #y = np.array( zip( np.reshape(t1,(np.prod(t1.shape),)), np.reshape(t2,(np.prod(t2.shape),)) ) )
    #print y[0,:]
    y = np.random.random((4,1))
    a = copy(y[0])
    print a
    y[0] = 1.2
    print a,1.2
    del a
    del y

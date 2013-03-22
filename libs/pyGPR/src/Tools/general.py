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
'''
Created on 08.09.2009

@author: Marion Neumann (last update 08/01/10)

Substantial updates by Daniel Marthaler July 2012.
get_nb_param() added by Marion Neumann (Aug 2012).
'''
import numpy as np
from GPR.MEAN import means
from GPR.COV import kernels
from GPR.LIK import lik
from GPR.INF import inf

def feval(funcName, *args):    
    assert(isinstance(funcName, list))
    if len(funcName) > 1:
        if funcName[0] == ['kernels.covFITC']:
            if not args:
                return feval(funcName[0],funcName[1],funcName[2])[0]
            else:
                return feval(funcName[0],funcName[1],funcName[2], *args)
        # This is a composition
        assert(len(funcName) == 2)
        z = funcName[0]

        # Split off the module name (before the period)
        mod,fun = z[0].split('.')

        if mod == 'kernels':
            return getattr(kernels,fun)(funcName[1],*args)
        elif mod == 'means':
            return getattr(means,fun)(funcName[1],*args)
        elif mod == 'lik':
            return getattr(lik,fun)(funcName[1],*args)
        elif mod == 'inf':
            return getattr(inf,fun)(*args)
        else:
            raise Exception("Error in parameter function type")
    else:
        # This is either a singleton call or a call of one of the composed covariances
        z = funcName[0]
        if isinstance(z,list):
            # This is a singleton call
            mod,fun = z[0].split('.')
        else:
            # This is one called from a composition
            # Call the function, split off the module name (before the period)
            mod,fun = z.split('.')
    if mod == 'kernels':
        return getattr(kernels,fun)(*args)
    elif mod == 'means':
        return getattr(means,fun)(*args)
    elif mod == 'lik':
        return getattr(lik,fun)(*args)
    elif mod == 'inf':
        return getattr(inf,fun)(*args)
    else:
        raise Exception("Error in parameter function type")

def convert_single(v,D):
    if isinstance(v,str):
        pram_str = v.split(' ')
        if pram_str[0]=='D':    temp = int(D)
        if pram_str[1]=='+':    temp += int(pram_str[2])
        elif pram_str[1]=='-':  temp -= int(pram_str[2])
        else:
            raise Exception("ERROR: string representation is incorrect for this covariance")
    else:
        temp = v
    return temp

def convert(v,D):
    if isinstance(v,list):
        w = []
        for ii in range(len(v)):
            w.append(convert_single(v[ii],D))
    else:
        w = convert_single(v,D)
    return w

def check_hyperparameters(gp,x):
    check_hyperparameters_func(gp['covfunc'],gp['covtheta'],x)
    check_hyperparameters_func(gp['meanfunc'],gp['meantheta'],x,meanFlag=True) 
    return True

def check_hyperparameters_func(func,logtheta,x,meanFlag=False):
    [n,D] = x.shape
    if not meanFlag:
        if func[0][0] == 'kernels.covFITC':
            # This is an FITC approximation
            func = func[1:-1]
            xu   = func[-1]
            if len(func) == 1:
                func = func[0]
    ## CHECK (hyper)parameters and mean/covariance function(s)
    if len(func) > 1:
        v = flatten(feval(func))
        try:
            assert( sum(convert(v,D)) - len(logtheta) == 0 )
        except AssertionError:
            if meanFlag:
                raise Exception('ERROR: number of hyperparameters does not match given mean function:' + str(sum(convert(v,D))) + ' hyperparameters needed (' + str(len(logtheta)) + ' given )!')
            else:
                raise Exception('ERROR: number of hyperparameters does not match given covariance function:'+ str(sum(convert(v,D))) + ' hyperparameters needed (' + str(len(logtheta)) + ' given )!')
    else:
        try:
            v = feval(func)
            assert( convert(v,D) - len(logtheta) == 0)
        except AssertionError:
            if meanFlag:
                raise Exception("ERROR: number of hyperparameters does not match given mean function: "+ str(convert(v,D)) + " hyperparameters needed ("+ str(len(logtheta)) + " given )!")
            else:
                raise Exception("ERROR: number of hyperparameters does not match given covariance function: "+ str(convert(v,D)) + " hyperparameters needed ("+ str(len(logtheta)) + " given )!")
    return True

def get_nb_param(covList, dim):
    num_hyp = 0
    for i in covList:
        if type(i) == int:
            num_hyp += i
        elif type(i) == str:
            add = True
            str2num = 0
            str_list = i.split(' ')
            for j in range(0,len(str_list)):
                if str_list[j] == '+':
                    add = True
                elif str_list[j] == '-':
                    add = False
                else: 
                    try:
                        current_number = int(str_list[j])
                    except ValueError:
                        if str_list[j] == 'D':
                            current_number = dim[1]
                        elif str_list[j] == '+':
                            add = True
                        else:
                            raise Exception( "ERROR: Not able to identify number of required hyperparameters!")
                            return
                    if add:
                        str2num += current_number
                    else: 
                        str2num -= current_number
            num_hyp += str2num
    return num_hyp

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

def get_index_vec(a,b):
    ''' 
    Returns vector of indices of b for the entries in a.
    Example:
    
         1        2                2
    a =  2    b = 3    index_vec = 0
         3        1                1
    
    Returns '-1' if entry in a is not an entry in b. 
    '''
    a = list(a)
    b = list(b)
    index_vec = []
    for i in range(len(a)):
        try:
            index_vec.append( b.index(a[i]) )
        except ValueError:
            index_vec.append(-1)
    return np.array(index_vec)    

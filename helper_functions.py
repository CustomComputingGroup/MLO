import logging
from numpy import *

def numpy_array_index(multi_array, array):
    #TODO - check if multi_array is non empty and if they match size.. throw appropariate warnings
    if multi_array is not None:
        for i,trainp in enumerate(multi_array):
                if array_equal(trainp,array):
                    return True, i
                    
    return False, 0
     
#TODO - helper functions booleandtrue... without this we might run into some problems soon
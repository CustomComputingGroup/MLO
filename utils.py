from imp import load_source
import logging
import sys
from numpy import array_equal

def load_script(filename, script_type):
    """
    Loads a fitness or configuration script. script_type is either
    'fitness' or 'configuration'.
    """
    try:
        return load_source(script_type, filename)
    except:
        logging.error('{} file ({}) could not be loaded'.format(
            script_type.capitalize(), filename), exc_info=sys.exc_info())
        return None

def numpy_array_index(multi_array, array):
    #TODO - check if multi_array is non empty and if they match size.. throw appropariate warnings
    if multi_array is not None:
        for i,trainp in enumerate(multi_array):
                if array_equal(trainp,array):
                    return True, i
                    
    return False, 0
     
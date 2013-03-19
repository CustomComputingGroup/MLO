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
    if not multi_array is None:
        try:
            for i,trainp in enumerate(multi_array):
                if array_equal(trainp,array):
                    return True, i
        except:
            pass ## should probabnly give debug messages
    return False, 0

##returns class constructor     

def get_trial_dict():
    from model.trials.trial import PSOTrial, PSOTrial_TimeAware 
    return {"PSOTrial" : PSOTrial, 
            "PSOTrial_TimeAware" : PSOTrial_TimeAware,
            "Blank" : None} 

def get_trial_constructor(str_name):
    return get_trial_dict()[str_name]

def get_possible_trial_type():
    return get_trial_dict().keys()
    
def get_trial_type_visualizer(trial_name):
    from views.visualizers.plot import MLOImageViewer, MLOTimeAware_ImageViewer
    return {"PSOTrial" : {"MLOImageViewer" : MLOImageViewer, "default" : MLOImageViewer}, "PSOTrial_TimeAware" : {"MLOTimeAware_ImageViewer" : MLOTimeAware_ImageViewer, "default" : MLOTimeAware_ImageViewer}, 
            "Blank" : {"Blank" : None, "default" : None}}[trial_name]

def get_run_type_visualizer(trial_name):
    from views.visualizers.plot import MLORunReportViewer
    return {"PSOTrial" : {"MLOReportViewer" : MLORunReportViewer, "default" : MLORunReportViewer}, 
            "PSOTrial_TimeAware" : {"MLOReportViewer" : MLORunReportViewer, "default" : MLORunReportViewer}, 
            "Blank" : {"Blank" : None, "default" : None}}[trial_name] 
            

    


    
     
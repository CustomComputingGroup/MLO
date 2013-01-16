from imp import load_source
import logging
import sys


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

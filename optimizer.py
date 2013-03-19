import logging
import argparse
import sys
import os

from views.modes import GUIView, TerminalView
from controller.controller import Controller

from utils import load_script

# Configure logging
LOG_FORMAT = '[%(process)d_%(thread)d] - [%(module)s][%(funcName)s][%(lineno)d] %(levelname)s: %(message)s'
LOG_LEVEL = logging.DEBUG
logging.basicConfig(format=LOG_FORMAT, level=LOG_LEVEL)


def main():
    logging.info('Running MLO')

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', help='fitness script, use fitnessTemplate.py')
    parser.add_argument('-c', help='config script, use test_configuration.py')
    parser.add_argument('--gui', help='enable the GUI',
                        action='store_true')
    parser.add_argument('--restart', help='restart after crash (in the terminal mode)',
                        action='store_true')
    args = parser.parse_args()

    if not (args.gui or args.f or args.c or args.restart):
        print 'Either run the program with --gui or provide a fitness and ' \
              'configuration scripts (with -f and -c)'
        sys.exit(1)
    gui = args.gui
    restart = args.restart
    fitness = None
    configuration = None

    if not gui and not restart and (not args.f or not args.c):
        print 'Make sure to provide both the fitness and configuration scripts or restart flag'
        sys.exit(1)    
    
    ## start in gui mode
    if gui:
        ## initialize controller
        restart = True
        controller = Controller(restart)
        logging.info('Will run with GUI')
        controller.view = GUIView()
    ## start in terminal mode
    else:
        ## initialize controller, restart runs only if requested
        controller = Controller(restart)
        controller.view = TerminalView()
        
        if restart:
            logging.info('Will restart most recent run, for other runs run in GUI mode..')
            fit = self.controller.get_most_recent_fitness_script_folder() + "/" + self.controller.get_most_recent_fitness_script_file() 
            config = self.controller.get_most_recent_configuration_script_folder() + "/" + self.controller.get_most_recent_configuration_script_file() 
        else:
            fit = args.f
            config = args.c
            
        # Load the fitness script
        controller.fitness = load_script(fit, 'fitness')
        if controller.fitness is None:
            logging.info('Could not load fitness file')
            sys.exit(1)

        # Load the configuration script
        controller.configuration = load_script(config, 'configuration')
        if controller.configuration is None:
            logging.info('Could not load configuration file')
            sys.exit(1)
            
    controller.take_over_control()

    logging.info('MLO finished successfully')
    sys.exit(0)
    
if __name__ == '__main__':
    main()

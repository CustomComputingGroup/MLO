import logging
import argparse
import sys
import os

from views.modes import GUIView, TerminalView
from controller.controller import Controller

from utils import load_script

# Configure logging
LOG_FORMAT = '[%(funcName)s] %(levelname)s: %(message)s'
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

    # Load the fitness script
    if args.f:
        fitness = load_script(args.f, 'fitness')
        if fitness is None:
            sys.exit(1)

    # Load the configuration script
    if args.c:
        configuration = load_script(args.c, 'configuration')
        if configuration is None:
            sys.exit(1)

    if not gui and not restart and (not fitness or not configuration):
        print 'Make sure to provide both the fitness and configuration scripts'
        sys.exit(1)

    # Now we have all the inputs and can run the program
    if restart:
        logging.info('Will restart optimisation')
    
    ## initialize controller
    restart = True
    controller = Controller(restart)
    ## start in gui mode
    if gui:
        logging.info('Will run with GUI')
        controller.view = GUIView()
    ## start in terminal mode
    else:
        controller.view = TerminalView()
        
    if not gui:
        controller.fitness = fitness
        controller.configuration = configuration

    controller.take_over_control()

    logging.info('MLO finished successfully')
    sys.exit(0)
    
if __name__ == '__main__':
    main()

import logging
import time
from threading import Thread
from multiprocessing import Process, Pipe
from Queue import Queue
import sys

class Visualizer(Thread):
    """
    Base abstract class for visualizers.
    """

    def __init__(self, controller):
        Thread.__init__(self)
        self.job_backlog = Queue()
        self.controller = controller
        self.stop = False

    def run(self):
        raise NotImplementedError('Visualizer is an abstract class, this '
                                  'should not be called.')

    #function is the printer
    #snapshot contains the state of the visualizer which is going to be fed into the printer. It contains a dictionary of ALL the relevant data. 
    def add_job(self, function, snapshot):
        run_name = snapshot['name']
        logging.info('{} job added to visualizer'.format(run_name))
        self.job_backlog.put((run_name, function, snapshot))

    def terminate(self):
        self.stop = True


class ParallelisedVisualizer(Visualizer):
    """
    A visualizers which uses multiple processes to parallelise and therefore
    speed up visualization.
    """

    def run(self):
        logging.info('Visualizer thread started')

        parent_end, child_end = Pipe()

        # Sensible default value for max_process
        max_process = 2
        process_count = 0

        while not self.stop or not self.job_backlog.empty():
            while parent_end.poll(0.1):
                parent_end.recv() ## currently not using the info... irrelevant
                
                ## TODO - a signal to notify the viewer that visuzaliztion job has been finished... 
                #self.controller.view_update(self)
                process_count -= 1

            if self.job_backlog.empty():
                time.sleep(1)
            elif process_count < max_process:
                process_count += 1

                run_name, function, snapshot = self.job_backlog.get_nowait()
                logging.info('Visualizing {}'.format(run_name))
                p = Process(target=self.render_graph,
                            args=(function, snapshot, run_name, child_end))
                p.daemon = True
                p.start()
                
        logging.info('Visualizer Finished')

    def render_graph(self, function, snapshot, name, child_end):
        try:
            function(snapshot)
            logging.info('{} visualized'.format(name))
            child_end.send(True)
        except Exception,e:
            logging.info('Exception {}'.format(e)) 
            child_end.send(False)
        sys.exit(0)
        
## Todo - need to modify it.... changed the architecture of visualizer ( its currently broken)
class SingleThreadVisualizer(Visualizer):
    """
    A slower visualizer dealing with how Python processes work on OS X.
    See http://bugs.python.org/issue13558 for more details.
    """

    def run(self):
        logging.info('Visualizer thread started')

        while not self.stop or not self.job_backlog.empty():
            if self.job_backlog.empty():
                time.sleep(1)
                continue
            else:
                function, snapshot, trial = self.job_backlog.get_nowait()
                logging.info('Visualizing {}'.format(trial.get_name()))
                self.render_graph(function, snapshot, trial.get_name())
                self.job_backlog.task_done()

    def render_graph(self, function, snapshot, name):
        function(snapshot)
        logging.info('{} visualized'.format(name))
        self.controller.find_trial(name).set_counter_plot(snapshot['counter'])

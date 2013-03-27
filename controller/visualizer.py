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
        self.remove_run_name = []

    def run(self):
        raise NotImplementedError('Visualizer is an abstract class, this '
                                  'should not be called.')

    #function is the printer
    #snapshot contains the state of the visualizer which is going to be fed into the printer. It contains a dictionary of ALL the relevant data. 
    def add_job(self, function, snapshot):
        name = snapshot['name']
        run_name = snapshot['run_name']
        logging.info(name + ' job added to visualizer')
        self.job_backlog.put((run_name, function, snapshot))

    def remove_run_name_jobs(self, run_name):
        logging.info('Jobs for the run ' + run_name + '  will be removed from the visualizer')
        self.remove_run_name.append(run_name)
        
    def terminate(self):
        self.stop = True

    def max_process(self, integer):
        pass

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
                if not (run_name in self.remove_run_name):
                    logging.info('Added job to visuzalizer Que: ' + str(run_name))
                    logging.info('No. of jobs in Que: ' + str(process_count))
                    p = Process(target=self.render_graph,
                                args=(function, snapshot, run_name, child_end))
                    p.start()
                
        logging.info('Visualizer Finished')

    def render_graph(self, function, snapshot, name, child_end):
        logging.info('Visualizing ' + str(name))
        try:
            function(snapshot)
            logging.info(str(name) + ' visualized' )
            child_end.send(True)
        except Exception,e:
            logging.info('Exception ' + str(e)) 
            child_end.send(False)
    
    def max_process(self, integer):
        self.max_process = integer
        
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
                run_name, function, snapshot = self.job_backlog.get_nowait()
                if not (run_name in self.remove_run_name):
                    logging.info('Visualizing ' + str(run_name))
                self.render_graph(function, snapshot, run_name)
                self.job_backlog.task_done()

    def render_graph(self, function, snapshot, name):
        function(snapshot)
        logging.info(str(name) + ' visualized')


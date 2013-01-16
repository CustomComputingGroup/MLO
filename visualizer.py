import logging
import time
from threading import Thread
from multiprocessing import Process, Pipe
from Queue import Queue


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

    def add_job(self, function, snapshot, trial):
        run_name = trial.get_name()
        logging.info('{} job added to visualizer'.format(run_name))
        self.job_backlog.put((function, snapshot, trial))

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
        max_process = 3
        process_count = 0

        while not self.stop or not self.job_backlog.empty():
            while parent_end.poll(0.1):
                (name, counter) = parent_end.recv()
                self.controller.find_trial(name).set_counter_plot(counter)
                process_count -= 1

            if self.job_backlog.empty():
                time.sleep(1)
            elif process_count < max_process:
                process_count += 1

                function, snapshot, trial = self.job_backlog.get_nowait()
                logging.info('Visualizing {}'.format(trial.get_name()))
                p = Process(target=self.render_graph,
                            args=(function, snapshot, trial.get_name(),
                                  child_end))
                p.start()
                self.job_backlog.task_done()

        logging.info('Visualizer Finished')

    def render_graph(self, function, snapshot, name, child_end):
        function(snapshot)
        logging.info('{} visualized'.format(name))
        child_end.send((name, snapshot['counter']))


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

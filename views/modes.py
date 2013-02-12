import wx

from views.gui.windows import RunWindow, UpdateEvent, UpdateEvent2
from visualizers.plot import MLOImageViewer



class View(object):
    """
    Base class for all types of views.
    """

    def initialize(self, controller):
        raise NotImplementedError('View is an abstract class, this should '
                                  'not be called.')

    # TODO - description of what it does
    def update(self, trial):
        raise NotImplementedError('View is an abstract class, this should '
                                  'not be called.')


class TerminalView(View):
    """
    View which just prints to the terminal.
    """

    def initialize(self, controller):
        self.controller = controller       
        self.plot_view = MLOImageViewer
        
        if not self.controller.restart and not (self.controller.fitness and  self.controller.configuration):
            logging.error('Benchmark and/or configuration script not '
                          'provided, terminating...')
            return
        controller.load_profile_dict()
        self.controller.start_run('Default run', self.controller.fitness, self.controller.configuration).join()
        self.controller.visualizer.terminate()
        
    ## Print out run statistics, define a new stats printer
    def update(self, trial=None, run=None, visualize=None):
        if visualize:
            if trial.get_main_counter() % trial.get_configuration().vis_every_X_steps == 0: ## TODO - its not ideal... rethink it... 
                snapshot = trial.snapshot()
                graphdict = self.controller.get_trial_visualization_dict(trial.get_trial_type())
                snapshot.update(graphdict)
                self.controller.visualize(snapshot, self.plot_view.render)

class GUIView(View):
    """
    Proper GUI view.
    """

    def initialize(self, controller):
        
        self.app = wx.App()
        self.window = RunWindow(controller)
        controller.load_profile_dict()
        self.app.MainLoop()

    def update(self, trial=None, run=None, visualize=False):
        wx.PostEvent(self.window.GetEventHandler(), UpdateEvent(trial=trial, run=run, visualize=visualize))
from views.gui.windows import RunWindow, UpdateEvent, RegenEvent
from visualizers.plot import MLOImageViewer

import wx


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

        self.controller.start_run('Default run', self.controller.fitness, self.controller.configuration).join()
        self.controller.visualizer.terminate()
        
    ## Print out run statistics, define a new stats printer
    def update(self, trial):
        if trial.counter_dictionary['g'] % trial.configuration.vis_every_X_steps == 0: ## TODO - its not ideal... rethink it... 
            snapshot = trial.snapshot()
            self.controller.visualize(snapshot, self.plot_view.render)

class GUIView(View):
    """
    Proper GUI view.
    """

    def initialize(self, controller):
        
        self.app = wx.App()
        self.window = RunWindow(controller)
        self.app.MainLoop()

    def update(self, trial):
        wx.PostEvent(self.window.GetEventHandler(), UpdateEvent(trial=trial))

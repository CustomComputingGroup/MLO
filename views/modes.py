from views.gui.windows import RunWindow, UpdateEvent, RegenEvent

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

    def regen(self, trial, counter_plot):
        pass


class TerminalView(View):
    """
    View which just prints to the terminal.
    """

    def initialize(self, controller):
        controller.run_in_terminal()

    def update(self, trial):
        pass


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

    def regen(self, trial, counter_plot):
        wx.PostEvent(self.window.GetEventHandler(),
                     RegenEvent(trial=trial, counter_plot=counter_plot))

import os
import logging
from imp import load_source
import time
import wx
from wx.lib.newevent import NewEvent

from ..visualizers.plot import MLOImageViewer
from utils import get_trial_constructor

UpdateEvent, EVT_UPDATE = NewEvent()
RegenEvent, EVT_REGEN = NewEvent()


class RunWindow(wx.Frame):

    def __init__(self, controller):
        super(RunWindow, self).__init__(None,
                                        title='Machine Learning Optimizer',
                                        size=(1100, 500))

        self.GetEventHandler().Bind(EVT_UPDATE, self.update_run)
        self.plot_view = MLOImageViewer
        self.controller = controller

        ### Set up display
        self.panel = wx.Panel(self, wx.ID_ANY)
        main_sizer = wx.BoxSizer(wx.VERTICAL)

        ### Create and set the Menu bar
        menu_bar = wx.MenuBar()
        file_menu = wx.Menu()
        menu_new = file_menu.Append(wx.ID_NEW, '&New run',
                                    ' Create new run')
        menu_load = file_menu.Append(wx.ID_OPEN, '&load run',
                                       ' load run')
        menu_exit = file_menu.Append(wx.ID_EXIT, 'E&xit',
                                     ' Quit')
        self.Bind(wx.EVT_MENU, self.on_new, menu_new)
        self.Bind(wx.EVT_MENU, self.on_load, menu_load)
        self.Bind(wx.EVT_MENU, self.on_exit, menu_exit)

        menu_bar.Append(file_menu, '&File')
        self.SetMenuBar(menu_bar)

        ### Create the list for displaying runs
        self.list_ctrl = wx.ListCtrl(self.panel, size=(-1, -1),
                                     style=wx.LC_REPORT | wx.LC_SINGLE_SEL |
                                     wx.LC_VRULES | wx.BORDER_SUNKEN)
        self.list_ctrl.InsertColumn(0, 'Run Name', width=280)
        self.list_ctrl.InsertColumn(1, 'Status',     width=100)
        self.list_ctrl.InsertColumn(2, 'Progress',   width=100)
        self.list_ctrl.InsertColumn(3, 'Start Time', width=200)
        self.list_ctrl.InsertColumn(4, 'Running Time', width=200)
        self.list_ctrl.InsertColumn(5, 'Left Time', width=200)
        self.list_ctrl.Bind(wx.EVT_LIST_ITEM_RIGHT_CLICK,
                            self.on_right_click_list)

        main_sizer.Add(self.list_ctrl, 1, wx.GROW | wx.ALL, 10)

        ### Events to catch for repainting the progress bars
        self.list_ctrl.Bind(wx.EVT_PAINT, self.on_paint)
        self.list_ctrl.Bind(wx.EVT_LIST_COL_DRAGGING, self.on_paint)
        self.list_ctrl.Bind(wx.EVT_LIST_COL_END_DRAG, self.on_paint)
        # TODO: EVT_SCROLL not currently supported for ListCtrls, progress bars
        #       will not be displayed correctly in an over-full window
        #self.list_ctrl.Bind(wx.EVT_SCROLL, self.on_paint)
        self.list_ctrl.Bind(wx.EVT_SIZE, self.on_paint)

        ### Add the buttons used to make new runs
        button_sizer = wx.BoxSizer(wx.HORIZONTAL)
        new_run_button = wx.Button(self.panel, label='New Run From Scripts')
        new_run_button.Bind(wx.EVT_BUTTON, self.on_new)
        button_sizer.Add(new_run_button, 1, wx.GROW | wx.RIGHT, 10)
        old_run_button = wx.Button(self.panel, label='Load Run')
        old_run_button.Bind(wx.EVT_BUTTON, self.on_load)
        button_sizer.Add(old_run_button, 1, wx.GROW)

        main_sizer.Add(button_sizer, 0,
                       wx.GROW | wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)

        ### Set the sizer and other variables, and display the window
        self.right_clicked_item = None
        self.childlist = []
        self.bars = {}

        self.panel.SetSizer(main_sizer)
        self.Show()

    def on_paint(self, event=None):
        total_items = self.list_ctrl.GetItemCount()
        if total_items == 0:
            if event:
                event.Skip()
            return

        top_item = self.list_ctrl.GetTopItem()
        visible_items = range(top_item,
                              top_item +
                              min(self.list_ctrl.GetCountPerPage()+1,
                              total_items))

        ### Initialise coordinates for creating the bars
        rect = self.list_ctrl.GetItemRect(top_item)
        size = (self.list_ctrl.GetColumnWidth(2)-4, rect[3]-4)
        x = rect[0] + sum([self.list_ctrl.GetColumnWidth(j)
                          for j in range(0, 2)]) + 13
        y = rect[1] + 14
        inc = rect[3]

        ### Show and hide bars as necessary
        for i in range(0, total_items):
            run_name = self.list_ctrl.GetItem(i).GetText()
            bar = self.bars[run_name]

            if i in visible_items:
                if bar.GetPosition() != (x, y):
                    # Necessary on Windows?
                    if wx.Platform != "__WXMSW__":
                        bar.Hide()
                    bar.SetPosition((x, y))
                bar.SetSize(size)
                bar.Show()
                y += inc
            else:
                bar.Hide()

        if event:
            event.Skip()

    def on_right_click_list(self, event):
        self.right_clicked_item = event.GetText()
        menu = wx.Menu()
        try:
            logging.info('click')
            status = self.controller.get_run_status(self.right_clicked_item)
            if status == 'Running':
                pause = menu.Append(wx.ID_ANY, 'Pause')
                self.Bind(wx.EVT_MENU, self.on_pause, pause)
            elif status == 'Paused':
                resume = menu.Append(wx.ID_ANY, 'Resume')
                self.Bind(wx.EVT_MENU, self.on_resume, resume)
            elif status == 'Finished':
                pass

            show_trials = menu.Append(wx.ID_ANY, 'Show Trials')
            self.Bind(wx.EVT_MENU, self.on_show_trials, show_trials)
                
            show_graphs = menu.Append(wx.ID_ANY, 'Show Graphs')
            self.Bind(wx.EVT_MENU, self.on_show_graphs, show_graphs)
            delete = menu.Append(wx.ID_ANY, 'Delete')
            self.Bind(wx.EVT_MENU, self.on_delete, delete)
            self.PopupMenu(menu, event.GetPosition())
            menu.Destroy()
        except KeyError,e:
            logging.info('No run selected')
            pass

    def on_new(self, event):
        self.childlist.append(NewRunWindow(self, 'New Run', self.controller))
        event.Skip()

    def on_load(self, event):
        self.childlist.append(loadRunWindow(self, 'load Run'))
        event.Skip()

    def on_pause(self, event):
        ### Pause run in controller
        logging.info('click')
        item = self.list_ctrl.GetFocusedItem()
        name = self.list_ctrl.GetItem(item).GetText()
        self.controller.pause_run(name)

    def on_resume(self, event):
        logging.info('click')
        ### Resume run in controller
        item = self.list_ctrl.GetFocusedItem()
        name = self.list_ctrl.GetItem(item).GetText()
        self.controller.resume_run(name)

    def on_delete(self, event):
        logging.info('click')
        dlg = wx.MessageDialog(self, 'Delete this run?', '',
                               wx.OK | wx.CANCEL | wx.ICON_QUESTION)
        result = dlg.ShowModal()
        if result == wx.ID_OK:
            ### Stop the run, and delete all references to it
            item = self.list_ctrl.GetFocusedItem()
            name = self.list_ctrl.GetItem(item).GetText()
            self.bars[name].Destroy()
            del self.bars[name]
            self.list_ctrl.DeleteItem(item)
            self.controller.delete_run(name)
            self.list_ctrl.Refresh()

    def on_show_graphs(self, event):
        logging.info('click')
        run_name = self.list_ctrl.GetItem(
            self.list_ctrl.GetFocusedItem()).GetText()
        run = self.controller.find_run(run_name)
        self.childlist.append(GraphWindow(self,
                                          'Graphs For ' +
                                          self.right_clicked_item,
                                          run,
                                          self.controller))
        event.Skip()

    def on_show_trials(self, event):
        logging.info('click')
        run_name = self.list_ctrl.GetItem(
            self.list_ctrl.GetFocusedItem()).GetText()
        run = self.controller.find_run(run_name)
        self.childlist.append(TrialWindow('Trials For ' +
                                          self.right_clicked_item,
                                          run,
                                          self.controller))
        event.Skip()
        
    def on_exit(self, event):
        logging.info('click')
        self.controller.kill_visualizer()
        self.Destroy()

    def add_listctrl_run(self, run):
        self.list_ctrl.Append((run.get_name(), run.get_status(), '',
                              run.get_start_time()))

    def add_progress_bar(self, run):
        rect = self.list_ctrl.GetItemRect(0)
        size = (self.list_ctrl.GetColumnWidth(2)-4, rect[3]-4)
        bar = wx.Gauge(self.panel, range=run.get_configuration().trials_count, size=size)
        bar.SetValue(run.get_main_counter())
        self.bars[run.get_name()] = bar

    def start_run(self, name, fitness, configuration):
        self.controller.start_run(name, fitness, configuration)

    def load_run(self, folder_path):
        self.controller.load_run(folder_path)

    def update_run(self, event):
        ### Called to update display to represent run's changed state
        run = event.run
        
        if not (event.trial is None): ## case when we only want to update run state (possible)
            trial = event.trial
            if (trial.get_main_counter() % trial.get_configuration().vis_every_X_steps) == 0: ### TODO - it should really be done a bit differently...
                 self.visuzalize_trial(trial)
             
        drawn = run.get_name() in self.bars
        if drawn:
            index = self.list_ctrl.FindItem(0, run.get_name())
            self.list_ctrl.SetStringItem(index, 1, run.get_status())
            self.update_bar(run)
        else:
            self.add_listctrl_run(run)
            self.add_progress_bar(run)
            self.on_paint()

    def update_run_graph(self, run):
        ### May update the displayed graph in all graph windows for run
        run_ref = run
        for child in self.childlist:
            try:
                if child.run == run_ref:
                    child.update_run()
            except (wx.PyDeadObjectError, AttributeError):
                pass
            
    def update_bar(self, run):
        bar = self.bars[run.get_name()]
        bar.SetValue(run.get_main_counter())

    def get_graph_attributes(self, run, graph_name):
        return self.plot_view.get_attributes(graph_name)  ### TODO -- change this so that the plow_view is going to be selected based on the run type

    def visuzalize_run(self, run):
        snapshot = run.snapshot()
        snapshot.update(self.controller.get_run_visualization_dict(run.get_run_type()))
        self.controller.visualize(snapshot, self.plot_view.render) ### TODO -- change this so that the plow_view is going to be selected based on the run type
        
    def visuzalize_trial(self, trial):
        snapshot = trial.snapshot()
        snapshot.update(self.controller.get_trial_visualization_dict(trial.get_trial_type()))
        self.controller.visualize(snapshot, self.plot_view.render) ### TODO -- change this so that the plow_view is going to be selected based on the trial type
        
class TrialWindow(wx.Frame):

    def __init__(self, window_title, run, controller):
        super(TrialWindow, self).__init__(None,
                                        title=window_title,
                                        size=(1100, 500))

        self.GetEventHandler().Bind(EVT_UPDATE, self.update_trial)
        self.plot_view = MLOImageViewer
        self.controller = controller
        self.run = run
        ### Set up display
        self.panel = wx.Panel(self, wx.ID_ANY)
        main_sizer = wx.BoxSizer(wx.VERTICAL)

        ### Create and set the Menu bar
        menu_bar = wx.MenuBar()
        file_menu = wx.Menu()
        menu_exit = file_menu.Append(wx.ID_EXIT, 'E&xit',
                                     ' Quit')
        self.Bind(wx.EVT_MENU, self.on_exit, menu_exit)

        menu_bar.Append(file_menu, '&File')
        self.SetMenuBar(menu_bar)

        ### Create the list for displaying trials
        self.list_ctrl = wx.ListCtrl(self.panel, size=(-1, -1),
                                     style=wx.LC_REPORT | wx.LC_SINGLE_SEL |
                                     wx.LC_VRULES | wx.BORDER_SUNKEN)
        self.list_ctrl.InsertColumn(0, 'Trial Name', width=280)
        self.list_ctrl.InsertColumn(1, 'Status',     width=100)
        self.list_ctrl.InsertColumn(2, 'Progress',   width=100)
        self.list_ctrl.InsertColumn(3, 'Start Time', width=200)
        self.list_ctrl.InsertColumn(4, 'Running Time', width=200)
        self.list_ctrl.InsertColumn(5, 'Left Time', width=200)
        self.list_ctrl.Bind(wx.EVT_LIST_ITEM_RIGHT_CLICK,
                            self.on_right_click_list)

        main_sizer.Add(self.list_ctrl, 1, wx.GROW | wx.ALL, 10)

        ### Events to catch for repainting the progress bars
        self.list_ctrl.Bind(wx.EVT_PAINT, self.on_paint)
        self.list_ctrl.Bind(wx.EVT_LIST_COL_DRAGGING, self.on_paint)
        self.list_ctrl.Bind(wx.EVT_LIST_COL_END_DRAG, self.on_paint)
        # TODO: EVT_SCROLL not currently supported for ListCtrls, progress bars
        #       will not be displayed correctly in an over-full window
        #self.list_ctrl.Bind(wx.EVT_SCROLL, self.on_paint)
        self.list_ctrl.Bind(wx.EVT_SIZE, self.on_paint)

        #main_sizer.Add(button_sizer, 0,
        #               wx.GROW | wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)

        ### Set the sizer and other variables, and display the window
        self.right_clicked_item = None
        self.childlist = []
        self.bars = {}

        ### add all the runs trials
        for trial in self.run.trials:
            self.add_listctrl_trial(trial)
            self.add_progress_bar(trial)
            self.on_paint()
            
        self.panel.SetSizer(main_sizer)
        self.Show()

    def on_paint(self, event=None):
        total_items = self.list_ctrl.GetItemCount()
        if total_items == 0:
            if event:
                event.Skip()
            return

        top_item = self.list_ctrl.GetTopItem()
        visible_items = range(top_item,
                              top_item +
                              min(self.list_ctrl.GetCountPerPage()+1,
                              total_items))

        ### Initialise coordinates for creating the bars
        rect = self.list_ctrl.GetItemRect(top_item)
        size = (self.list_ctrl.GetColumnWidth(2)-4, rect[3]-4)
        x = rect[0] + sum([self.list_ctrl.GetColumnWidth(j)
                          for j in range(0, 2)]) + 13
        y = rect[1] + 14
        inc = rect[3]

        ### Show and hide bars as necessary
        for i in range(0, total_items):
            trial_name = self.list_ctrl.GetItem(i).GetText()
            bar = self.bars[trial_name]

            if i in visible_items:
                if bar.GetPosition() != (x, y):
                    # Necessary on Windows?
                    if wx.Platform != "__WXMSW__":
                        bar.Hide()
                    bar.SetPosition((x, y))
                bar.SetSize(size)
                bar.Show()
                y += inc
            else:
                bar.Hide()

        if event:
            event.Skip()

    def on_right_click_list(self, event):
        self.right_clicked_item = event.GetText()
        menu = wx.Menu()
        try:
            logging.info('click')
            status = self.controller.get_trial_status(self.right_clicked_item)
            if status == 'Running':
                pause = menu.Append(wx.ID_ANY, 'Pause')
                self.Bind(wx.EVT_MENU, self.on_pause, pause)
            elif status == 'Paused':
                resume = menu.Append(wx.ID_ANY, 'Resume')
                self.Bind(wx.EVT_MENU, self.on_resume, resume)
            elif status == 'Finished':
                pass

            show_graphs = menu.Append(wx.ID_ANY, 'Show Graphs')
            self.Bind(wx.EVT_MENU, self.on_show_graphs, show_graphs)
            delete = menu.Append(wx.ID_ANY, 'Delete')
            self.Bind(wx.EVT_MENU, self.on_delete, delete)
            self.PopupMenu(menu, event.GetPosition())
            menu.Destroy()
        except KeyError,e:
            logging.info('No trial selected')
            pass

    def on_pause(self, event):
        ### Pause trial in controller
        logging.info('click')
        item = self.list_ctrl.GetFocusedItem()
        name = self.list_ctrl.GetItem(item).GetText()
        self.controller.pause_trial(name)

    def on_resume(self, event):
        logging.info('click')
        ### Resume trial in controller
        item = self.list_ctrl.GetFocusedItem()
        name = self.list_ctrl.GetItem(item).GetText()
        self.controller.resume_trial(name)

        
    ## TODO - not sure if not to disable this...
    def on_delete(self, event):
        logging.info('click')
        dlg = wx.MessageDialog(self, 'Delete this trial?', '',
                               wx.OK | wx.CANCEL | wx.ICON_QUESTION)
        result = dlg.ShowModal()
        if result == wx.ID_OK:
            ### Stop the trial, and delete all references to it
            item = self.list_ctrl.GetFocusedItem()
            name = self.list_ctrl.GetItem(item).GetText()
            self.bars[name].Destroy()
            del self.bars[name]
            self.list_ctrl.DeleteItem(item)
            self.controller.delete_trial(name)
            self.list_ctrl.Refresh()

    def on_show_graphs(self, event):
        logging.info('click')
        trial_name = self.list_ctrl.GetItem(
            self.list_ctrl.GetFocusedItem()).GetText()
        trial = self.controller.find_trial(trial_name)
        self.childlist.append(GraphWindow(self,
                                          'Graphs For ' +
                                          self.right_clicked_item,
                                          trial,
                                          self.controller))
        event.Skip()

    def on_exit(self, event):
        logging.info('click')
        self.Destroy()

    def add_listctrl_trial(self, trial):
        self.list_ctrl.Append((trial.get_name(), trial.get_status(), '',
                              trial.get_start_time()))

    def add_progress_bar(self, trial):
        rect = self.list_ctrl.GetItemRect(0)
        size = (self.list_ctrl.GetColumnWidth(2)-4, rect[3]-4)
        bar = wx.Gauge(self.panel, range=trial.get_configuration().max_iter, size=size)
        bar.SetValue(trial.get_main_counter())
        self.bars[trial.get_name()] = bar

    def update_trial(self, event):
        ### Called to update display to represent trial's changed state
        trial = event.trial
        run = event.run
        
        if self.run == run:
            drawn = trial.get_name() in self.bars
            index = self.list_ctrl.FindItem(0, trial.get_name())
            self.list_ctrl.SetStringItem(index, 1, trial.get_status())
            self.update_bar(trial)


    def update_trial_graph(self, trial):
        ### May update the displayed graph in all graph windows for trial
        trial_ref = trial
        for child in self.childlist:
            try:
                if child.trial == trial_ref:
                    child.update_trial()
            except (wx.PyDeadObjectError, AttributeError):
                pass
            
    def update_bar(self, trial):
        bar = self.bars[trial.get_name()]
        bar.SetValue(trial.get_main_counter())

    def get_graph_attributes(self, trial, graph_name):
        return self.plot_view.get_attributes(graph_name)  ### TODO -- change this so that the plow_view is going to be selected based on the trial type

    def visuzalize_trial(self, trial):
        self.GetParent().visuzalize_trial(trial)
        
class GraphWindow(wx.Frame):

    def __init__(self, parent, title, trial, controller):
        super(GraphWindow, self).__init__(parent, title=title,
                                          size=(1320, 650))

        ### Initialisation (take graph options from given trial)
        self.trial = trial
        self.controller = controller
        self.images_folder = trial.get_images_folder()
        
        ### Set up display
        self.panel = wx.ScrolledWindow(self)
        self.panel.SetScrollRate(4, 16)
        main_sizer = wx.BoxSizer(wx.VERTICAL)
        graph_sizer = wx.BoxSizer(wx.HORIZONTAL)
        option_sizer = wx.BoxSizer(wx.HORIZONTAL)

        ### Initialise scrolling and options buttons
        first_button = wx.Button(self.panel, label='<<', size=(40, -1))
        first_button.Bind(wx.EVT_BUTTON, self.on_first)
        
        left_button = wx.Button(self.panel, label='<', size=(40, -1))
        left_button.Bind(wx.EVT_BUTTON, self.on_left)
        
        right_button = wx.Button(self.panel, label='>', size=(40, -1))
        right_button.Bind(wx.EVT_BUTTON, self.on_right)
        
        last_button = wx.Button(self.panel, label='>>', size=(40, -1))
        last_button.Bind(wx.EVT_BUTTON, self.on_last)
        
        options_button = wx.Button(self.panel, label='Options', size=(400, -1))
        options_button.Bind(wx.EVT_BUTTON, self.on_options)

        ## TODO - Fix the functionality of 
        ##self.update_checkbox = wx.CheckBox(self.panel,
        ##                                   label='Automatically update graphs')
                       
        refresh_button = wx.Button(self.panel, label='Refresh', size=(60, -1))
        refresh_button.Bind(wx.EVT_BUTTON, self.on_refresh)

        regenerate_button = wx.Button(self.panel, label='Regenerate All', size=(120, -1))
        regenerate_button.Bind(wx.EVT_BUTTON, self.on_regenerate)
                       
        self.bmp = None
        self.update_plots()
        self.set_first_plot()

        ### Finish sizer setup
        graph_sizer.Add(first_button, 0, wx.GROW | wx.ALL, 10)
        graph_sizer.Add(left_button, 0, wx.GROW | wx.ALL, 10)
        graph_sizer.Add(self.bmp, 1, wx.GROW | wx.ALL, 10)
        graph_sizer.Add(right_button, 0, wx.GROW | wx.ALL, 10)
        graph_sizer.Add(last_button, 0, wx.GROW | wx.ALL, 10)
        main_sizer.Add(graph_sizer, 0)

        ##option_sizer.Add(self.update_checkbox, 0, wx.ALL, 10)
        option_sizer.Add(refresh_button, 0, wx.ALL, 10)
        option_sizer.Add(regenerate_button, 0, wx.ALL, 10)
        option_sizer.AddStretchSpacer(1)
        option_sizer.Add(options_button, 0, wx.ALIGN_RIGHT | wx.ALL, 10)
        main_sizer.Add(option_sizer, 1, wx.GROW)

        self.childlist = []

        self.panel.SetSizer(main_sizer)
        self.Centre()
        self.Show()

    def on_regenerate(self, event):
        trial = self.trial
        for iteration in self.trial.get_main_counter_iterator():
            Trial = get_trial_constructor(trial.get_trial_type())
            vis_trial = Trial(trial.get_trial_no(), trial.get_name(), trial.get_fitness(),
                              trial.get_configuration(), trial.controller,
                              trial.get_run_results_folder_path())
            vis_trial.load(iteration)
            self.visuzalize_trial(vis_trial)
        
    def on_refresh(self, event):
        logging.info('click')
        self.set_previous_plot()
        self.update_image()
        
    def on_left(self, event):
        logging.info('click')
        self.set_previous_plot()
        self.update_image()
        
    def on_first(self, event):
        logging.info('click')
        self.set_first_plot()
        self.update_image()

    def on_right(self, event):
        logging.info('click')
        self.set_next_plot()
        self.update_image()

    def on_last(self, event):
        logging.info('click')
        self.set_last_plot()
        self.update_image()
        
    def on_options(self, event):
        logging.info('click')
        self.childlist.append(OptionsWindow(self,
                                            'Graphs Options For ' +
                                            self.trial.get_name(),
                                            self.trial,
                                            self.controller))
        event.Skip()

    def make_file_name(self, plot):
        return self.images_folder + "/" + plot

    def update_trial(self):
        self.update_image()
        ##if self.update_checkbox.GetValue():
        ##    self.set_last_plot()
        ##    self.update_image()
        ##
            ### Update to keep track of trial's newest plot
          ##  if self.latest_plot % self.gap == 0:
            ##    self.current_plot = self.latest_plot
             ##   self.update_image()
        ##elif counter_plot == self.current_plot:
            ### Or if the graph has been regenerated
            
    def update_image(self):
        ### Called when current_plot is updated, to display the new graph
        try:
            self.current_plot = self.all_plots[self.plot_index]
            self.file_name = self.make_file_name(self.current_plot)
            image = wx.Image(self.file_name, wx.BITMAP_TYPE_PNG)
        except Exception,e:
            current_wd = os.getcwd()
            os.chdir(os.path.dirname(os.path.realpath(__file__)))
            self.file_name = 'img/nothing.jpg'
            logging.info('No images found or error happened {}'.format(e))
            self.plot_index = 0
            self.file_name
            image = wx.Image(self.file_name, wx.BITMAP_TYPE_JPEG)
            os.chdir(current_wd)
        image = image.Scale(1060, 580, wx.IMAGE_QUALITY_HIGH)
        if self.bmp is None:
            self.bmp = wx.StaticBitmap(self.panel, wx.ID_ANY,
                           wx.BitmapFromImage(image),
                           size=(image.GetWidth(), image.GetHeight()))
        else:
            self.bmp.SetBitmap(wx.BitmapFromImage(image))   

    def regenerate_graph(self, trial):
        logging.debug('current plot: {}'.format(self.current_plot))
        gd = trial.graph_dictionary  # Save graph_dictionary
        Trial = get_trial_constructor(trial.get_trial_type())
        vis_trial = Trial(trial.get_trial_no(), trial.get_name(), trial.get_fitness(),
                          trial.get_configuration(), trial.controller,
                          trial.run_results_folder_path)
        vis_trial.load(generation)
        vis_trial.graph_dictionary = gd
        self.visuzalize_trial(trial)
        
    def visuzalize_trial(self, trial):
        self.GetParent().visuzalize_trial(trial)
        
    def get_graph_attributes(self, trial, graph_name):
        return self.GetParent().get_graph_attributes(trial, graph_name)
       
    ## returns the list of all possible plots
    def update_plots(self):
        self.all_plots = os.listdir(self.images_folder)
        self.all_plots.sort()
        
    def set_first_plot(self):
        self.update_plots()
        self.plot_index = 0
        self.update_image()
       
    def set_last_plot(self):
        self.update_plots()
        self.plot_index = len(self.all_plots) - 1
        self.update_image()
       
    def set_previous_plot(self):
        self.update_plots()
        self.plot_index = max(0,self.plot_index - 1)
        self.update_image()
       
    def set_next_plot(self):
        self.update_plots()
        self.plot_index = min(self.plot_index + 1,len(self.all_plots) - 1)
        self.update_image()

class OptionsWindow(wx.Frame):

    def __init__(self, parent, title, trial, controller):
        super(OptionsWindow, self).__init__(parent, title=title,
                                            size=(1000, 500))

        self.trial = trial
        self.controller = controller
        self.graph_names = self.controller.get_trial_visualization_dict(self.trial.get_trial_type())['graph_names']

        ### Set up display
        self.panel = wx.ScrolledWindow(self)
        self.panel.SetScrollRate(4, 16)
        self.tc_dictionary = {}
        self.checkbox_dictionary = {}

        main_sizer = wx.BoxSizer(wx.VERTICAL)
        option_sizer = wx.BoxSizer(wx.VERTICAL)
        button_sizer = wx.BoxSizer(wx.HORIZONTAL)

        ### Add the title option
        title_sizer = wx.BoxSizer(wx.VERTICAL)
        self.tc_dictionary['title'] = wx.TextCtrl(
            self.panel,
            value=self.controller.get_trial_visualization_dict(self.trial.get_trial_type())['graph_title'])
        title_sizer.Add(wx.StaticText(self.panel, -1, 'Title:'), 0,
                        wx.TOP | wx.LEFT, 10)
        title_sizer.Add(self.tc_dictionary['title'], 1,
                        wx.GROW | wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)
        option_sizer.Add(title_sizer, 1, wx.GROW)

        ### Add the checkbox sizers
        trial_gd = self.controller.get_trial_visualization_dict(self.trial.get_trial_type())['all_graph_dicts']
        checkbox_sizer = wx.BoxSizer(wx.HORIZONTAL)
        for graph_name in self.graph_names:
            self.checkbox_dictionary[graph_name] = wx.CheckBox(
                self.panel, label='Regenerate '+graph_name+' Graph?')
            self.checkbox_dictionary[graph_name].SetValue(
                trial_gd[graph_name]['generate'])
            checkbox_sizer.Add(self.checkbox_dictionary[graph_name],
                               1, wx.GROW)
        option_sizer.Add(checkbox_sizer, 0, wx.GROW)

        ### Generate and add sizers for each attribute
        attributes = self.get_graph_attributes(self.trial, 'All')
        graph_attributes_dictionary = {}
        for graph_name in self.graph_names:
            graph_attributes_dictionary[graph_name] = \
                self.get_graph_attributes(self.trial, graph_name)

        for attribute in attributes:
            generate = False
            attribute_sizer = wx.BoxSizer(wx.HORIZONTAL)
            for graph_name in self.graph_names:
                if attribute in graph_attributes_dictionary[graph_name]:
                    temp_sizer, self.tc_dictionary[attribute+graph_name] = \
                        self.make_sizer(graph_name, attribute,
                                        'Plot '+graph_name+' '+attribute+':')
                    attribute_sizer.Add(temp_sizer, 1, wx.GROW)
                    generate = True
                else:
                    attribute_sizer.AddStretchSpacer(1)
            if generate:
                option_sizer.Add(attribute_sizer, 1, wx.GROW)

        
        ### refresh view button TODO - 
        save_button = wx.Button(self.panel, label='Save', size=(-1, -1))
        save_button.Bind(wx.EVT_BUTTON, self.save_visualization_dict)
        button_sizer.Add(save_button, 0, wx.ALIGN_RIGHT | wx.ALL, 10)
        
        
        main_sizer.Add(option_sizer, 0, wx.GROW)
        main_sizer.Add(button_sizer, 0, wx.ALIGN_RIGHT | wx.TOP, 10)

        self.panel.SetSizer(main_sizer)
        self.Centre()
        self.Show()

    def make_sizer(self, name, dictionary_value, text):
        new_sizer = wx.BoxSizer(wx.VERTICAL)
        new_name = wx.StaticText(self.panel, -1, text)
        trial_gd_name = self.controller.get_trial_visualization_dict(self.trial.get_trial_type())['all_graph_dicts'][name]
        if dictionary_value in trial_gd_name:
            tc_value = trial_gd_name[dictionary_value]
        else:
            tc_value = ''
        new_tc = wx.TextCtrl(self.panel, value=tc_value)

        new_sizer.Add(new_name, 0, wx.TOP | wx.LEFT, 10)
        new_sizer.Add(new_tc, 1, wx.GROW | wx.LEFT | wx.RIGHT | wx.BOTTOM, 10)
        return (new_sizer, new_tc)
        
    def save_visualization_dict(self, event):
        ### Save all regeneration data
        trial_gd = self.controller.get_trial_visualization_dict(self.trial.get_trial_type())['all_graph_dicts']
        self.controller.get_trial_visualization_dict(self.trial.get_trial_type())['graph_title'] = \
            self.tc_dictionary['title'].GetValue()
        for graph_name in self.graph_names:
            trial_gd[graph_name]['generate'] = \
                self.checkbox_dictionary[graph_name].IsChecked()

            attributes = self.get_graph_attributes(self.trial, graph_name)
            for attribute in attributes:
                trial_gd[graph_name][attribute] = \
                    self.tc_dictionary[attribute+graph_name].GetValue()
        #self.controller.update_trial_visualization_dict(self.trial.get_trial_type(), trial_gd)
        self.controller.save_profile_dict()
        self.Close(True)
        
    def get_graph_attributes(self, trial, graph_name):
        return self.GetParent().get_graph_attributes(trial, graph_name)


class NewRunWindow(wx.Frame):

    def __init__(self, parent, title, controller):
        super(NewRunWindow, self).__init__(parent, title=title,
                                           size=(440, 305),
                                           style=wx.DEFAULT_FRAME_STYLE ^
                                           wx.RESIZE_BORDER)

        
        self.fitness_path = None
        self.config_path = None
        self.controller = controller
        #self.currentDirectory = self.controller.get_most_recent_dir()
        ### Set up display
        self.panel = wx.Panel(self)
        main_sizer = wx.BoxSizer(wx.VERTICAL)

        ### Initialise options
        name_sizer = wx.BoxSizer(wx.HORIZONTAL)
        trial_name = wx.StaticText(self.panel, -1, 'Run name:')
        name_sizer.Add(trial_name, 0, wx.ALL, border=10)
        self.tc = wx.TextCtrl(self.panel)
        name_sizer.Add(self.tc, 1, wx.ALL, 10)

        ### Fitness sizer and description
        fitness_sizer = wx.BoxSizer(wx.HORIZONTAL)
        fitness_button = wx.Button(self.panel,
                                   label='Select fitness script',
                                   size=(200, -1))
        fitness_button.name = 'fitness'
        fitness_button.Bind(wx.EVT_BUTTON, self.get_on_open_file(self.controller.get_most_recent_fitness_script_file(), self.controller.get_most_recent_configuration_script_folder()))
        
        self.fitness_path = self.controller.get_most_recent_fitness_script_file()
        if self.fitness_path:
            self.name_of_fitness = wx.StaticText(self.panel, -1,self.controller.get_most_recent_fitness_script_file())
            self.fitness_path = self.controller.get_most_recent_fitness_script_folder() + "/" + self.controller.get_most_recent_fitness_script_file()
            logging.info(self.fitness_path)
        else:
            self.name_of_fitness = wx.StaticText(self.panel, -1, 
                                             'Please select a file')
                                                 
        fitness_sizer.Add(fitness_button, 0, wx.ALIGN_LEFT | wx.ALL, 10)
        fitness_sizer.Add(self.name_of_fitness, 1, wx.ALL, 15)

        fitness_description_sizer = wx.BoxSizer(wx.HORIZONTAL)
        fitness_description_sizer.Add(wx.StaticText(
            self.panel, -1,
            'Fitness script defines a fitness function which will be used ' +
            'for\nevaluating fitness of particles found during ' +
            'optimisation.'))

        ### Configuration sizer and description
        config_sizer = wx.BoxSizer(wx.HORIZONTAL)
        config_button = wx.Button(self.panel,
                                  label='Select configuration script',
                                  size=(200, -1))
        config_button.name = 'config'
        
        
        config_button.Bind(wx.EVT_BUTTON, self.get_on_open_file(self.controller.get_most_recent_configuration_script_file(), self.controller.get_most_recent_configuration_script_folder()))
        
        if self.fitness_path:
            self.name_of_config = wx.StaticText(self.panel, -1,self.controller.get_most_recent_configuration_script_file())
            self.config_path = self.controller.get_most_recent_configuration_script_folder() + "/" + self.controller.get_most_recent_configuration_script_file()
            logging.info(self.config_path)
            logging.info(self.controller.get_most_recent_configuration_script_file())
        else:
            self.name_of_config = wx.StaticText(self.panel, -1,
                                                'Please select a file')
                                                
        config_sizer.Add(config_button, 0, wx.ALIGN_LEFT | wx.ALL, 10)
        config_sizer.Add(self.name_of_config, 1, wx.ALL, 15)

        config_description_sizer = wx.BoxSizer(wx.HORIZONTAL)
        config_description_sizer.Add(wx.StaticText(
            self.panel, -1,
            'Configuration script defines settings for trials (e.g. count,\n' +
            'population size) and output graphs (e.g. axes labels).'))

        ### Button sizer
        button_sizer = wx.BoxSizer(wx.HORIZONTAL)
        confirm_button = wx.Button(self.panel, id=wx.ID_OK, size=(-1, -1))
        confirm_button.Bind(wx.EVT_BUTTON, self.on_confirm)
        button_sizer.Add(confirm_button, 0, wx.ALIGN_RIGHT | wx.ALL, 10)
        cancel_button = wx.Button(self.panel, id=wx.ID_CANCEL, size=(-1, -1))
        cancel_button.Bind(wx.EVT_BUTTON, self.on_cancel)
        button_sizer.Add(cancel_button, 0, wx.ALIGN_RIGHT | wx.ALL, 10)

        ### Add all options, and display
        main_sizer.Add(name_sizer, 0,
                       wx.GROW | wx.LEFT | wx.RIGHT | wx.TOP, 10)
        main_sizer.Add(fitness_sizer, 0,
                       wx.ALIGN_LEFT | wx.LEFT | wx.RIGHT, 10)
        main_sizer.Add(fitness_description_sizer, 0, wx.ALIGN_CENTER)
        main_sizer.AddSpacer(5)
        main_sizer.Add(config_sizer, 0,
                       wx.ALIGN_LEFT | wx.TOP | wx.LEFT | wx.RIGHT, 10)
        main_sizer.Add(config_description_sizer, 0, wx.ALIGN_CENTER)
        main_sizer.AddSpacer(15)
        main_sizer.Add(button_sizer, 0, wx.ALIGN_RIGHT)

        self.panel.SetSizer(main_sizer)
        self.Centre()
        self.Show()

    def on_confirm(self, event):
        if not self.fitness_path:
            logging.debug('No fitness script selected')
            wx.MessageBox('Please select a fitness script',
                          'Action required',
                          wx.OK | wx.ICON_EXCLAMATION)
            return

        if not self.config_path:
            logging.debug('No config script selected')
            wx.MessageBox('Please select a configuration script',
                          'Action required',
                          wx.OK | wx.ICON_EXCLAMATION)
            return

        name = self.tc.GetValue()
        fitness = load_source('fitness', self.fitness_path)
        configuration = load_source('configuration', self.config_path)

        self.GetParent().start_run(name, fitness, configuration)
        self.Close(True)

    def on_cancel(self, event):
        self.Close(True)

    def get_on_open_file(self, default_file, current_directory):
        logging.info(str(current_directory))
        logging.info(str(default_file))
        def on_open_file(event):
            logging.info(default_file)
            wildcard = "Python source (*.py)|*.py|" \
                       "All files (*.*)|*.*"
            dlg = wx.FileDialog(
                self, message='Choose a file',
                defaultDir=current_directory,
                defaultFile=default_file,
                wildcard=wildcard,
                style=wx.OPEN | wx.CHANGE_DIR
            )
            if dlg.ShowModal() == wx.ID_OK:
                path = dlg.GetPaths()

                name = event.GetEventObject().name
                if name == 'config':
                    self.config_path = dlg.GetPath()
                    self.name_of_config.SetLabel(dlg.GetFilename())
                    logging.info('config path:' + dlg.GetFilename())
                elif name == 'fitness':
                    self.fitness_path = dlg.GetPath()
                    self.name_of_fitness.SetLabel(dlg.GetFilename())
                    logging.info('fitness path:' + dlg.GetFilename())
                else:
                    logging.error('File Dialog incorrectly called')
        return on_open_file
    

class loadRunWindow(wx.Frame):

    def __init__(self, parent, title):
        super(loadRunWindow, self).__init__(parent, title=title,
                                              size=(600, 170),
                                              style=wx.DEFAULT_FRAME_STYLE ^
                                              wx.RESIZE_BORDER)

        self.currentDirectory = os.getcwd()
        self.folder_path = None

        ### Set up display
        self.panel = wx.Panel(self)
        main_sizer = wx.BoxSizer(wx.VERTICAL)

        ### CSV sizer and description
        select_sizer = wx.BoxSizer(wx.HORIZONTAL)
        select_button = wx.Button(self.panel, label='Select Run Folder',
                                  size=(200, -1))
        select_button.Bind(wx.EVT_BUTTON, self.on_open_file)
        self.selected_folder_name = wx.StaticText(self.panel, -1,
                                                  'Please select a folder')
        select_sizer.Add(select_button, 0, wx.ALIGN_LEFT | wx.ALL, 10)
        select_sizer.Add(self.selected_folder_name, 1, wx.ALL, 15)

        select_description_sizer = wx.BoxSizer(wx.HORIZONTAL)
        select_description_sizer.Add(wx.StaticText(
            self.panel, -1,
            'The CSV files are automatically generated each run.\n' +
            'Please select the folder containing the files for the ' +
            'trial you wish to rerun.'))

        ### Button sizer
        button_sizer = wx.BoxSizer(wx.HORIZONTAL)
        confirm_button = wx.Button(self.panel, id=wx.ID_OK, size=(-1, -1))
        confirm_button.Bind(wx.EVT_BUTTON, self.on_confirm)
        button_sizer.Add(confirm_button, 0, wx.ALIGN_RIGHT | wx.ALL, 10)
        cancel_button = wx.Button(self.panel, id=wx.ID_CANCEL, size=(-1, -1))
        cancel_button.Bind(wx.EVT_BUTTON, self.on_cancel)
        button_sizer.Add(cancel_button, 0, wx.ALIGN_RIGHT | wx.ALL, 10)

        ### Add all options to main sizer, and display
        main_sizer.Add(select_sizer, 0,
                       wx.ALIGN_LEFT | wx.TOP | wx.LEFT | wx.RIGHT, 10)
        main_sizer.Add(select_description_sizer, 0, wx.ALIGN_CENTER)
        main_sizer.AddSpacer(15)
        main_sizer.Add(button_sizer, 0, wx.ALIGN_RIGHT)

        self.panel.SetSizer(main_sizer)
        self.Centre()
        self.Show()

    def on_confirm(self, event):
        if not self.folder_path:
            logging.debug('No folder selected')
            wx.MessageBox('Please select a folder', 'Action required',
                          wx.OK | wx.ICON_EXCLAMATION)
            return

        self.GetParent().load_run(self.folder_path)
        self.Close(True)

    def on_cancel(self, event):
        self.Close(True)

    def on_open_file(self, event):
        dlg = wx.DirDialog(
            self, message='Choose a folder',
            defaultPath=self.currentDirectory,
            style=wx.DD_DIR_MUST_EXIST)

        if dlg.ShowModal() == wx.ID_OK:
            self.folder_path = dlg.GetPath()
            self.selected_folder_name.SetLabel(self.folder_path)
            logging.info('Selected folder path:' + self.folder_path)

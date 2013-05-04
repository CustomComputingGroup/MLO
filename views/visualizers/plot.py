import logging
import sys
import copy
import os
import io
import pickle

from time import gmtime, strftime,asctime
from multiprocessing import Process
import matplotlib as mpl
# Force matplotlib to not use any Xwindows backend.
mpl.use('Agg')
from matplotlib import pyplot
from matplotlib.ticker import LinearLocator, FormatStrFormatter, MaxNLocator
from mpl_toolkits.mplot3d import axes3d, Axes3D
from numpy import array, linspace, meshgrid, reshape, argmin
from scipy.interpolate import griddata

import HTML
import StringIO
import ho.pisa as pisa
import git

### abstract class to define plot viewers
class ImageViewer(object):

    @staticmethod
    def render(dictionary):
        raise NotImplementedError('Trial is an abstract class, '
                                  'this should not be called.')
    @staticmethod
    def save_fig(figure, filename, DPI):
        raise NotImplementedError('Trial is an abstract class, '
                                  'this should not be called.')
        
    @staticmethod
    def get_attributes(name):
        raise NotImplementedError('Trial is an abstract class, '
                                  'this should not be called.')
        
    @staticmethod
    def get_default_attributes():
        raise NotImplementedError('Trial is an abstract class, '
                                  'this should not be called.')
                                  
## This class containts 
class MLOImageViewer(ImageViewer):

    DPI = 400
    LABEL_FONT_SIZE = 10
    TITLE_FONT_SIZE = 10

    @staticmethod
    def render(input_dictionary):
        if input_dictionary["generate"]:
            dictionary = MLOImageViewer.get_default_attributes() ##this way the default view will be used if different one was not supplied
            dictionary.update(input_dictionary)
            figure = mpl.pyplot.figure()
            figure.subplots_adjust(wspace=0.35, hspace=0.35)
            counter_headers = []
            header = []
            #counters = first_trial_snapshot['counter_dict'].keys()
            #for counter in counters: ## list of names of Counters
            #    header.append('Counter "' + counter + '"')
            #    counter_headers.append(counter)
            #[trial_snapshot['counter_dict'][counter_header] for counter_header in counter_headers] 
            figure.suptitle(dictionary['graph_title'])

            rerender = True ## pointless... remove
            designSpace = dictionary['fitness'].designSpace
            npts = 100

            ### Initialize some graph points
            dimensions = len(designSpace)
            if dimensions < 3 :
                x = linspace(designSpace[0]['min'], designSpace[0]['max'], npts)
                y = linspace(designSpace[1]['min'], designSpace[1]['max'], npts)
                x, y = meshgrid(x, y)
                dictionary['x'] = reshape(x, -1)
                dictionary['y'] = reshape(y, -1)
                dictionary['z'] = array([[a, b] for (a, b) in zip(dictionary['x'],
                                                                  dictionary['y'])])

                ### Define grid
                dictionary['xi'] = linspace(designSpace[0]['min'] - 0.01,
                                            designSpace[0]['max'] + 0.01, npts)
                dictionary['yi'] = linspace(designSpace[1]['min'] - 0.01,
                                            designSpace[1]['max'] + 0.01, npts)
                dictionary['X'], dictionary['Y'] = meshgrid(dictionary['xi'],
                                                            dictionary['yi'])
            else:
                logging.info("We only support visualization of 1 and 2 dimensional spaces")
            ### Generate the graphs according to the user's selection
            if dimensions < 3 :
                if dictionary['all_graph_dicts']['Mean']['generate']:
                    MLOImageViewer.plot_MU(figure, dictionary)
                if dictionary['all_graph_dicts']['Fitness']['generate']:
                    MLOImageViewer.plot_fitness_function(figure, dictionary)
            if dictionary['all_graph_dicts']['Progression']['generate']:
                MLOImageViewer.plot_fitness_progression(figure, dictionary)
            if dimensions < 3 :
                if dictionary['all_graph_dicts']['DesignSpace']['generate']:
                    MLOImageViewer.plot_design_space(figure, dictionary)
                if dictionary['all_graph_dicts']['Cost']['generate']:
                    MLOImageViewer.plot_cost_function(figure, dictionary)
                
            ### Save and exit
            filename = str(dictionary['images_folder']) + '/plot' + str(dictionary['counter']) + '.png'
            if rerender and os.path.isfile(filename):
                os.remove(filename)
            try:
                #P = Process(target=Plot_View.save_fig, args=(figure, filename,
                #                                             Plot_View.DPI))
                MLOImageViewer.save_fig(figure, filename, MLOImageViewer.DPI)
            except:
                logging.error(
                    'MLOImageViewer could not render a plot for ' + str(name),
                    exc_info=sys.exc_info())
            mpl.pyplot.close(figure)
            #sys.exit(0) ## I let it as a reminder... do NOT uncomment this! will get the applciation to get stuck
        else: ## do not regenerate
            pass
        
    @staticmethod
    def save_fig(figure, filename, DPI):
        logging.info('Save fig ' + str(filename))
        figure.savefig(filename, dpi=DPI)

    @staticmethod
    def plot_MU(figure, d):
        graph_dict = d['all_graph_dicts']['Mean']
        plot = figure.add_subplot(int(graph_dict['position']),
                                  projection='3d', elev=60)

        ### User settings
        font_size = int(graph_dict['font size'])
        plot.set_title(graph_dict['subtitle'],
                       fontsize=MLOImageViewer.TITLE_FONT_SIZE)
        plot.set_ylabel('\n' + graph_dict['y-axis'], linespacing=3,
                        fontsize=font_size)
        plot.set_xlabel('\n' + graph_dict['x-axis'], linespacing=3,
                        fontsize=font_size)
        plot.set_zlabel('\n' + graph_dict['z-axis'], linespacing=3,
                        fontsize=font_size)
        colour_map = mpl.pyplot.get_cmap(graph_dict['colour map'])

        ### Other settings
        fitness = d['fitness']
        plot.w_xaxis.set_major_locator(MaxNLocator(5))
        plot.w_zaxis.set_major_locator(MaxNLocator(5))
        plot.w_yaxis.set_major_locator(MaxNLocator(5))
        plot.set_zlim3d(fitness.minVal, fitness.maxVal)

        ### Data
        if not (d['regressor'] is None):
            try:
                MU, S2 = d['regressor'].predict(d['z'])
                MU_z = MU
                MU_z = array([item[0] for item in MU_z])
                zi = griddata((d['x'], d['y']), MU_z,
                              (d['xi'][None, :], d['yi'][:, None]), method='nearest')

                norm = mpl.pyplot.matplotlib.colors.Normalize(MU_z)
                surf = plot.plot_surface(d['X'], d['Y'], zi, rstride=1, cstride=1,
                                         linewidth=0.05, antialiased=True,
                                         cmap=colour_map)
            except TypeError,e:
                logging.error('Could not create MU plot for the GPR plot')

    @staticmethod
    def plot_fitness_function(figure, d):
        graph_dict = d['all_graph_dicts']['Fitness']
        plot = figure.add_subplot(int(graph_dict['position']),
                                  projection='3d', elev=60)

        ### User settings
        font_size = int(graph_dict['font size'])
        plot.set_title(graph_dict['subtitle'],
                       fontsize=MLOImageViewer.TITLE_FONT_SIZE)
        plot.set_ylabel('\n' + graph_dict['y-axis'], linespacing=3,
                        fontsize=font_size)
        plot.set_xlabel('\n' + graph_dict['x-axis'], linespacing=3,
                        fontsize=font_size)
        plot.set_zlabel('\n' + graph_dict['z-axis'], linespacing=3,
                        fontsize=font_size)
        colour_map = mpl.pyplot.get_cmap(graph_dict['colour map'])

        ### Other settings
        fitness = d['fitness']
        #plot.set_tick_params(labelsize="small")
        plot.w_xaxis.set_major_locator(MaxNLocator(5))
        plot.w_zaxis.set_major_locator(MaxNLocator(5))
        plot.w_yaxis.set_major_locator(MaxNLocator(5))
        plot.set_zlim3d(fitness.minVal, fitness.maxVal)

        '''
        if fitness.rotate:
            plot1.view_init(azim=45)
            plot1.w_yaxis.set_major_formatter(
                FormatStrFormatter('%d          '))
            plot1.w_zaxis.set_major_formatter(
                FormatStrFormatter('%d          '))
            plot1.set_zlabel('\n' + fitness.z_axis_name, linespacing=5.5,
                             fontsize=Plot_View.LABEL_FONT_SIZE)
        '''

        ### Data
        #plot = Axes3D(figure, azim=-29, elev=60)
        try:            
            zReal = array([fitness.fitnessFunc(a, d['fitness_state'])[0][0][0] for a in d['z']])
        except:
            zReal = array([fitness.fitnessFunc(a)[0][0] for a in d['z']]) ###no fitness state
        ziReal = griddata((d['x'], d['y']), zReal,
                          (d['xi'][None, :], d['yi'][:, None]),
                          method='nearest')
        surfReal = plot.plot_surface(d['X'], d['Y'], ziReal, rstride=1,
                                     cstride=1, linewidth=0.05,
                                     antialiased=True, cmap=colour_map)

    @staticmethod
    def plot_fitness_progression(figure, d):
        graph_dict = d['all_graph_dicts']['Progression']
        plot = figure.add_subplot(int(graph_dict['position']))

        ### User settings
        font_size = int(graph_dict['font size'])
        plot.set_title(graph_dict['subtitle'],
                       fontsize=MLOImageViewer.TITLE_FONT_SIZE)
        plot.set_xlabel(graph_dict['x-axis'], fontsize=font_size)
        plot.set_ylabel(graph_dict['y-axis'], fontsize=font_size)

        ### Other settings
        try:
            plot.set_xlim(1,   max(10, max(d['generations_array'])))
        except ValueError, e:
            ##passing here will automatically set the limits ( not great)
            pass
        try:
            plot.set_ylim(0.0, max(d['best_fitness_array']) * 1.1)
        except ValueError, e:
            ##passing here will automatically set the limits ( not great)
            pass 
           
        ### Data
        plot.plot(d['generations_array'], d['best_fitness_array'],
                  c='red', marker='x')

    @staticmethod
    def plot_design_space(figure, d):
        graph_dict = d['all_graph_dicts']['DesignSpace']
        plot = figure.add_subplot(int(graph_dict['position']))

        ### User settings
        font_size = int(graph_dict['font size'])
        plot.set_title(graph_dict['subtitle'],
                       fontsize=MLOImageViewer.TITLE_FONT_SIZE)
        plot.set_xlabel(graph_dict['x-axis'], fontsize=font_size)
        plot.set_ylabel(graph_dict['y-axis'], fontsize=font_size)
        colour_map = mpl.cm.get_cmap(graph_dict['colour map'])
        xcolour = graph_dict['x-colour']
        ocolour = graph_dict['o-colour']

        ### Other settings
        #plot.w_xaxis.set_major_locator(MaxNLocator(5))
        #plot.w_yaxis.set_major_locator(MaxNLocator(5))

        ### Data
        fitness = d['fitness']
        if not (d['classifier'] is None):
            zClass = d['classifier'].predict(d['z'])
            zi3 = griddata((d['x'], d['y']), zClass,
                           (d['xi'][None, :], d['yi'][:, None]), method='nearest')

            levels = [k for k, v in fitness.error_labels.items()]
            levels = [l-0.1 for l in levels]
            levels.append(levels[-1]+1.0)
            CS = plot.contourf(d['X'], d['Y'], zi3, levels, cmap=colour_map)

            cbar = figure.colorbar(CS, ticks=CS.levels)
            cbar.ax.set_yticklabels(["" * (int(len(v)/2) + 13) + v
                                     for k, v in fitness.error_labels.items()],
                                    rotation='vertical',
                                    fontsize=MLOImageViewer.TITLE_FONT_SIZE)

            #
            plot_trainingset_x = [] 
            plot_trainingset_y = []
            training_set = d['classifier'].training_set
            training_labels = d['classifier'].training_labels
            
            for i in range(0, len(training_set)):
                p = training_set[i]
                plot_trainingset_x.append(p[0])
                plot_trainingset_y.append(p[1])

            if len(plot_trainingset_x) > 0:
                plot.scatter(x=plot_trainingset_x, y=plot_trainingset_y, c=ocolour, marker='x')
        
            ## plot meta-heuristic specific markers 
            ## TODO - come up with a way of adding extra colours
            #print d['all_graph_dicts']
            for key in d['meta_plot'].keys():
                data = d['meta_plot'][key]["data"]
                plot.scatter(array([item[0] for item in data]),array([item[1] for item in data]), c="white",marker=d['meta_plot'][key]["marker"])
        
    @staticmethod
    def plot_cost_function(figure, d):
        graph_dict = d['all_graph_dicts']['Cost']
        plot = figure.add_subplot(int(graph_dict['position']),
                                  projection='3d', elev=60)

        ### User settings
        font_size = int(graph_dict['font size'])
        plot.set_title(graph_dict['subtitle'],
                       fontsize=MLOImageViewer.TITLE_FONT_SIZE)
        plot.set_ylabel('\n' + graph_dict['y-axis'], linespacing=3,
                        fontsize=font_size)
        plot.set_xlabel('\n' + graph_dict['x-axis'], linespacing=3,
                        fontsize=font_size)
        plot.set_zlabel('\n' + graph_dict['z-axis'], linespacing=3,
                        fontsize=font_size)
        colour_map = mpl.pyplot.get_cmap(graph_dict['colour map'])

        ### Other settings
        fitness = d['fitness']
        #plot.set_tick_params(labelsize="small")
        plot.w_xaxis.set_major_locator(MaxNLocator(5))
        plot.w_zaxis.set_major_locator(MaxNLocator(5))
        plot.w_yaxis.set_major_locator(MaxNLocator(5))
        plot.set_zlim3d(fitness.cost_minVal, fitness.cost_maxVal)

        '''
        if fitness.rotate:
            plot1.view_init(azim=45)
            plot1.w_yaxis.set_major_formatter(
                FormatStrFormatter('%d          '))
            plot1.w_zaxis.set_major_formatter(
                FormatStrFormatter('%d          '))
            plot1.set_zlabel('\n' + fitness.z_axis_name, linespacing=5.5,
                             fontsize=Plot_View.LABEL_FONT_SIZE)
        '''

        ### Data
        #plot = Axes3D(figure, azim=-29, elev=60)

        try:            
            zReal = array([fitness.fitnessFunc(a, d['fitness_state'])[0][3][0] for a in d['z']])
        except:
            zReal = array([fitness.fitnessFunc(a)[3][0] for a in d['z']]) ###no fitness state
            
        ziReal = griddata((d['x'], d['y']), zReal,
                          (d['xi'][None, :], d['yi'][:, None]),
                          method='nearest')
        surfReal = plot.plot_surface(d['X'], d['Y'], ziReal, rstride=1,
                                     cstride=1, linewidth=0.05,
                                     antialiased=True, cmap=colour_map)
        
    @staticmethod
    def get_attributes(name):
    
        attribute_dictionary = {
            'All':         ['subtitle', 'x-axis', 'y-axis', 'z-axis',
                            'font size', 'colour map', 'x-colour', 'o-colour',
                            'position'],
            'DesignSpace':         ['subtitle', 'x-axis', 'y-axis', 'font size',
                            'colour map', 'x-colour', 'o-colour', 'position'],
            'Mean':        ['subtitle', 'x-axis', 'y-axis', 'z-axis',
                            'font size', 'colour map', 'position'],
            'Progression': ['subtitle', 'x-axis', 'y-axis', 'font size',
                            'position'],
            'Fitness':     ['subtitle', 'x-axis', 'y-axis', 'z-axis',
                            'font size', 'colour map', 'position'],
            'Cost':     ['subtitle', 'x-axis', 'y-axis', 'z-axis',
                            'font size', 'colour map', 'position']
        }
        return attribute_dictionary.get(name, None)
        
    @staticmethod
    def get_default_attributes():
        # Default values for describing graph visualization
        graph_title = 'Title'
        graph_names = ['Progression', 'Fitness', 'Mean', 'DesignSpace', "Cost"]

        graph_dict1 = {'subtitle': 'Currently Best Found Solution',
                       'x-axis': 'Iteration',
                       'y-axis': 'Fitness',
                       'font size': '10',
                       'position': '231'}
        graph_dict2 = {'subtitle': 'Fitness Function',
                       'x-axis': 'X',
                       'y-axis': 'Y',
                       'z-axis': 'Fitness',
                       'font size': '10',
                       'colour map': 'PuBu',
                       'position': '232'}
        graph_dict3 = {'subtitle': 'Regression Mean',
                       'x-axis': 'X',
                       'y-axis': 'Y',
                       'z-axis': 'Fitness',
                       'font size': '10',
                       'colour map': 'PuBuGn',
                       'position': '233'}
        graph_dict4 = {'subtitle': 'Design Space',
                       'x-axis': 'X',
                       'y-axis': 'Y',
                       'font size': '10',
                       'colour map': 'PuBu',
                       'x-colour': 'black',
                       'o-colour': 'black',
                       'position': '234'}
        graph_dict5 = {'subtitle': 'Cost Function',
                       'x-axis': 'X',
                       'y-axis': 'Y',
                       'z-axis': 'Cost',
                       'font size': '10',
                       'colour map': 'PuBu',
                       'x-colour': 'black',
                       'o-colour': 'black',
                       'position': '235'}
        all_graph_dicts = {'Progression': graph_dict1,
                           'Fitness': graph_dict2,
                           'Mean': graph_dict3,
                           'DesignSpace': graph_dict4,
                           'Cost': graph_dict5
                           }
                           
        
            
        graph_dictionary = {
            'rerendering': False,
            'graph_title': graph_title,
            'graph_names': graph_names,
            'all_graph_dicts': all_graph_dicts
        }
        
        for name in graph_names:
            graph_dictionary['all_graph_dicts'][name]['generate'] = True
                          
        return graph_dictionary
        
## This class returns a pdf containing a summary of the runs
## 

class MOMLOImageViewer(ImageViewer):

    DPI = 400
    LABEL_FONT_SIZE = 10
    TITLE_FONT_SIZE = 10

    @staticmethod
    def render(input_dictionary):
        if input_dictionary["generate"]:
            dictionary = MOMLOImageViewer.get_default_attributes() ##this way the default view will be used if different one was not supplied
            dictionary.update(input_dictionary)

            figure = mpl.pyplot.figure()
            figure.subplots_adjust(wspace=0.35, hspace=0.35)
            counter_headers = []
            header = []
            #counters = first_trial_snapshot['counter_dict'].keys()
            #for counter in counters: ## list of names of Counters
            #    header.append('Counter "' + counter + '"')
            #    counter_headers.append(counter)
            #[trial_snapshot['counter_dict'][counter_header] for counter_header in counter_headers] 
            figure.suptitle(dictionary['graph_title'])

            rerender = True ## pointless... remove
            designSpace = dictionary['fitness'].designSpace
            objectives = dictionary['fitness'].objectives
            npts = 100

            ### Initialize some graph points
            dimensions = len(designSpace)
            if dimensions < 3:
                x = linspace(designSpace[0]['min'], designSpace[0]['max'], npts)
                y = linspace(designSpace[1]['min'], designSpace[1]['max'], npts)
                x, y = meshgrid(x, y)
                dictionary['x'] = reshape(x, -1)
                dictionary['y'] = reshape(y, -1)
                dictionary['z'] = array([[a, b] for (a, b) in zip(dictionary['x'],dictionary['y'])])
                

                ### Define grid
                dictionary['xi'] = linspace(designSpace[0]['min'] - 0.01,
                                            designSpace[0]['max'] + 0.01, npts)
                dictionary['yi'] = linspace(designSpace[1]['min'] - 0.01,
                                            designSpace[1]['max'] + 0.01, npts)
                dictionary['X'], dictionary['Y'] = meshgrid(dictionary['xi'],
                                                            dictionary['yi'])
            else:
                logging.info("We only support visualization of 1 and 2 dimensional spaces")
                
            ### Generate the graphs according to the user's selection
            if dimensions < 3 :
                if dictionary['all_graph_dicts']['Mean1']['generate'] and dictionary['all_graph_dicts']['Mean2']['generate']:
                    MOMLOImageViewer.plot_MU(figure, dictionary)
                if dictionary['all_graph_dicts']['Fitness1']['generate'] and dictionary['all_graph_dicts']['Fitness2']['generate']:
                    MOMLOImageViewer.plot_fitness_function(figure, dictionary)
            if dictionary['all_graph_dicts']['Progression']['generate'] and objectives == 2:
                MOMLOImageViewer.plot_pareto_progression(figure, dictionary)
                MOMLOImageViewer.plot_speed_array(figure, dictionary)

            if dimensions < 3 :
                if dictionary['all_graph_dicts']['DesignSpace1']['generate'] and dictionary['all_graph_dicts']['DesignSpace2']['generate']:
                    MOMLOImageViewer.plot_design_space(figure, dictionary)
                #if dictionary['all_graph_dicts']['Cost']['generate']:
                   # MOMLOImageViewer.plot_cost_function(figure, dictionary)
            
            ### Save and exit
            filename = str(dictionary['images_folder']) + '/plot' + str(dictionary['counter']) + '.png'
            if rerender and os.path.isfile(filename):
                os.remove(filename)
            try:
                #P = Process(target=Plot_View.save_fig, args=(figure, filename,
                #                                             Plot_View.DPI))
                MOMLOImageViewer.save_fig(figure, filename, MOMLOImageViewer.DPI)
            except:
                logging.error(
                    'MLOImageViewer could not render a plot for ' + str(name),
                    exc_info=sys.exc_info())
            mpl.pyplot.close(figure)
            #sys.exit(0) ## I let it as a reminder... do NOT uncomment this! will get the applciation to get stuck
            
            # plot another graph
            figure1 = mpl.pyplot.figure()
            figure1.subplots_adjust(wspace=0.35, hspace=0.35)
            figure1.suptitle('Pareto Optimal Set & Particle Velocity')
            npts = 100
            if dictionary['all_graph_dicts']['Progression']['generate']:
                MOMLOImageViewer.plot_pareto_velocity(figure1, dictionary)
            filename1 = str(dictionary['images_folder']) + '/pareto' + str(dictionary['counter']) + '.png'
            if rerender and os.path.isfile(filename1):
                os.remove(filename1)
            try:
                #P = Process(target=Plot_View.save_fig, args=(figure, filename,
                #                                             Plot_View.DPI))
                MOMLOImageViewer.save_fig(figure1, filename1, MOMLOImageViewer.DPI)
            except:
                logging.error(
                    'MLOImageViewer could not render a plot for ',
                    exc_info=sys.exc_info())
            mpl.pyplot.close(figure1)
            
            # if 3 d, only plot front
            if objectives == 3:
                    figure2 = mpl.pyplot.figure()
                    figure2.subplots_adjust(wspace=0.35, hspace=0.35)
                    figure2.suptitle('Pareto Optimal Set & Particle Velocity')
                    npts = 100
                    if dictionary['all_graph_dicts']['Progression']['generate']:
                        MOMLOImageViewer.plot_pareto_progression_3d(figure2,dictionary)
                    filename2 = str(dictionary['images_folder']) + '/plot3d' + str(dictionary['counter']) + '.png'
                    if rerender and os.path.isfile(filename2):
                        os.remove(filename2)
                    try:
                        #P = Process(target=Plot_View.save_fig, args=(figure, filename,
                        #                                             Plot_View.DPI))
                        MOMLOImageViewer.save_fig(figure2, filename2, MOMLOImageViewer.DPI)
                    except:
                        logging.error(
                            'MOMLOImageViewer could not render a plot for ',
                            exc_info=sys.exc_info())
                    mpl.pyplot.close(figure2)
        else: ## do not regenerate
            pass
        
    @staticmethod
    def save_fig(figure, filename, DPI):
        logging.info('Save fig ' + str(filename))
        figure.savefig(filename, dpi=DPI)
        
        
    @staticmethod
    def plot_pareto_velocity(figure, d):
        graph_dict = {'subtitle': 'Pareto Optimal Set',
                       'x-axis': 'Fitness 1',
                       'y-axis': 'Fitness 2',
                       'z-axis': 'Fitness3',
                       'font size': '10',
                       'position': '221'}
        plot = figure.add_subplot(int(graph_dict['position']))
        #######read from pareto files

        ### User settings
        font_size = int(graph_dict['font size'])
        plot.set_title(graph_dict['subtitle'],
                       fontsize=MOMLOImageViewer.TITLE_FONT_SIZE)
        plot.set_xlabel(graph_dict['x-axis'], fontsize=font_size)
        plot.set_ylabel(graph_dict['y-axis'], fontsize=font_size)
        pareto_dict = d['pareto_dict']
        x_axis = []
        y_axis = []
        for k in pareto_dict.keys():
            if pareto_dict[k] != 'null':
                x_axis.append(pareto_dict[k]['position'][0])
                y_axis.append(pareto_dict[k]['position'][1])
        x_axis_array = array(x_axis)
        y_axis_array = array(y_axis)
        
        ### Other settings
        try:
            plot.set_xlim(min(x_axis_array),max(x_axis_array))
        except ValueError, e:
            ##passing here will automatically set the limits ( not great)
            pass
        try:
            plot.set_ylim(min(y_axis_array),max(y_axis_array))
        except ValueError, e:
            ##passing here will automatically set the limits ( not great)
            pass 
           
        plot.scatter(x=x_axis_array, y=y_axis_array, c='red', marker='.')
        
        ## Another grah
        graph_dict = {'subtitle': 'Particle Velocity',
                       'x-axis': 'Generation',
                       'y-axis': 'Velocity',
                       'font size': '10',
                       'position': '222'}
        plot = figure.add_subplot(int('222'))

        ### User settings
        font_size = int(graph_dict['font size'])
        plot.set_title(graph_dict['subtitle'],
                       fontsize=MLOImageViewer.TITLE_FONT_SIZE)
        plot.set_xlabel(graph_dict['x-axis'], fontsize=font_size)
        plot.set_ylabel(graph_dict['y-axis'], fontsize=font_size)

        ### Other settings
        speed = d['speed_array']
        x = []
        for i in range(len(speed)):
            x.append(i+1)
        try:
            plot.set_xlim(0, max(x)+1)
        except ValueError, e:
            ##passing here will automatically set the limits ( not great)
            pass
        try:
            plot.set_ylim(min(speed), max(speed) * 1.1)
        except ValueError, e:
            ##passing here will automatically set the limits ( not great)
            pass 
           
        ### Data
        try:
            plot.plot(x, speed, c='red', marker='.')
        except:
            pass
    
    @staticmethod
    def plot_pareto_progression(figure, d):
        graph_dict = d['all_graph_dicts']['Progression']
        plot = figure.add_subplot(int(graph_dict['position']))
        #######read from pareto files

        ### User settings
        font_size = int(graph_dict['font size'])
        plot.set_title(graph_dict['subtitle'],
                       fontsize=MOMLOImageViewer.TITLE_FONT_SIZE)
        plot.set_xlabel(graph_dict['x-axis'], fontsize=font_size)
        plot.set_ylabel(graph_dict['y-axis'], fontsize=font_size)
        pareto_dict = d['pareto_dict']
        x_axis = []
        y_axis = []
        for k in pareto_dict.keys():
            if pareto_dict[k] != 'null':
                x_axis.append(pareto_dict[k]['position'][0])
                y_axis.append(pareto_dict[k]['position'][1])
        x_axis_array = array(x_axis)
        y_axis_array = array(y_axis)
        
        ### Other settings
        try:
            plot.set_xlim(min(x_axis_array),max(x_axis_array))
        except ValueError, e:
            ##passing here will automatically set the limits ( not great)
            pass
        try:
            plot.set_ylim(min(y_axis_array),max(y_axis_array))
        except ValueError, e:
            ##passing here will automatically set the limits ( not great)
            pass 
           
        plot.scatter(x=x_axis_array, y=y_axis_array, c='red', marker='.')
        ### Data
        #plot.plot(x_axis_array, y_axis_array,
         #         c='red', marker='x')
    
    @staticmethod
    def plot_pareto_progression_3d(figure, d):
        graph_dict = d['all_graph_dicts']['Progression']
        #plot = figure.add_subplot(int('111'),projection = '3d',elev=20)
        plot = Axes3D(figure)
        plot.view_init(20,-200)
        #######read from pareto files

        ### User settings
        font_size = int(graph_dict['font size'])
        plot.set_title(graph_dict['subtitle'],
                       fontsize=MOMLOImageViewer.TITLE_FONT_SIZE)
        plot.set_xlabel(graph_dict['x-axis'], fontsize=font_size)
        plot.set_ylabel(graph_dict['y-axis'], fontsize=font_size)
        plot.set_zlabel(graph_dict['z-axis'], fontsize=font_size)
        pareto_dict = d['pareto_dict']
        x_axis = []
        y_axis = []
        z_axis = []
        for k in pareto_dict.keys():
            if pareto_dict[k] != 'null':
                x_axis.append(pareto_dict[k]['position'][0])
                y_axis.append(pareto_dict[k]['position'][1])
                z_axis.append(pareto_dict[k]['position'][2])
        x_axis_array = array(x_axis)
        y_axis_array = array(y_axis)
        z_axis_array = array(z_axis)
        ### Other settings
        try:
            plot.set_xlim(min(x_axis_array),max(x_axis_array))
        except ValueError, e:
            ##passing here will automatically set the limits ( not great)
            pass
        try:
            plot.set_ylim(min(y_axis_array),max(y_axis_array))
            plot.set_zlim(min(z_axis_array),max(z_axis_array))
        except ValueError, e:
            ##passing here will automatically set the limits ( not great)
            pass 
        plot.scatter(x_axis_array, y_axis_array, z_axis_array, c='red', marker='.')
         

    @staticmethod
    def plot_speed_array(figure, d):
        graph_dict = d['all_graph_dicts']['Speed']
        plot = figure.add_subplot(int(graph_dict['position']))

        ### User settings
        font_size = int(graph_dict['font size'])
        plot.set_title(graph_dict['subtitle'],
                       fontsize=MLOImageViewer.TITLE_FONT_SIZE)
        plot.set_xlabel(graph_dict['x-axis'], fontsize=font_size)
        plot.set_ylabel(graph_dict['y-axis'], fontsize=font_size)

        ### Other settings
        speed = d['speed_array']
        x = []
        for i in range(len(speed)):
            x.append(i+1)
        try:
            plot.set_xlim(0, max(x)+1)
        except ValueError, e:
            ##passing here will automatically set the limits ( not great)
            pass
        try:
            plot.set_ylim(min(speed), max(speed) * 1.1)
        except ValueError, e:
            ##passing here will automatically set the limits ( not great)
            pass 
           
        ### Data
        try:
            plot.plot(x, speed, c='red', marker='.')
        except:
            pass
    @staticmethod
    def plot_MU(figure, d):
        graph_dict_dict = {}
        graph_dict_dict['1'] = d['all_graph_dicts']['Mean1']
        graph_dict_dict['2'] = d['all_graph_dicts']['Mean2']
        for k in graph_dict_dict.keys():
            graph_dict = graph_dict_dict[k]
            
            plot = figure.add_subplot(int(graph_dict['position']), projection='3d', elev=5)
            ### User settings
            font_size = int(graph_dict['font size'])
            plot.set_title(graph_dict['subtitle'],
                fontsize=MOMLOImageViewer.TITLE_FONT_SIZE)
            plot.set_ylabel('\n' + graph_dict['y-axis'], linespacing=3,
                fontsize=font_size)
            plot.set_xlabel('\n' + graph_dict['x-axis'], linespacing=3,
                fontsize=font_size)
            plot.set_zlabel('\n' + graph_dict['z-axis'], linespacing=3,
                fontsize=font_size)
            colour_map = mpl.pyplot.get_cmap(graph_dict['colour map'])
            
            ### Other settings
            fitness = d['fitness']
            plot.w_xaxis.set_major_locator(MaxNLocator(5))
            plot.w_zaxis.set_major_locator(MaxNLocator(5))
            plot.w_yaxis.set_major_locator(MaxNLocator(5))
            plot.set_zlim3d(fitness.minVal, fitness.maxVal)
            regressor = 'regressor' + k
            ### Data
            if not (d[regressor] is None):
                try:
                        MU, S2 = d[regressor].predict(d['z'])
                        MU_z = MU
                        MU_z = array([item[0] for item in MU_z])
                        zi = griddata((d['x'], d['y']), MU_z,
                            (d['xi'][None, :], d['yi'][:, None]), method='nearest')
                        norm = mpl.pyplot.matplotlib.colors.Normalize(MU_z)
                        surf = plot.plot_surface(d['X'], d['Y'], zi, rstride=1, cstride=1,
                                 linewidth=0.05, antialiased=True,
                                 cmap=colour_map)
                except TypeError,e:
                        logging.error('Could not create MU plot for the GPR plot')

    @staticmethod
    def plot_fitness_function(figure, d):
        graph_dict_dict = {}
        graph_dict_dict['1'] = d['all_graph_dicts']['Fitness1']
        graph_dict_dict['2'] = d['all_graph_dicts']['Fitness2']
        for k in graph_dict_dict.keys():
            graph_dict = graph_dict_dict[k]
            plot = figure.add_subplot(int(graph_dict['position']),
                                      projection='3d', elev=5)

                ### User settings
            font_size = int(graph_dict['font size'])
            plot.set_title(graph_dict['subtitle'],
                           fontsize=MOMLOImageViewer.TITLE_FONT_SIZE)
            plot.set_ylabel('\n' + graph_dict['y-axis'], linespacing=3,
                            fontsize=font_size)
            plot.set_xlabel('\n' + graph_dict['x-axis'], linespacing=3,
                            fontsize=font_size)
            plot.set_zlabel('\n' + graph_dict['z-axis'], linespacing=3,
                            fontsize=font_size)
            colour_map = mpl.pyplot.get_cmap(graph_dict['colour map'])

            ### Other settings
            fitness = d['fitness']
            #plot.set_tick_params(labelsize="small")
            plot.w_xaxis.set_major_locator(MaxNLocator(5))
            plot.w_zaxis.set_major_locator(MaxNLocator(5))
            plot.w_yaxis.set_major_locator(MaxNLocator(5))
            plot.set_zlim3d(fitness.minVal, fitness.maxVal)

            ### Data
            #plot = Axes3D(figure, azim=-29, elev=60)
            zReal = {}
            try:           
                zReal[k] = array([fitness.MOFitnessFunc(a, d['fitness_state'])[0][0][int(k)-1] for a in d['z']])
            except:
                zReal[k] = array([fitness.MOFitnessFunc(a)[0][int(k)-1] for a in d['z']]) ###no fitness state
            ziReal = griddata((d['x'], d['y']), zReal[k],
                              (d['xi'][None, :], d['yi'][:, None]),
                              method='nearest')
            surfReal = plot.plot_surface(d['X'], d['Y'], ziReal, rstride=1,
                                         cstride=1, linewidth=0.05,
                                         antialiased=True, cmap=colour_map)

    @staticmethod
    def plot_design_space(figure, d):
        graph_dict_dict = {}
        graph_dict_dict['1'] = d['all_graph_dicts']['DesignSpace1']
        graph_dict_dict['2'] = d['all_graph_dicts']['DesignSpace2']
        
        for k in graph_dict_dict.keys():
            graph_dict = graph_dict_dict[k]
            plot = figure.add_subplot(int(graph_dict['position']))
            ### User settings
            font_size = int(graph_dict['font size'])
            plot.set_title(graph_dict['subtitle'],
               fontsize=MLOImageViewer.TITLE_FONT_SIZE)
            plot.set_xlabel(graph_dict['x-axis'], fontsize=font_size)
            plot.set_ylabel(graph_dict['y-axis'], fontsize=font_size)
            colour_map = mpl.cm.get_cmap(graph_dict['colour map'])
            xcolour = graph_dict['x-colour']
            ocolour = graph_dict['o-colour']
            ### Other settings
            #plot.w_xaxis.set_major_locator(MaxNLocator(5))
            #plot.w_yaxis.set_major_locator(MaxNLocator(5))
            ### Data
            fitness = d['fitness']
            classifier = 'classifier' + k
            
            if not (d[classifier] is None):
                zClass = d[classifier].predict(d['z'])
                zi3 = griddata((d['x'], d['y']), zClass,
                (d['xi'][None, :], d['yi'][:, None]), method='nearest')
                levels = [k for k, v in fitness.error_labels.items()]
                levels = [l-0.1 for l in levels]
                levels.append(levels[-1]+1.0)
                CS = plot.contourf(d['X'], d['Y'], zi3, levels, cmap=colour_map)
                cbar = figure.colorbar(CS, ticks=CS.levels)
                cbar.ax.set_yticklabels(["" * (int(len(v)/2) + 13) + v for k, v in fitness.error_labels.items()],rotation='vertical',
                            fontsize=MLOImageViewer.TITLE_FONT_SIZE)
                plot_trainingset_x = [] 
                plot_trainingset_y = []
                
                training_set = d[classifier].training_set
                training_labels = d[classifier].training_labels
                for i in range(0, len(training_set)):
                    p = training_set[i]
                    plot_trainingset_x.append(p[0])
                    plot_trainingset_y.append(p[1])

                    if len(plot_trainingset_x) > 0:
                        plot.scatter(x=plot_trainingset_x, y=plot_trainingset_y, c=ocolour, marker='x')
                
                    ## plot meta-heuristic specific markers 
                    ## TODO - come up with a way of adding extra colours
                    #print d['all_graph_dicts']
                    for key in d['meta_plot'].keys():
                        data = d['meta_plot'][key]["data"]
                        plot.scatter(array([item[0] for item in data]),array([item[1] for item in data]), c="white",marker=d['meta_plot'][key]["marker"])
        
    @staticmethod
    def plot_cost_function(figure, d):
        graph_dict = d['all_graph_dicts']['Cost']
        plot = figure.add_subplot(int(graph_dict['position']),
                                  projection='3d', elev=60)

        ### User settings
        font_size = int(graph_dict['font size'])
        plot.set_title(graph_dict['subtitle'],
                       fontsize=MLOImageViewer.TITLE_FONT_SIZE)
        plot.set_ylabel('\n' + graph_dict['y-axis'], linespacing=3,
                        fontsize=font_size)
        plot.set_xlabel('\n' + graph_dict['x-axis'], linespacing=3,
                        fontsize=font_size)
        plot.set_zlabel('\n' + graph_dict['z-axis'], linespacing=3,
                        fontsize=font_size)
        colour_map = mpl.pyplot.get_cmap(graph_dict['colour map'])

        ### Other settings
        fitness = d['fitness']
        #plot.set_tick_params(labelsize="small")
        plot.w_xaxis.set_major_locator(MaxNLocator(5))
        plot.w_zaxis.set_major_locator(MaxNLocator(5))
        plot.w_yaxis.set_major_locator(MaxNLocator(5))
        plot.set_zlim3d(fitness.cost_minVal, fitness.cost_maxVal)

        '''
        if fitness.rotate:
            plot1.view_init(azim=45)
            plot1.w_yaxis.set_major_formatter(
                FormatStrFormatter('%d          '))
            plot1.w_zaxis.set_major_formatter(
                FormatStrFormatter('%d          '))
            plot1.set_zlabel('\n' + fitness.z_axis_name, linespacing=5.5,
                             fontsize=Plot_View.LABEL_FONT_SIZE)
        '''

        ### Data
        #plot = Axes3D(figure, azim=-29, elev=60)

        try:            
            zReal = array([fitness.fitnessFunc(a, d['fitness_state'])[0][3][0] for a in d['z']])
        except:
            zReal = array([fitness.fitnessFunc(a)[3][0] for a in d['z']]) ###no fitness state
            
        ziReal = griddata((d['x'], d['y']), zReal,
                          (d['xi'][None, :], d['yi'][:, None]),
                          method='nearest')
        surfReal = plot.plot_surface(d['X'], d['Y'], ziReal, rstride=1,
                                     cstride=1, linewidth=0.05,
                                     antialiased=True, cmap=colour_map)
        
    @staticmethod
    def get_attributes(name):
    
        attribute_dictionary = {
            'All':         ['subtitle', 'x-axis', 'y-axis', 'z-axis',
                            'font size', 'colour map', 'x-colour', 'o-colour',
                            'position'],
            'DesignSpace':         ['subtitle', 'x-axis', 'y-axis', 'font size',
                            'colour map', 'x-colour', 'o-colour', 'position'],
            'Mean':        ['subtitle', 'x-axis', 'y-axis', 'z-axis',
                            'font size', 'colour map', 'position'],
            'Progression': ['subtitle', 'x-axis', 'y-axis', 'font size',
                            'position'],
            'Fitness':     ['subtitle', 'x-axis', 'y-axis', 'z-axis',
                            'font size', 'colour map', 'position'],
            'Cost':     ['subtitle', 'x-axis', 'y-axis', 'z-axis',
                            'font size', 'colour map', 'position']
        }
        return attribute_dictionary.get(name, None)
        
    @staticmethod
    def get_default_attributes():
        # Default values for describing graph visualization
        graph_title = 'Multi-Objective Optimizer Graphs'
        graph_names = ['Progression', 'Fitness1', 'Mean1', 'DesignSpace1', "Cost",'Fitness2', 'Mean2', 'DesignSpace2', "Cost"]

        graph_dict1 = {'subtitle': 'Pareto front point',
                       'x-axis': 'Fitness 1',
                       'y-axis': 'Fitness 2',
                       'z-axis': 'Fitness 3',
                       'font size': '10',
                       'position': '331'}
        graph_dict2 = {'subtitle': 'Fitness Function 1',
                       'x-axis': 'X',
                       'y-axis': 'Y',
                       'z-axis': 'Fitness',
                       'font size': '10',
                       'colour map': 'PuBu',
                       'position': '332'}
        graph_dict3 = {'subtitle': 'Regression Mean 1',
                       'x-axis': 'X',
                       'y-axis': 'Y',
                       'z-axis': 'Fitness',
                       'font size': '10',
                       'colour map': 'PuBuGn',
                       'position': '335'}
        graph_dict4 = {'subtitle': 'Design Space 1',
                       'x-axis': 'X',
                       'y-axis': 'Y',
                       'font size': '10',
                       'colour map': 'PuBu',
                       'x-colour': 'green',
                       'o-colour': 'green',
                       'position': '338'}
        graph_dict5 = {'subtitle': 'Cost Function',
                       'x-axis': 'X',
                       'y-axis': 'Y',
                       'z-axis': 'Cost',
                       'font size': '10',
                       'colour map': 'PuBu',
                       'x-colour': 'green',
                       'o-colour': 'black',
                       'position': '334'}
        graph_dict6 = {'subtitle': 'Fitness Function 2',
                       'x-axis': 'X',
                       'y-axis': 'Y',
                       'z-axis': 'Fitness',
                       'font size': '10',
                       'colour map': 'PuBu',
                       'position': '333'}
        graph_dict7 = {'subtitle': 'Regression Mean 2',
                       'x-axis': 'X',
                       'y-axis': 'Y',
                       'z-axis': 'Fitness',
                       'font size': '10',
                       'colour map': 'PuBuGn',
                       'position': '336'}
        graph_dict8 = {'subtitle': 'Design Space 2',
                       'x-axis': 'X',
                       'y-axis': 'Y',
                       'font size': '10',
                       'colour map': 'PuBu',
                       'x-colour': 'green',
                       'o-colour': 'green',
                       'position': '339'}
        graph_dict9 = {'subtitle': 'Particle Velocity',
                       'x-axis': 'Generation',
                       'y-axis': 'Velocity',
                       'font size': '10',
                       'position': '337'}
        all_graph_dicts = {'Progression': graph_dict1,
                           'Fitness1': graph_dict2,
                           'Mean1': graph_dict3,
                           'DesignSpace1': graph_dict4,
                           'Cost': graph_dict5,
                           'Fitness2': graph_dict6,
                           'Mean2': graph_dict7,
                           'DesignSpace2': graph_dict8,
                           'Speed': graph_dict9
                           }
                           
        graph_dictionary = {
            'rerendering': False,
            'graph_title': graph_title,
            'graph_names': graph_names,
            'all_graph_dicts': all_graph_dicts
        }
        
        for name in graph_names:
            graph_dictionary['all_graph_dicts'][name]['generate'] = True
                          
        return graph_dictionary
        


class MLORunReportViewer2(ImageViewer):

    DPI = 400
    LABEL_FONT_SIZE = 10
    TITLE_FONT_SIZE = 10

    @staticmethod
    def render(input_dictionary):
        if input_dictionary["generate"]:
            dictionary = MLOImageViewer.get_default_attributes() ##this way the default view will be used if different one was not supplied
            dictionary.update(input_dictionary)
            figure = mpl.pyplot.figure()
            figure.subplots_adjust(wspace=0.35, hspace=0.35)
            figure.suptitle(dictionary['graph_title'])

            rerender = True ## pointless... remove
            designSpace = dictionary['fitness'].designSpace
            npts = 100

            ### Initialize some graph points
            x = linspace(designSpace[0]['min'], designSpace[0]['max'], npts)
            y = linspace(designSpace[1]['min'], designSpace[1]['max'], npts)
            x, y = meshgrid(x, y)
            dictionary['x'] = reshape(x, -1)
            dictionary['y'] = reshape(y, -1)
            dictionary['z'] = array([[a, b] for (a, b) in zip(dictionary['x'],
                                                              dictionary['y'])])

            ### Define grid
            dictionary['xi'] = linspace(designSpace[0]['min'] - 0.01,
                                        designSpace[0]['max'] + 0.01, npts)
            dictionary['yi'] = linspace(designSpace[1]['min'] - 0.01,
                                        designSpace[1]['max'] + 0.01, npts)
            dictionary['X'], dictionary['Y'] = meshgrid(dictionary['xi'],
                                                        dictionary['yi'])

            ### Generate the graphs according to the user's selection
            if dictionary['all_graph_dicts']['Mean']['generate']:
                MLOImageViewer.plot_MU(figure, dictionary)
            if dictionary['all_graph_dicts']['Fitness']['generate']:
                MLOImageViewer.plot_fitness_function(figure, dictionary)
            if dictionary['all_graph_dicts']['Progression']['generate']:
                MLOImageViewer.plot_fitness_progression(figure, dictionary)
            if dictionary['all_graph_dicts']['DesignSpace']['generate']:
                MLOImageViewer.plot_design_space(figure, dictionary)

            ### Save and exit
            filename = str(dictionary['images_folder']) + '/plot' + str(dictionary['counter']) + '.png'
            if rerender and os.path.isfile(filename):
                os.remove(filename)
            try:
                #P = Process(target=Plot_View.save_fig, args=(figure, filename,
                #                                             Plot_View.DPI))
                MLOImageViewer.save_fig(figure, filename, MLOImageViewer.DPI)
            except:
                logging.error(
                    'MLOImageViewer could not render a plot for ' + str(name),
                    exc_info=sys.exc_info())
            mpl.pyplot.close(figure)
            #sys.exit(0) ## I let it as a reminder... do NOT uncomment this! will get the applciation to get stuck
        else: ## do not regenerate
            pass
        
    @staticmethod
    def save_fig(figure, filename, Format):
        logging.info('Save fig ' + str(filename))
        figure.savefig(filename, dpi=DPI)
        
    @staticmethod
    def get_attributes(name):
    
        attribute_dictionary = {
            'All':         ['subtitle', 'x-axis', 'y-axis', 'z-axis',
                            'font size', 'colour map', 'x-colour', 'o-colour',
                            'position'],
            'DesignSpace':         ['subtitle', 'x-axis', 'y-axis', 'font size',
                            'colour map', 'x-colour', 'o-colour', 'position'],
            'Mean':        ['subtitle', 'x-axis', 'y-axis', 'z-axis',
                            'font size', 'colour map', 'position'],
            'Progression': ['subtitle', 'x-axis', 'y-axis', 'font size',
                            'position'],
            'Fitness':     ['subtitle', 'x-axis', 'y-axis', 'z-axis',
                            'font size', 'colour map', 'position']
        }
        return attribute_dictionary.get(name, None)
        
    @staticmethod
    def get_default_attributes():
        # Default values for describing graph visualization
        graph_title = 'Title'
        graph_names = ['Progression', 'Fitness', 'Mean', 'DesignSpace']

        graph_dict1 = {'subtitle': 'Currently Best Found Solution',
                       'x-axis': 'Iteration',
                       'y-axis': 'Fitness',
                       'font size': '10',
                       'position': '221'}
        graph_dict2 = {'subtitle': 'Fitness Function',
                       'x-axis': 'X',
                       'y-axis': 'Y',
                       'z-axis': 'Fitness',
                       'font size': '10',
                       'colour map': 'PuBu',
                       'position': '222'}
        graph_dict3 = {'subtitle': 'Regression Mean',
                       'x-axis': 'X',
                       'y-axis': 'Y',
                       'z-axis': 'Fitness',
                       'font size': '10',
                       'colour map': 'PuBuGn',
                       'position': '223'}
        graph_dict4 = {'subtitle': 'Design Space',
                       'x-axis': 'X',
                       'y-axis': 'Y',
                       'font size': '10',
                       'colour map': 'PuBu',
                       'x-colour': 'black',
                       'o-colour': 'black',
                       'position': '224'}
        all_graph_dicts = {'Progression': graph_dict1,
                           'Fitness': graph_dict2,
                           'Mean': graph_dict3,
                           'DesignSpace': graph_dict4}
                           
        
            
        graph_dictionary = {
            'rerendering': False,
            'graph_title': graph_title,
            'graph_names': graph_names,
            'all_graph_dicts': all_graph_dicts
        }
        
        for name in graph_names:
            graph_dictionary['all_graph_dicts'][name]['generate'] = True
                          
        return graph_dictionary
        
##
class MLORunReportViewer(object):

    @staticmethod
    def render(dictionary):
        logging.info("Generating Report")
        ## Generate Header   
        header = ['Trial Name', 'Trial Number']  
        counter_headers = []
        timer_headers = []
        trial_snapshots = dictionary["trials_snapshots"]
        first_trial_snapshot = trial_snapshots[0]
        ## get counter names
        counters = first_trial_snapshot['counter_dict'].keys()
        for counter in counters: ## list of names of Counters
            header.append('Counter "' + counter + '"')
            counter_headers.append(counter)
        ## get timing names
        timers = first_trial_snapshot['timer_dict'].keys()
        for timer in timers: ## list of names of Counters
            header.append(timer)
            timer_headers.append(timer)
            
        htmlcode = HTML.Table(header_row=header)
        
        statistics = ["mean","std","max","min",]
        data = []
        
        for trial_snapshot in trial_snapshots:
            ## Display trial timers
            trial_name = [trial_snapshot["name"]]
            trial_no = [1]
            trial_counters = [trial_snapshot['counter_dict'][counter_header] for counter_header in counter_headers] 
            trial_timers = [trial_snapshot['timer_dict'][timer_header] for timer_header in timer_headers] 
            data.append(trial_counters + trial_timers)
            row = [HTML.TableCell(cell, bgcolor='Lime') for cell in trial_name + trial_no + trial_counters + trial_timers]
            htmlcode.rows.append(row)
            ## Display trial counters
        htmlcode = str(htmlcode)    
        
        data = array(data)
        htmlcode2 = HTML.Table(header_row=header)
        statistic_no = 0
        for statistic in statistics:
            statistic_name = statistic
            result = [str(elem) for elem in eval("data." + statistic + "(axis=0)")]
            row = [HTML.TableCell(cell, bgcolor='Lime') for cell in [statistic_name] + [str(statistic_no)] + result]
            statistic_no = statistic_no + 1
            htmlcode2.rows.append(row)
            ## append to list used to calculate statistical data
        htmlcode2 = str(htmlcode2)
        ### Save and exit
        filename = dictionary["results_folder_path"] + "/run_report.pdf"
        filename2 = dictionary["results_folder_path"] + "/run_report.html"
        
        if os.path.isfile(filename):
            os.remove(filename)
        try:
            f = file(filename, 'wb')
            pdf = pisa.CreatePDF(htmlcode + htmlcode2,f)
            if not pdf.err:
                pisa.startViewer(f)
            f.close()
        except Exception, e:
            logging.error('could not create a report for ' + str(e))
            
        if os.path.isfile(filename2):
            os.remove(filename2)
        try:
            f = file(filename2, 'wb')
            f.write(htmlcode + htmlcode2)
            f.close()
        except Exception, e:
            logging.error('could not create a report for ' + str(e))
        logging.info("Done Generating Report")

    @staticmethod
    def get_attributes(name):
        return {}
        
    @staticmethod
    def get_default_attributes():
        return {}

##This class returns a string 
##It should return either a string, a file reference or 
class MLORegressionReportViewer(object):
    @staticmethod
    def get_error_code(compare_dict):
            ## Initialize all run to be true
            color='lime'
            ErrorCode = []
            message = []
           
            # read from compare_dict
            trial_snapshot = compare_dict['trial_snapshot']
            trial_counters = compare_dict['trial_counters']
            trial_timers = compare_dict['trial_timers']
            golden_counters = compare_dict['goldenResult']['golden_counters']
            golden_timers = compare_dict['goldenResult']['golden_timers']
            counter_headers = compare_dict['goldenResult']['counter_headers']
            timer_headers = compare_dict['goldenResult']['timer_headers']
            
            ## Tell if one fails and give the errorCode and message
            if trial_snapshot['counter_dict']['fit']>trial_snapshot['max_fitness']:
                color = 'red'
                ErrorCode.append('1')
                message.append('Run out of fitness budget')
           
            if trial_snapshot['counter_dict']['g']>trial_snapshot['max_iter']: 
                color = 'red'
                ErrorCode.append('2')
                message.append('Run out of iteration budget')
            
            for i, (Counter,gCounter,Timer,gTimer,counter_header,timer_header) in enumerate(zip(trial_counters,golden_counters,trial_timers, golden_timers,counter_headers,timer_headers)):
                # compare the counters
                if int(Counter) > int(gCounter):
                    color = 'red'
                    if not '3' in ErrorCode:
                        ErrorCode.append('3')
                    outnumber = 100 * (int(Counter) - int(gCounter))/int(gCounter)
                    message.append(trial_snapshot["name"]+" -- The counter " + str(counter_header) + " outnumbers the golden result by " + str(outnumber) +"%.")
                    trial_counters[i] =str(trial_counters[i]) + ' ('+ str(outnumber) + '%)'
                # compare the timers
                if int(Timer) > int(gTimer):
                    color = 'red'
                    if not '4' in ErrorCode:
                        ErrorCode.append('4')
                    outnumber = 100 * (int(Timer) - int(gTimer))/int(gTimer)
                    message.append(trial_snapshot["name"]+" -- The timer " + str(timer_header) + " outnumbers the golden result by " + str(outnumber) +"%.")
                    trial_timers[i] =str(trial_timers[i]) + ' ('+ str(outnumber) + '%)'
            
            #output the errorCode
            ErrorC = ''
            if len(ErrorCode)==0:
                ErrorC = ErrorC + '0'
            else:
                for e in ErrorCode:
                    ErrorC = ErrorC + e + ' '
            
            
            return_dictionary = {
                                 'color': color,
                                 'ErrorCode' : [ErrorC],
                                 'trial_counters' : trial_counters,
                                 'trial_timers' : trial_timers,
                                 'message' : message
                                 }
            return return_dictionary
    
    @staticmethod
    def render(dictionary):
        logging.info("Generating Report...")
        ## Generate Header   
        header = ['Trial Name', 'Trial Number']  
        counter_headers = []
        timer_headers = []
        trial_snapshots = dictionary["trials_snapshots"]
        first_trial_snapshot = trial_snapshots[0]
        
        run_name = str(first_trial_snapshot['run_name'])
        #run_name = run_name[0:run_name.find('_')]
        ## get counter names
        counters = first_trial_snapshot['counter_dict'].keys()
        for counter in counters: ## list of names of Counters
            header.append('Counter "' + counter + '"')
            counter_headers.append(counter)
        ## get timing names
        timers = first_trial_snapshot['timer_dict'].keys()
        for timer in timers: ## list of names of Counters
            header.append(timer)
            timer_headers.append(timer)
        
        header.append('Error Code')
        htmlcode1 = HTML.Table(header_row=header)
        data = []
        failurecount=0
        failure_trial=[]
        
        # set golden file path
        goldenResultsFile = first_trial_snapshot['configuration_folder_path']+'goldenResult.txt'
        goldenResult = {}

        trial_message = []
        for trial_snapshot in trial_snapshots:
            ## Display trial timers
            trial_name = [trial_snapshot["name"]]
            trial_no = [1]
            trial_counters = [trial_snapshot['counter_dict'][counter_header] for counter_header in counter_headers] 
            trial_timers = [trial_snapshot['timer_dict'][timer_header] for timer_header in timer_headers] 
            data.append(trial_counters + trial_timers)
            
            # create golden file or read from the file
            if os.path.exists(goldenResultsFile):
                #read from file
                logging.info("Reading golden file...")
                with open(goldenResultsFile, 'rb') as outfile:
                    dict = pickle.load(outfile)
                    dict['dir'] = goldenResultsFile
                    goldenResult = dict
            else:
                #create new golden resultfile
                logging.info("A new golden result to be created")
                goldenResult = {
                               'Saved_Time': strftime("%d/%b/%Y %H:%M:%S", gmtime()),
                               'golden_counters' : trial_counters,
                               'golden_timers' : trial_timers,
                               'counter_headers' : counter_headers,
                               'timer_headers' : timer_headers,
                               'dir':goldenResultsFile
                               }
                with io.open(goldenResult['dir'], 'wb') as outfile:
                    pickle.dump(goldenResult, outfile)            

            compare_dict = {
                            'trial_snapshot' : trial_snapshot,
                            'trial_counters': trial_counters, 
                            'trial_timers' : trial_timers, 
                            'goldenResult' : goldenResult
                            }
            return_dictionary = MLORegressionReportViewer.get_error_code(compare_dict)
            color = return_dictionary['color']
            ErrorCode = return_dictionary['ErrorCode']
            trial_counters = return_dictionary['trial_counters']
            trial_timers = return_dictionary['trial_timers']
            trial_message.append(return_dictionary['message'])

            if color == 'red':
                failure_trial.append(trial_name)
                failurecount = failurecount + 1
            row = [HTML.TableCell( cell, bgcolor = color) for cell in trial_name + trial_no + trial_counters + trial_timers + ErrorCode]
            
            htmlcode1.rows.append( row )
            
        repo_message = ""
        for m in trial_message:
            if len(m)>0:
                for ms in m:
                    repo_message = repo_message + ms + '<br>'
        ## statistics
        header = ['Statistics', 'Total Trials']  
        ## get counter names
        counters = first_trial_snapshot['counter_dict'].keys()
        for counter in counters: ## list of names of Counters
            header.append('Counter "' + counter + '"')
        ## get timing names
        timers = first_trial_snapshot['timer_dict'].keys()
        for timer in timers: ## list of names of Counters
            header.append(timer)
            
        
        statistics = ["mean","std","max","min"]
        data = array(data)
        htmlcode2 = HTML.Table(header_row=header)
        total_trials=len(trial_snapshots)

        for statistic in statistics:
            statistic_name = statistic
            result = [str(elem) for elem in eval("data." + statistic + "(axis=0)")]
            row = [HTML.TableCell(cell) for cell in [statistic_name] + [str(total_trials)] + result]
            htmlcode2.rows.append(row)
            ## append to list used to calculate statistical data

        # information list in report
        headlist = []
        time=strftime("<hr><hgroup> <h3>Repo Time: %d/%b/%Y %H:%M:%S</h3></hgroup>", gmtime())
        headlist.append(time)
        
        # generate git information
        repo = git.Repo( os.getcwd() )
        headcommit = repo.head.commit
        headlist.append("Git Committer :  " + str(headcommit.committer))
        headlist.append("Commit Date : " + asctime(gmtime(headcommit.committed_date)))

        # generate run information
        headlist.append("Run Name: "+ run_name)
        
        # generate the regressor name and classifier name
        reg = str(trial_snapshots[0]['regressor'])
        cla = str(trial_snapshots[0]['classifier'])
        headlist.append("Regressor:     " + reg[34:reg.find("object")])
        headlist.append("Classifier:     " + cla[35:cla.find("object")])
        
        ## generate the trial information: fail or not?
        headlist.append( "Total Trials :  " + str(len(trial_snapshots)) )
        headlist.append( "Total Fails : " + str(failurecount))
        if failurecount>0:
            headlist.append("Fail trials: " + str(failure_trial))

        html_list = HTML.list(headlist)
        # create the report contents
        htmlcode1 = str( htmlcode1 )
        htmlcode2 = str( htmlcode2 )
        htmlcode3 = str ( html_list )
        repocontent = htmlcode3 + '<center> <b><font size="5">Trial information:</font></b> '+'<br>'+'[( ) shows how many percent it exceed the golden results]'+'<br>' + htmlcode1 +'<b>Regression Message from trials: </b>'+'<br>'+ repo_message +'</center><br>' + '<center><b><font size="5">Statistics:</font></b> ' + htmlcode2 + '</center><br>'
        
            
        ### Save and exit
        filename1 = dictionary["results_folder_path"] + "/run_report.pdf"
        filename2 = dictionary["results_folder_path"] + "/run_report.html"
        filename3 = first_trial_snapshot["run_folders_path"] + "/regression_report.html"
        
        # save pdf report 
        if os.path.isfile(filename1):
            os.remove(filename1)
        try:
            f = file(filename1, 'w')
            pdf = pisa.CreatePDF(repocontent,f)
            if not pdf.err:
                pisa.startViewer(f)
                logging.info("Viewing Report...")
                
            f.close()
        except Exception, e:
            logging.error('could not create a report for {}'.format(str(e)))
            
        # save html report 
        if os.path.isfile(filename2):
            os.remove(filename2)
        try:
            f = file(filename2, 'w')
            f.writelines(repocontent)
            f.close()

        except Exception, e:
            logging.error('could not create a report for {}'.format(str(e)))

        # save the report for all the runs in the folder
        loglist=os.listdir(first_trial_snapshot["run_folders_path"])
        reportfile=[]
        for i in range(0,len(loglist)):
            reportfile.append(first_trial_snapshot["run_folders_path"]+"/"+loglist[i]+"/run_report.html")
        f = file(filename3, 'w' )
        f.write("<hgroup> <center> <h2> MLO Report </h2> </center> </hgroup>")
        f.write("<center> (Error Code:  0 - No Error; 1 - Exceed Max Fitness, 2 - Exceed Max Generation, 3- Exceed the golden counters, 4 - Exceed the golden timers) </center>")
        for e in reportfile:
            if os.path.exists(e):
                fr = file(e,'r')
                temp = fr.readlines()
                f.writelines(temp)
                fr.close()
        f.close()

    @staticmethod
    def get_attributes(name):
        return {}
        
    @staticmethod
    def get_default_attributes():
        return {}
        
## This class containts 
class MLOTimeAware_ImageViewer(MLOImageViewer):

    DPI = 400
    LABEL_FONT_SIZE = 10
    TITLE_FONT_SIZE = 10

    @staticmethod
    def render(input_dictionary):
        if input_dictionary["generate"]:
            dictionary = MLOTimeAware_ImageViewer.get_default_attributes() ##this way the default view will be used if different one was not supplied
            dictionary.update(input_dictionary)
            figure = mpl.pyplot.figure()
            figure.subplots_adjust(wspace=0.35, hspace=0.35)
            figure.suptitle(dictionary['graph_title'])

            rerender = True ## pointless... remove
            designSpace = dictionary['fitness'].designSpace
            npts = 100

            ### Initialize some graph points
            logging.info(  "designSpace " + designSpace)
            x = linspace(designSpace[0]['min'], designSpace[0]['max'], npts)
            y = linspace(designSpace[1]['min'], designSpace[1]['max'], npts)
            x, y = meshgrid(x, y)
            
            dictionary['x'] = reshape(x, -1)
            logging.info(  "x " + dictionary['x'])
            
            dictionary['y'] = reshape(y, -1)
            logging.info(  "y " + dictionary['y'])
            
            dictionary['z'] = array([[a, b] for (a, b) in zip(dictionary['x'],
                                                              dictionary['y'])])
            logging.info(  "z " + dictionary['z'])
            ### Define grid
            dictionary['xi'] = linspace(designSpace[0]['min'] - 0.01,
                                        designSpace[0]['max'] + 0.01, npts)
            dictionary['yi'] = linspace(designSpace[1]['min'] - 0.01,
                                        designSpace[1]['max'] + 0.01, npts)
            dictionary['X'], dictionary['Y'] = meshgrid(dictionary['xi'],
                                                        dictionary['yi'])

            ### Generate the graphs according to the user's selection
            if dictionary['all_graph_dicts']['Mean']['generate']:
                MLOImageViewer.plot_MU(figure, dictionary)
            if dictionary['all_graph_dicts']['Fitness']['generate']:
                MLOImageViewer.plot_fitness_function(figure, dictionary)
            if dictionary['all_graph_dicts']['Progression']['generate']:
                MLOImageViewer.plot_fitness_progression(figure, dictionary)
            if dictionary['all_graph_dicts']['DesignSpace']['generate']:
                MLOImageViewer.plot_design_space(figure, dictionary)
            if dictionary['all_graph_dicts']['Cost']['generate']:
                MLOImageViewer.plot_cost_function(figure, dictionary)
            if dictionary['all_graph_dicts']['Cost_model']['generate']:
                MLOTimeAware_ImageViewer.plot_cost_space_model(figure, dictionary)    
            ### Save and exit
            filename = str(dictionary['images_folder']) + '/plot' + str(dictionary['counter']) + '.png'
            if rerender and os.path.isfile(filename):
                os.remove(filename)
            try:
                #P = Process(target=Plot_View.save_fig, args=(figure, filename,
                #                                             Plot_View.DPI))
                MLOTimeAware_ImageViewer.save_fig(figure, filename, MLOTimeAware_ImageViewer.DPI)
            except:
                logging.error(
                    'MLOTimeAware_ImageViewer could not render a plot for ' + str(name),
                    exc_info=sys.exc_info())
            mpl.pyplot.close(figure)
            #sys.exit(0) ## I let it as a reminder... do NOT uncomment this! will get the applciation to get stuck
        else: ## do not regenerate
            pass
        
    def plot_cost_space_model(figure, d):
        graph_dict = d['all_graph_dicts']['Cost_model']
        plot = figure.add_subplot(int(graph_dict['position']),
                                  projection='3d', elev=60)

        ### User settings
        font_size = int(graph_dict['font size'])
        plot.set_title(graph_dict['subtitle'],
                       fontsize=MLOTimeAware_ImageViewer.TITLE_FONT_SIZE)
        plot.set_ylabel('\n' + graph_dict['y-axis'], linespacing=3,
                        fontsize=font_size)
        plot.set_xlabel('\n' + graph_dict['x-axis'], linespacing=3,
                        fontsize=font_size)
        plot.set_zlabel('\n' + graph_dict['z-axis'], linespacing=3,
                        fontsize=font_size)
        colour_map = mpl.pyplot.get_cmap(graph_dict['colour map'])

        ### Other settings
        fitness = d['fitness']
        plot.w_xaxis.set_major_locator(MaxNLocator(5))
        plot.w_zaxis.set_major_locator(MaxNLocator(5))
        plot.w_yaxis.set_major_locator(MaxNLocator(5))
        plot.set_zlim3d(fitness.cost_minVal, fitness.cost_maxVal)

        ### Data
        if not (d['cost_model'] is None):
            try:
                MU = [d['cost_model'].predict(point) for point in d["z"]]
                MU = array([item[0] for item in MU])
                zi = griddata((d['x'], d['y']), MU,
                              (d['xi'][None, :], d['yi'][:, None]), method='nearest')

                norm = mpl.pyplot.matplotlib.colors.Normalize(MU_z)
                surf = plot.plot_surface(d['X'], d['Y'], zi, rstride=1, cstride=1,
                                         linewidth=0.05, antialiased=True,
                                         cmap=colour_map)
            except TypeError,e:
                logging.error('Could not create MU plot for the GPR plot')
        
    @staticmethod
    def get_attributes(name):
    
        attribute_dictionary = {
            'All':         ['subtitle', 'x-axis', 'y-axis', 'z-axis',
                            'font size', 'colour map', 'x-colour', 'o-colour',
                            'position'],
            'DesignSpace':         ['subtitle', 'x-axis', 'y-axis', 'font size',
                            'colour map', 'x-colour', 'o-colour', 'position'],
            'Mean':        ['subtitle', 'x-axis', 'y-axis', 'z-axis',
                            'font size', 'colour map', 'position'],
            'Cost_model':  ['subtitle', 'x-axis', 'y-axis', 'z-axis',
                            'font size', 'colour map', 'position'],
            'Cost':        ['subtitle', 'x-axis', 'y-axis', 'z-axis',
                            'font size', 'colour map', 'position'],
            'Progression': ['subtitle', 'x-axis', 'y-axis', 'font size',
                            'position'],
            'Fitness':     ['subtitle', 'x-axis', 'y-axis', 'z-axis',
                            'font size', 'colour map', 'position']
        }
        return attribute_dictionary.get(name, None)
        
    @staticmethod
    def get_default_attributes():
        # Default values for describing graph visualization
        graph_title = 'Title'
        graph_names = ['Progression', 'Fitness', 'Mean', 'DesignSpace', 'Cost_model', 'Cost']

        graph_dict1 = {'subtitle': 'Currently Best Found Solution',
                       'x-axis': 'Iteration',
                       'y-axis': 'Fitness',
                       'font size': '10',
                       'position': '231'}
        graph_dict2 = {'subtitle': 'Fitness Function',
                       'x-axis': 'X',
                       'y-axis': 'Y',
                       'z-axis': 'Fitness',
                       'font size': '10',
                       'colour map': 'PuBu',
                       'position': '232'}
        graph_dict3 = {'subtitle': 'Regression Mean',
                       'x-axis': 'X',
                       'y-axis': 'Y',
                       'z-axis': 'Fitness',
                       'font size': '10',
                       'colour map': 'PuBuGn',
                       'position': '233'}
        graph_dict4 = {'subtitle': 'Design Space',
                       'x-axis': 'X',
                       'y-axis': 'Y',
                       'font size': '10',
                       'colour map': 'PuBu',
                       'x-colour': 'black',
                       'o-colour': 'black',
                       'position': '234'}
        graph_dict5 = {'subtitle': 'Cost Model',
                       'x-axis': 'X',
                       'y-axis': 'Y',
                       'z-axis': 'Cost',
                       'font size': '10',
                       'colour map': 'PuBuGn',
                       'position': '235'}
        graph_dict6 = {'subtitle': 'Cost Function',
                       'x-axis': 'X',
                       'y-axis': 'Y',
                       'z-axis': 'Cost',
                       'font size': '10',
                       'colour map': 'PuBu',
                       'x-colour': 'black',
                       'o-colour': 'black',
                       'position': '236'}
        all_graph_dicts = {'Progression': graph_dict1,
                           'Fitness': graph_dict2,
                           'Mean': graph_dict3,
                           'DesignSpace': graph_dict4,
                           'Cost_model': graph_dict5,
                           'Cost': graph_dict6,
                           }
                           
        
            
        graph_dictionary = {
            'rerendering': False,
            'graph_title': graph_title,
            'graph_names': graph_names,
            'all_graph_dicts': all_graph_dicts
        }
        
        for name in graph_names:
            graph_dictionary['all_graph_dicts'][name]['generate'] = True
                          
        return graph_dictionary
 

import logging
import sys
import copy
import os

from multiprocessing import Process
import matplotlib as mpl

# Force matplotlib to not use any Xwindows backend.
mpl.use('Agg')

from matplotlib import pyplot
from matplotlib.ticker import LinearLocator, FormatStrFormatter, MaxNLocator
from mpl_toolkits.mplot3d import axes3d, Axes3D
from numpy import array, linspace, meshgrid, reshape, argmin
from scipy.interpolate import griddata


### TODO: this should be renamed
class Plot_View(object):

    DPI = 400
    LABEL_FONT_SIZE = 10
    TITLE_FONT_SIZE = 10

    ### Takes a snapshot of the values for visualization
    @staticmethod
    def snapshot(t):
        fitness = t.fitness
        best_fitness_array = copy.copy(t.best_fitness_array)
        generations_array = copy.copy(t.generations_array)
        results_folder = copy.copy(t.results_folder)
        counter = copy.copy(t.counter_dictionary[t.configuration.counter])
        name = t.get_name()

        sm = t.surrogate_model
        classifier = copy.copy(sm.classifier) if hasattr(sm, 'classifier') \
            else None
        regressor = copy.copy(sm.regressor) if hasattr(sm, 'regressor') \
            else None

        return_dictionary = {
            'fitness': fitness,
            'best_fitness_array': best_fitness_array,
            'generations_array': generations_array,
            'results_folder': results_folder,
            'counter': counter,
            'name': name,
            'classifier': classifier,
            'regressor': regressor
        }
        return return_dictionary

    @staticmethod
    def render(dictionary):
        figure = mpl.pyplot.figure()
        figure.subplots_adjust(wspace=0.35, hspace=0.35)
        figure.suptitle(dictionary['graph_title'])

        rerender = dictionary.get('rerendering')
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
            Plot_View.plot_MU(figure, dictionary)
        if dictionary['all_graph_dicts']['Fitness']['generate']:
            Plot_View.plot_fitness_function(figure, dictionary)
        if dictionary['all_graph_dicts']['Progression']['generate']:
            Plot_View.plot_fitness_progression(figure, dictionary)
        if dictionary['all_graph_dicts']['DesignSpace']['generate']:
            Plot_View.plot_design_space(figure, dictionary)

        ### Save and exit
        filename = '{}/plot{:03d}.png'.format(dictionary['results_folder'],
                                              dictionary['counter'])
        if rerender and os.path.isfile(filename):
            os.remove(filename)
        try:
            #P = Process(target=Plot_View.save_fig, args=(figure, filename,
            #                                             Plot_View.DPI))
            Plot_View.save_fig(figure, filename, Plot_View.DPI)
        except:
            logging.error(
                'Plot_View could not render a plot for {}'.format(name),
                exc_info=sys.exc_info())
        mpl.pyplot.close(figure)
        sys.exit(0)

    @staticmethod
    def save_fig(figure, filename, DPI):
        logging.debug('Save fig {}'.format(filename))
        figure.savefig(filename, dpi=DPI)

    @staticmethod
    def plot_MU(figure, d):
        graph_dict = d['all_graph_dicts']['Mean']
        plot = figure.add_subplot(int(graph_dict['position']),
                                  projection='3d', elev=60)

        ### User settings
        font_size = int(graph_dict['font size'])
        plot.set_title(graph_dict['subtitle'],
                       fontsize=Plot_View.TITLE_FONT_SIZE)
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
        MU, S2 = d['regressor'].predict(d['z'])
        MU_z = MU
        MU_z = array([item[0] for item in MU_z])
        zi = griddata((d['x'], d['y']), MU_z,
                      (d['xi'][None, :], d['yi'][:, None]), method='nearest')

        norm = mpl.pyplot.matplotlib.colors.Normalize(MU_z)
        surf = plot.plot_surface(d['X'], d['Y'], zi, rstride=1, cstride=1,
                                 linewidth=0.05, antialiased=True,
                                 cmap=colour_map)

    @staticmethod
    def plot_fitness_function(figure, d):
        graph_dict = d['all_graph_dicts']['Fitness']
        plot = figure.add_subplot(int(graph_dict['position']),
                                  projection='3d', elev=60)

        ### User settings
        font_size = int(graph_dict['font size'])
        plot.set_title(graph_dict['subtitle'],
                       fontsize=Plot_View.TITLE_FONT_SIZE)
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
        zReal = array([fitness.fitnessFunc(a)[0][0] for a in d['z']])
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
                       fontsize=Plot_View.TITLE_FONT_SIZE)
        plot.set_xlabel(graph_dict['x-axis'], fontsize=font_size)
        plot.set_ylabel(graph_dict['y-axis'], fontsize=font_size)

        ### Other settings
        plot.set_xlim(1,   max(10, max(d['generations_array'])))
        plot.set_ylim(0.0, max(d['best_fitness_array']) * 1.1)

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
                       fontsize=Plot_View.TITLE_FONT_SIZE)
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
                                fontsize=Plot_View.TITLE_FONT_SIZE)

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
    def get_attributes(name):
        attribute_dictionary = {
            'All':         ['subtitle', 'x-axis', 'y-axis', 'z-axis',
                            'font size', 'colour map', 'x-colour', 'o-colour',
                            'position'],
            'SVM':         ['subtitle', 'x-axis', 'y-axis', 'font size',
                            'colour map', 'x-colour', 'o-colour', 'position'],
            'Mean':        ['subtitle', 'x-axis', 'y-axis', 'z-axis',
                            'font size', 'colour map', 'position'],
            'Progression': ['subtitle', 'x-axis', 'y-axis', 'font size',
                            'position'],
            'Fitness':     ['subtitle', 'x-axis', 'y-axis', 'z-axis',
                            'font size', 'colour map', 'position']
        }
        return attribute_dictionary.get(name, None)

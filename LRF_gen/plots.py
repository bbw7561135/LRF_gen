import numpy
import math

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors
from mpl_toolkits.axes_grid1 import ImageGrid, Divider


def set_plot_params():
    fig_width_pt = 240.0  # Get this from LaTeX using \showthe\columnwidth
    inches_per_pt = 1.0/72.27               # Convert pt to inch
    golden_mean = (math.sqrt(5)-1.0)/2.0         # Aesthetic ratio
    fig_width = fig_width_pt*inches_per_pt  # width in inches
    fig_height = fig_width*golden_mean      # height in inches
    fig_size = [fig_width, fig_height]
    params = {'backend': 'ps', 'axes.labelsize': 10,  'text.fontsize': 10,
              'legend.fontsize': 10, 'xtick.labelsize': 10,
              'ytick.labelsize': 10, 'text.usetex': True,
              'figure.figsize': fig_size, 'font.weight': "bolder",
              'ytick.major.pad': 10, 'xtick.major.pad': 10,
              'axes.titlesize': 10, 'ps.distiller.res': 24000}
    return params


def plot_slice(cube, index, filename):

    mpl.rcParams.update(set_plot_params())

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.22, top=0.95)

    mean = numpy.mean(numpy.log(cube[index, :, :]))
    std = numpy.std(numpy.log(cube[index, :, :]))

    plate = ((numpy.log(cube[index, :, :])-mean)/std)

    cax = ax.imshow(plate, vmin=-3, vmax=3.)
    cbar = fig.colorbar(cax, ticks=[-3, -2, -1, 0, 1, 2, 3],)
    cbar.set_label(r'$\ln \rho$')
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    plt.savefig(filename)

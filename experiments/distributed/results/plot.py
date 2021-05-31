#!/usr/bin/env python3

import matplotlib.pyplot as plt
import pickle
import traceback
import argparse
import pandas as pd

from IPython.display import set_matplotlib_formats

set_matplotlib_formats('pdf')

plt.rcParams["font.family"] = "Verdana"
plt.rcParams["font.sans-serif"] = "Verdana"

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rc('text', usetex=True)

###############################################################################
###############################################################################
# fmt = '[color][line][marker]'
# see Notes in: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html
fmt = {
    'baseline': 'k-o',
    'fedavg': 'k-o',
    'count_sketch': 'g-*',
    'hadamard': 'b-.x',
    'kashin': 'r:>',
    'drive': 'C8-.v',
    'drive_plus': 'C9-.^',
    'terngrad_vanilla': 'C5--.',
}

label = {
    'fedavg': 'FedAvg (32-Bit floats)',
    'hadamard': 'Hadamard + 1-bit SQ',
    'kashin': 'Kashin + 1-Bit SQ',
    'drive': 'Drive (Hadamard)',
    'drive_plus': r'Drive$^+$ (Hadamard)',
    'terngrad_vanilla': 'TernGrad',
    'baseline': 'Baseline (32-Bit floats)',
    'count_sketch': 'Sketched-SGD',
}

clients = {
    'kmeans': [10, 100, 1000],
    'power_iteration': [10, 100, 1000],
    'cnn': [10],
}

datasets = {
    'kmeans': ['CIFAR10_DA', 'MNIST_DA'],
    'power_iteration': ['CIFAR10_DA', 'MNIST_DA'],
    'cnn': ['CIFAR10', 'CIFAR100'],
}

dataset_text = {
    'CIFAR10_DA': 'CIFAR-10',
    'CIFAR10': 'CIFAR-10',
    'CIFAR100': 'CIFAR-100',
    'MNIST_DA': 'MNIST',
    'EMNIST': 'EMNIST',
    'Shakespeare': 'Shakespeare',
    'Stack Overflow': 'Stack Overflow',
}

compression_algs = {
    'MNIST_DA': ['baseline', 'hadamard', 'kashin', 'drive', 'drive_plus'],
    'CIFAR10_DA': ['baseline', 'hadamard', 'kashin', 'drive', 'drive_plus'],
    'CIFAR10': ['fedavg', 'hadamard', 'kashin', 'drive', 'drive_plus', 'count_sketch', 'terngrad_vanilla'],
    'CIFAR100': ['fedavg', 'hadamard', 'kashin', 'drive', 'drive_plus', 'count_sketch', 'terngrad_vanilla'],
}

lr_not_cnn = {
    'kmeans': None,
    'power_iteration': 0.1,
}

lr_cnn = {
    'CIFAR10': 0.1,
    'CIFAR100': 0.05,
}

ylabels = {
    'kmeans': ['L2 Error'],
    'power_iteration': ['L2 Error'],
    'cnn': ['Accuracy'],
}

objectives = {
    'kmeans': ['L2_error'],
    'power_iteration': ['L2_error'],
    'cnn': ['testACCs'],
}


###############################################################################
###############################################################################

# left to right lables order when ncol>1
def legend_order_not_cnn(labels, handles):
    new_order = {'Baseline (32-Bit floats)': 0, 'Hadamard + 1-bit SQ': 1, 'Drive (Hadamard)': 3,
                 'Drive$^+$ (Hadamard)': 4, 'Kashin + 1-Bit SQ': 2}
    new_lables = [None] * len(labels)
    new_handles = [None] * len(handles)
    for label, handle in zip(labels, handles):
        new_lables[new_order[label]] = label
        new_handles[new_order[label]] = handle
    return new_lables, new_handles


def legend_order_cnn(labels, handles, ncols):
    if ncols == 4:
        new_order = {'FedAvg (32-Bit floats)': 0, 'TernGrad': 2, 'Hadamard + 1-bit SQ': 6, 'Sketched-SGD': 4,
                     'FetchSGD': 4, 'Kashin + 1-Bit SQ': 1,
                     'Drive$^+$ (Hadamard)': 5, 'Drive (Hadamard)': 3}

        new_lables = [None] * len(labels)
        new_handles = [None] * len(handles)
        for label, handle in zip(labels, handles):
            new_lables[new_order[label]] = label
            new_handles[new_order[label]] = handle
        return new_lables, new_handles
    else:
        print('*** Warning: notice legend order')
        return labels, handles


def get_suffix(dataset, alg, clients_num, learning_rate):
    if learning_rate is None:
        return '{}_{}_{}'.format(dataset, alg, clients_num)
    else:
        return '{}_{}_{}_{}'.format(dataset, alg, clients_num, str(learning_rate).replace('.', ''))


def saveFig(filename):
    plt.savefig('{}.pdf'.format(filename), dpi=None, facecolor='w', edgecolor='w',
                orientation='portrait', papertype=None, format='pdf',
                transparent=False, bbox_inches='tight', pad_inches=0.1,
                frameon=None, metadata=None)

    plt.savefig('{}.svg'.format(filename), dpi=None, facecolor='w', edgecolor='w',
                orientation='portrait', papertype=None, format='svg',
                transparent=False, bbox_inches=None, pad_inches=0.1,
                frameon=None, metadata=None)


def ylabel_str(y):
    if y == 1:
        return '1.0'
    if y == 0:
        return '0'
    else:  # <1
        return str(y)[1:]


def prepare_data(algorithm, suffix, objectives_key, factor=1):

    fn = 'distributed_' + algorithm + '/results_' + suffix + '.pkl'

    # Prepare the data
    with open(fn, 'rb') as f:
        results = pickle.load(f)

    epochs = results['epochs']
    objectives = results[objectives_key]

    if factor != 1:
        objectives = [x / factor for x in objectives]

    return objectives, epochs


def read_result(experiment_name, metric_col_name,
                path='/Users/yanivbenitzhak/federated-learning-research/experiments/my_gradients_are_better'):
    return pd.read_csv(f'{path}/output/results/{experiment_name}/experiment.metrics.csv', usecols=[metric_col_name],
                       squeeze=True).rolling(150).mean()


def plot_line(objectives, epochs, label, fmt, ax=None, linewidth=2, markersize=3, markevery=10, epoch=None):
    e = None
    if epoch is not None:
        e = epoch

    # Plot the data
    if ax is None:
        h = None
        plt.plot(epochs, objectives, fmt, label=label)

    else:
        if e is None:
            h, = ax.plot(epochs, objectives, fmt, label=label, linewidth=linewidth, markersize=markersize,
                         markevery=markevery)
        else:
            h, = ax.plot(epochs[e[0]:e[1]], objectives[e[0]:e[1]], fmt, label=label, linewidth=linewidth,
                         markersize=markersize, markevery=markevery)

    return h, label


def plot_da(algorithm, clients_list, markevery):
    handles = []
    labels = []
    # define new 1x4 figure with space for a legend
    fig, ax = plt.subplots(2, 4, figsize=(16, 4))
    plt.subplots_adjust(wspace=.267, hspace=.255)

    # space for a legend
    for i in range(0, 4):
        ax[1, i].axis('off')

    col = 0
    objectives_key, ylabel = objectives[algorithm][0], ylabels[algorithm][0]

    for dataset in datasets[algorithm]:
        for clients_num in clients_list:

            title = '{}\n{} clients'.format(dataset_text[dataset], clients_num)

            axis = ax[0][col]

            for compression_alg in compression_algs[dataset]:
                suffix = get_suffix(dataset, compression_alg, clients_num, lr_not_cnn[algorithm])

                # skipping unfounded file (usually due to exploding error bug, so irrelevant anyway...
                try:
                    data, x = prepare_data(algorithm, suffix, objectives_key)
                    h, l = plot_line(data, x, label[compression_alg], fmt[compression_alg], ax=axis,
                                     linewidth=linewidth, markersize=markersize, markevery=markevery,
                                     epoch=epochs[algorithm][clients_num])
                    if l not in labels:
                        handles.append(h)
                        labels.append(l)
                except Exception as e:
                    print(e)
                    print(traceback.format_exc())
                    pass

            if col == 0:
                axis.set_ylabel(ylabel, fontsize=axisLabelsFont)

            axis.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
            axis.tick_params('y', colors='k', labelsize=axisTicksFont)
            axis.tick_params('x', colors='k', labelsize=axisTicksFont)
            # if yscale == 'linear':
            #     axis.ticklabel_format(style='sci')
            #     axis.ticklabel_format(style='sci', axis='y', scilimits=(-1, 2))

            axis.set_title(title, fontsize=titleFont)

            axis.grid(linestyle='dashed')
            col = col + 1

    labels, handles = legend_order_not_cnn(labels, handles)
    fig.legend(handles=handles,  # The line objects
               labels=labels,  # The labels for each line
               loc="lower center",  # Position of legend
               bbox_to_anchor=(0.5, 0.34),
               borderaxespad=0.1,  # Small spacing around legend box
               fontsize=legendFont,
               shadow=True,
               ncol=6
               )

    saveFig('distributed_{}'.format(algorithm))
    plt.show()


###############################################################################
###############################################################################
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="""
    Plot the distributed simulation results in the paper: "DRIVE: One-bit Distributed Mean Estimation".
    Use: --plot distributed-cnn to generate Figure 3, 
         --plot distributed-power-iteration to generate Figure 4,
         --plot distributed-kmeans to generate Figure 5,
         and --plot distributed-algs-appendix to generate Figure 6.
    """,
                                     formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument('--plot', required=True, type=str,
                        choices=['distributed-cnn', 'distributed-kmeans', 'distributed-power-iteration',
                                 'distributed-algs-appendix'],
                        help='')
    args = parser.parse_args()

    ###############################################################################
    ###############################################################################

    if args.plot == 'distributed-algs-appendix':
        titleFont = 16
        axisTicksFont = 15
        legendFont = 14
        linewidth = 1
        markersize = 8

        epochs = {
            'kmeans': {10: [1, 25], 100: [1, 25], 1000: [1, 25]},
            'power_iteration': {10: [0, 50], 100: [0, 50], 1000: [0, 50]},
        }

        markevery = {'kmeans': 5, 'power_iteration': epochs['power_iteration'][10][1] // 5}

        algorithm_text = {'kmeans': 'K-means', 'power_iteration': 'Power iteration'}
        handles = []
        labels = []

        # define new 1x4 figure with space for a legend
        fig, ax = plt.subplots(2, 4, figsize=(16, 4))
        plt.subplots_adjust(wspace=.267, hspace=.255)

        # space for a legend
        for i in range(0, 4):
            ax[1, i].axis('off')

        col = 0
        clients_num = 1000

        for algorithm in ['kmeans', 'power_iteration']:
            objectives_key, ylabel = objectives[algorithm][0], ylabels[algorithm][0]

            for dataset in datasets[algorithm]:
                title = '{}\n{}'.format(algorithm_text[algorithm], dataset_text[dataset])

                axis = ax[0][col]
                axis.ticklabel_format(style='sci', axis='y', scilimits=(-1, 2))

                for compression_alg in compression_algs[dataset]:
                    suffix = get_suffix(dataset, compression_alg, clients_num, lr_not_cnn[algorithm])

                    # skipping unfounded file (usually due to exploding error bug, so irrelevant anyway...
                    try:
                        data, x = prepare_data(algorithm, suffix, objectives_key)
                        h, l = plot_line(data, x, label[compression_alg], fmt[compression_alg], ax=axis,
                                         linewidth=linewidth, markersize=markersize, markevery=markevery[algorithm],
                                         epoch=epochs[algorithm][clients_num])
                        if l not in labels:
                            handles.append(h)
                            labels.append(l)
                    except Exception as e:
                        print(e)
                        print(traceback.format_exc())
                        pass

                if col == 0:
                    axis.set_ylabel(ylabel, fontsize=axisTicksFont)

                axis.ticklabel_format(style='sci', axis='y', scilimits=(-1, 2))
                axis.tick_params('y', colors='k', labelsize=axisTicksFont)
                axis.tick_params('x', colors='k', labelsize=axisTicksFont)
                axis.set_title(title, fontsize=titleFont)
                axis.grid(linestyle='dashed')
                col = col + 1

        labels, handles = legend_order_not_cnn(labels, handles)
        fig.legend(handles=handles,  # The line objects
                   labels=labels,  # The labels for each line
                   loc="lower center",  # Position of legend
                   bbox_to_anchor=(0.5, 0.34),
                   borderaxespad=0.1,  # Small spacing around legend box
                   fontsize=legendFont,
                   shadow=True,
                   ncol=6
                   )
        saveFig('{}'.format('distributed-algs-appendix-{}-clients'.format(clients_num)))
        plt.show()

    elif args.plot in ['distributed-kmeans', 'distributed-power-iteration']:
        titleFont = 16
        axisLabelsFont = 15
        axisTicksFont = 15
        legendFont = 14
        linewidth = 1
        markersize = 8
        markevery = 10

        epochs = {
            'kmeans': {10: [1, 25], 100: [1, 25], 1000: [1, 25]},
            'power_iteration': {10: [0, 70], 100: [0, 60], 1000: [0, 100]},
        }

        if args.plot == 'distributed-kmeans':
            plot_da('kmeans', [10, 100], 5)

        elif args.plot == 'distributed-power-iteration':
            plot_da('power_iteration', [10, 100], epochs['power_iteration'][10][1] // 5)

    elif args.plot == 'distributed-cnn':
        algorithm = 'cnn'
        titleFont = 16
        axisLabelsFont = 24
        axisTicksFont = 15
        legendFont = 17
        linewidth = 1
        markersize = 8
        ncols = 4


        def markevery(x, zoom):
            if zoom == 'zoomout':
                return x // 10
            else:
                return 10


        epochs_zoom = {
            'CIFAR10': {'zoomout': [0, 150], 'zoomin': [100, 150]},
            'CIFAR100': {'zoomout': [0, 250], 'zoomin': [200, 250]},
        }
        epochs_xticks = {
            'CIFAR10': {'zoomout': [0, 75, 150], 'zoomin': [100, 125, 150]},
            'CIFAR100': {'zoomout': [0, 125, 250], 'zoomin': [200, 225, 250]},
        }
        epochs_yticks = {
            'CIFAR10': {'zoomout': [0.5, 0.75, 1], 'zoomin': [0.85, 0.9]},
            'CIFAR100': {'zoomout': [0.25, 0.5, 0.75], 'zoomin': [0.6, 0.65, 0.7]},
        }

        handles = []
        labels = []

        # define new 3x2 figure with space for a legend
        fig, ax = plt.subplots(2, 4, figsize=(16, 4))
        plt.subplots_adjust(wspace=.267, hspace=.255)

        # space for a legend
        for i in range(0, 4):
            ax[1, i].axis('off')

        row = 0
        col = 0

        for dataset in datasets[algorithm]:
            row = 0

            objectives_key = objectives[algorithm][0]
            ylabel = ylabels[algorithm][0]
            clients_num = clients[algorithm][0]

            for zoom in ['zoomout', 'zoomin']:
                axis = ax[row][col]

                for compression_alg in compression_algs[dataset]:
                    suffix = get_suffix(dataset, compression_alg, clients_num, lr_cnn[dataset])
                    # skipping unfounded file (usually due to exploding error bug, so irrelevant anyway...
                    try:
                        data, x = prepare_data(algorithm, suffix, objectives_key, factor=100)
                        h, l = plot_line(data, x, label[compression_alg], fmt[compression_alg],
                                         ax=axis, linewidth=linewidth, markersize=markersize,
                                         markevery=markevery(len(data), zoom),
                                         epoch=epochs_zoom[dataset][zoom])
                        if l not in labels:
                            handles.append(h)
                            labels.append(l)
                    except Exception as e:
                        print(e)
                        print(traceback.format_exc())
                        pass

                if col == 0:
                    axis.set_ylabel(ylabel, fontsize=axisLabelsFont)

                axis.ticklabel_format(style='plain', axis='y', scilimits=(0, 0))
                axis.set_xticks(epochs_xticks[dataset][zoom])

                if epochs_yticks[dataset][zoom]:
                    axis.set_yticks(epochs_yticks[dataset][zoom])
                    axis.set_yticklabels([ylabel_str(y) for y in epochs_yticks[dataset][zoom]])

                axis.tick_params('y', colors='k', labelsize=axisTicksFont)
                axis.tick_params('x', colors='k', labelsize=axisTicksFont)

                if zoom == 'zoomout':
                    axis.set_title(dataset_text[dataset], fontsize=titleFont)
                else:
                    axis.set_title(dataset_text[dataset] + ' - Zoom In', fontsize=titleFont)

                axis.grid(linestyle='dashed')

                col = col + 1

        labels, handles = legend_order_cnn(labels, handles, ncols)
        fig.legend(handles=handles,  # The line objects
                   labels=labels,  # The labels for each line
                   loc="lower center",  # Position of legend
                   bbox_to_anchor=(0.5, 0.25),
                   borderaxespad=0.1,  # Small spacing around legend box
                   fontsize=legendFont,
                   shadow=True,
                   ncol=ncols
                   )

        saveFig('Distributed-CNN')
        plt.show()

    else:
        raise Exception('Unknown plot type = {}'.format(args.plot))

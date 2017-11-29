import matplotlib.pyplot as plt
import matplotlib.colorbar as cb
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

__all__ = ["plot1DGrid", "plot2DGrid", "plot3DGrid", "plotScores"]


def plot1DGrid(scores, paramsToPlot, scoreLabel, vrange):
    """
    Makes a line plot of scores, over the parameter to plot
    :param scores: A list of scores, estimated using scoreModels
    :param paramsToPlot: The parameter to plot, chosen automatically by plotScores
    :param scoreLabel: The specified score label (dependent on scoring metric used)
    :param vrange: The yrange of the plot
    """
    key = list(paramsToPlot.keys())
    plt.figure(figsize=(int(round(len(paramsToPlot[key[0]]) / 1.33)), 6))
    plt.plot(np.linspace(0, max(paramsToPlot[key[0]]), len(
        paramsToPlot[key[0]])), scores, '-or')
    plt.xlabel(key[0])
    plt.xticks(np.linspace(0, max(paramsToPlot[key[0]]), len(
        paramsToPlot[key[0]])), paramsToPlot[key[0]])
    if scoreLabel is not None:
        plt.ylabel(scoreLabel)
    else:
        plt.ylabel('Score')
    if vrange is not None:
        plt.ylim(vrange[0], vrange[1])
    plt.box(on=False)
    plt.show()


def plot2DGrid(scores, paramsToPlot, keysToPlot, scoreLabel, vrange):
    """
    Plots a heatmap of scores, over the paramsToPlot
    :param scores: A list of scores, estimated using parallelizeScore
    :param paramsToPlot: The parameters to plot, chosen automatically by plotScores
    :param scoreLabel: The specified score label (dependent on scoring metric used)
    :param vrange: The visible range of the heatmap (range you wish the heatmap to be specified over)
    """
    scoreGrid = np.reshape(
        scores, (len(paramsToPlot[keysToPlot[0]]), len(paramsToPlot[keysToPlot[1]])))
    plt.figure(figsize=(int(round(len(paramsToPlot[keysToPlot[1]]) / 1.33)), int(
        round(len(paramsToPlot[keysToPlot[0]]) / 1.33))))
    if vrange is not None:
        plt.imshow(scoreGrid, cmap='jet', vmin=vrange[0], vmax=vrange[1])
    else:
        plt.imshow(scoreGrid, cmap='jet')
    plt.xlabel(keysToPlot[1])
    plt.xticks(
        np.arange(len(paramsToPlot[keysToPlot[1]])), paramsToPlot[keysToPlot[1]])
    plt.ylabel(keysToPlot[0])
    plt.yticks(
        np.arange(len(paramsToPlot[keysToPlot[0]])), paramsToPlot[keysToPlot[0]])
    if scoreLabel is not None:
        plt.title(scoreLabel)
    else:
        plt.title('Score')
    plt.colorbar()
    plt.box(on=False)
    plt.show()


def plot3DGrid(scores, paramsToPlot, keysToPlot, scoreLabel, vrange):
    """
    Plots a grid of heatmaps of scores, over the paramsToPlot
    :param scores: A list of scores, estimated using parallelizeScore
    :param paramsToPlot: The parameters to plot, chosen automatically by plotScores
    :param scoreLabel: The specified score label (dependent on scoring metric used)
    :param vrange: The visible range of the heatmap (range you wish the heatmap to be specified over)
    """
    vmin = np.min(scores)
    vmax = np.max(scores)
    scoreGrid = np.reshape(scores, (len(paramsToPlot[keysToPlot[0]]), len(
        paramsToPlot[keysToPlot[1]]), len(paramsToPlot[keysToPlot[2]])))

    nelements = scoreGrid.shape[2]
    nrows = np.floor(nelements ** 0.5).astype(int)
    ncols = np.ceil(1. * nelements / nrows).astype(int)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex='all', sharey='all', figsize=(int(round(len(
        paramsToPlot[keysToPlot[1]]) * ncols * 1.33)), int(round(len(paramsToPlot[keysToPlot[0]]) * nrows * 1.33))))
    i = 0
    for ax in axes.flat:
        if vrange is not None:
            im = ax.imshow(scoreGrid[:, :, i], cmap='jet',
                           vmin=vrange[0], vmax=vrange[1])
        else:
            im = ax.imshow(scoreGrid[:, :, i],
                           cmap='jet', vmin=vmin, vmax=vmax)
        ax.set_xlabel(keysToPlot[1])
        ax.set_xticks(np.arange(len(paramsToPlot[keysToPlot[1]])))
        ax.set_xticklabels(paramsToPlot[keysToPlot[1]])
        ax.set_ylabel(keysToPlot[0])
        ax.set_yticks(np.arange(len(paramsToPlot[keysToPlot[0]])))
        ax.set_yticklabels(paramsToPlot[keysToPlot[0]])
        ax.set_title(keysToPlot[2] + ' = ' +
                     str(paramsToPlot[keysToPlot[2]][i]))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
        i += 1
    if scoreLabel is not None:
        fig.suptitle(scoreLabel, fontsize=18)
    else:
        fig.suptitle('Score', fontsize=18)
    fig.subplots_adjust(right=0.8)
    cbar = cb.make_axes(ax, location='right', fraction=0.03)
    fig.colorbar(im, cax=cbar[0])
    plt.show()


def plotScores(scores, paramGrid, scoreLabel=None, vrange=None):
    """
    Makes a plot representing how the scores vary over the parameter grid
        Automatically decides whether to use a simple line plot (varying over one parameter)
        or a heatmap (varying over two parameters)
    :param scores: A list of scores, estimated using scoreModels
    :param paramGrid: The parameter grid specified when fitting the models using fitModels
    :param scoreLabel: The specified label (dependent on scoring metric used), e.g. 'AUC'
    :param vrange: The visible range over which to display the scores
    :return:
    """
    keys = sorted(list(paramGrid)[0].keys())
    uniqParams = dict()
    order = dict()
    for k in keys:
        order[k] = np.unique([params[k]
                              for params in list(paramGrid)], return_index=True)[1]
        uniqParams[k] = [params[k]
                         for params in np.asarray(list(paramGrid))[sorted(order[k])]]

    keysToPlot = list()
    for k in keys:
        if len(uniqParams[k]) > 1:
            keysToPlot.append(k)

    for k in keys:
        if k not in keysToPlot:
            uniqParams.pop(k, None)

    numDim = len(keysToPlot)
    if numDim > 3:
        print('Too many dimensions to plot.')
    elif numDim == 3:
        plot3DGrid(scores, uniqParams, keysToPlot, scoreLabel, vrange)
    elif numDim == 2:
        plot2DGrid(scores, uniqParams, keysToPlot, scoreLabel, vrange)
    elif numDim == 1:
        plot1DGrid(scores, uniqParams, scoreLabel, vrange)
    else:
        print('No parameters that vary in the grid')

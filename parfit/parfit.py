from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import matplotlib.colorbar as cb
import numpy as np

#------------------------ Fitting routines ------------------------#
def fitOne(model, X, y, params):
    """
    Makes one model fit using provided data and parameters
    :param model: The function name of the model you wish to pass,
        e.g. LogisticRegression [NOTE: do not instantiate with ()]
    :param X: The independent variable data
    :param y: The response variable data
    :param params: The parameters passed through to the model from the parameter grid
    :return: Returns the fitted model
    """
    m = model(**params)
    return m.fit(X, y)


def fitModels(model, paramGrid, X, y, n_jobs=-1, verbose=10):
    """
    Parallelizes fitting all models using all combinations of parameters in paramGrid on provided data.
    :param model: The function name of the model you wish to pass,
        e.g. LogisticRegression [NOTE: do not instantiate with ()]
    :param paramGrid: The ParameterGrid object created from sklearn.model_selection
    :param X: The independent variable data
    :param y: The response variable data
    :param n_jobs: Number of cores to use in parallelization (defaults to -1: all cores)
    :param verbose: The level of verbosity of reporting updates on parallel process
        Default is 10 (send an update at the completion of each job)
    :return: Returns a list of fitted models

    Example usage:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import ParameterGrid
        model = LogisticRegression
        grid = {
            'C': [1e-4, 1e-3], # regularization
            'penalty': ['l1','l2'], # penalty type
            'n_jobs': [-1] # parallelize within each fit over all cores
        }
        paramGrid = ParameterGrid(grid)
        myModels = fitModels(model, paramGrid, X_train, y_train)
    """
    return Parallel(n_jobs=n_jobs, verbose=verbose)(delayed(fitOne)(model,
                                                           X,
                                                           y,
                                                           params) for params in paramGrid)


#------------------------ Scoring routines ------------------------#
def scoreOne(model, X, y, metric, predictType):
    """
    Scores one model fit using provided metric for given model on scoring data.
    :param model: The fitted model you wish to score
    :param X: The dependent variable data you wish to use for prediction
    :param y: The ground truth independent variable data you wish to compare the predictions to
    :param metric: The metric you wish to use to score the predictions using
    :param predictType: Choice between using 'predict_proba' and 'predict' for scoring routine
    :return: Returns the score
    """
    if predictType is None:
        if 'predict_proba' in list(dir(model)):
            try:
                return metric(y, model.predict_proba(X)[:, 1])
            except:
                return metric(y, model.predict(X))
        else:
            return metric(y, model.predict(X))
    else:
        if predictType == 'predict_proba':
            try:
                return metric(y, model.predict_proba(X)[:, 1])
            except:
                print('This model/metric cannot use predict_proba. Using predict for scoring instead.')
                return metric(y, model.predict(X))
        elif predictType == 'predict':
            return metric(y, model.predict(X))


def scoreModels(models, X, y, metric=roc_auc_score, predictType=None, n_jobs=-1, verbose=10):
    """
    Parallelizes scoring all models using provided metric for given models on scoring data
    :param models: The lists of fitted models you wish to score, fitted using fitModels
    :param X: The X data ou wish to use for prediction
    :param y: The ground truth y data you wish to compare the predictions to
    :param metric: The metric you wish to use to score the predictions using
        Defaults to roc_auc_score
    :param predictType: Choice between 'predict_proba' and 'predict' for scoring routine
        Defaults to 'predict_proba' when possible
    :param n_jobs: Number of cores to use in parallelization (defaults to -1: all cores)
    :param verbose: The level of verbosity of reporting updates on parallel process
        Default is 10 (send an update at the completion of each job)
    :return: Returns a list of scores in the same order as the list of models

    Example usage:
        from sklearn.metrics import recall_score
        myScores = scoreModels(myModels, X_val, y_val, recall_score)

    """
    return Parallel(n_jobs=n_jobs, verbose=verbose)(delayed(scoreOne)(m,
                                                                      X,
                                                                      y,
                                                                      metric,
                                                                      predictType) for m in models)


def getBestModel(models, scores, bestScore='max'):
    """
    Returns the best model from the models list based on the scores from
    the scores list. Requires "best" to mean 'max' or 'min' of scores
    :param models: List of models returned by fitModels
    :param scores: List of corresponding scores returned by scoreModels
    :param bestScore: Is 'max' of scores list best, or 'min' or scores list best?
        Default: 'max'
    :return: The best model, with associated hyper-parameters
    """
    if bestScore == 'max':
        return models[np.argmax(scores)]
    elif bestScore == 'min':
        return models[np.argmin(scores)]
    else:
        print('Please choose "max" or "min" for bestScore parameter')


def getBestScore(models, scores, bestScore='max'):
    """
    Returns the score of the best model from the models list based on the scores from
    the scores lsit. Requires "best" to mean 'max' or 'min' of scores
    :param models: List of models returned by fitModels
    :param scores: List of corresponding scores returned by scoreModels
    :param bestScore: Is 'max' of scores list best, or 'min' or scores list best?
        Default: 'max'
    :return: The score of the best model
    """
    if bestScore == 'max':
        return np.max(scores)
    elif bestScore == 'min':
        return np.min(scores)
    else:
        print('Please choose "max" or "min" for bestScore parameter')


#------------------------ Plotting routines ------------------------#
def plot1DGrid(scores, paramsToPlot, scoreLabel, vrange):
    """
    Makes a line plot of scores, over the parameter to plot
    :param scores: A list of scores, estimated using scoreModels
    :param paramsToPlot: The parameter to plot, chosen automatically by plotScores
    :param scoreLabel: The specified score label (dependent on scoring metric used)
    :param vrange: The yrange of the plot
    """
    key = paramsToPlot.keys()
    plt.figure(figsize=(int(round(len(paramsToPlot[key[0]]) / 1.33)), 6))
    plt.plot(np.linspace(0, max(paramsToPlot[key[0]]), len(paramsToPlot[key[0]])), scores, '-or')
    plt.xlabel(key[0])
    plt.xticks(np.linspace(0, max(paramsToPlot[key[0]]), len(paramsToPlot[key[0]])), paramsToPlot[key[0]])
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
    scoreGrid = np.reshape(scores, (len(paramsToPlot[keysToPlot[0]]), len(paramsToPlot[keysToPlot[1]])))
    plt.figure(figsize=(int(round(len(paramsToPlot[keysToPlot[1]]) / 1.33)), int(round(len(paramsToPlot[keysToPlot[0]]) / 1.33))))
    if vrange is not None:
        plt.imshow(scoreGrid, cmap='jet', vmin=vrange[0], vmax=vrange[1])
    else:
        plt.imshow(scoreGrid, cmap='jet')
    plt.xlabel(keysToPlot[1])
    plt.xticks(np.arange(len(paramsToPlot[keysToPlot[1]])), paramsToPlot[keysToPlot[1]])
    plt.ylabel(keysToPlot[0])
    plt.yticks(np.arange(len(paramsToPlot[keysToPlot[0]])), paramsToPlot[keysToPlot[0]])
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
    scoreGrid = np.reshape(scores, (len(paramsToPlot[keysToPlot[0]]), len(paramsToPlot[keysToPlot[1]]), len(paramsToPlot[keysToPlot[2]])))

    nelements = scoreGrid.shape[2]
    nrows = np.floor(nelements ** 0.5).astype(int)
    ncols = np.ceil(1. * nelements / nrows).astype(int)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex='all',sharey='all',figsize=(int(round(len(paramsToPlot[keysToPlot[1]])*ncols*1.33)), int(round(len(paramsToPlot[keysToPlot[0]])*nrows*1.33))))
    i = 0
    for ax in axes.flat:
        if vrange is not None:
            im = ax.imshow(scoreGrid[:,:,i], cmap='jet', vmin=vrange[0], vmax=vrange[1])
        else:
            im = ax.imshow(scoreGrid[:,:,i], cmap='jet', vmin=vmin, vmax=vmax)
        ax.set_xlabel(keysToPlot[1])
        ax.set_xticks(np.arange(len(paramsToPlot[keysToPlot[1]])))
        ax.set_xticklabels(paramsToPlot[keysToPlot[1]])
        ax.set_ylabel(keysToPlot[0])
        ax.set_yticks(np.arange(len(paramsToPlot[keysToPlot[0]])))
        ax.set_yticklabels(paramsToPlot[keysToPlot[0]])
        ax.set_title(keysToPlot[2] + ' = ' + str(paramsToPlot[keysToPlot[2]][i]))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
        i += 1
    if scoreLabel is not None:
        fig.suptitle(scoreLabel,fontsize=18)
    else:
        fig.suptitle('Score', fontsize=18)
    fig.subplots_adjust(right=0.8)
    cbar = cb.make_axes(ax,location='right', fraction = 0.03)
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
        order[k] = np.unique([params[k] for params in list(paramGrid)], return_index=True)[1]
        uniqParams[k] = [params[k] for params in np.asarray(list(paramGrid))[sorted(order[k])]]

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


#------------------------ Full routine ------------------------#
def bestFit(model, paramGrid, X_train, y_train, X_val, y_val, metric=roc_auc_score, bestScore='max', predictType=None, showPlot=True, scoreLabel=None, vrange=None, n_jobs=-1, verbose=10):
    """
    Parallelizes choosing the best fitting model on the validation set, doing a grid search over the parameter space.
        Models are scored using specified metric, and user must determine whether the best score is the 'max' or 'min' of scores.
    :param model: The function name of the model you wish to pass,
        e.g. LogisticRegression [NOTE: do not instantiate with ()]
    :param paramGrid: The ParameterGrid object created from sklearn.model_selection
    :param X_train: The independent variable data used to fit the models
    :param y_train: The dependent variable data used to fit the models
    :param X_val: The independent variable data used to score the models
    :param y_val: The dependent variable data used to score the models
    :param metric: The metric used to score the models, e.g. imported from sklearn.metrics
    :param bestScore: Is 'max' of scores list best, or 'min' or scores list best? (default is 'max')
    :param predictType: Choice between 'predict_proba' and 'predict' for scoring routine
        Defaults to 'predict_proba' when possible
    :param showPlot: Whether or not to display the plot of the scores over the parameter grid
    :param scoreLabel: The specified label (dependent on scoring metric used), e.g. 'AUC'
    :param vrange: The visible range over which to display the scores
    :param n_jobs: Number of cores to use in parallelization (defaults to -1: all cores)
    :param verbose: The level of verbosity of reporting updates on parallel process
        Default is 10 (send an update at the completion of each job)
    :return: Returns a tuple including the best scoring model, the score of the best model, all models, and all scores
    """
    print('-------------FITTING MODELS-------------')
    models = fitModels(model, paramGrid, X_train, y_train, n_jobs, verbose)
    print('-------------SCORING MODELS-------------')
    scores = scoreModels(models, X_val, y_val, metric, predictType, n_jobs, verbose)
    if showPlot:
        plotScores(scores, paramGrid, scoreLabel, vrange)
    return getBestModel(models, scores, bestScore), getBestScore(models, scores, bestScore), models, scores
from joblib import Parallel, delayed
from sklearn.metrics import *
import matplotlib.pyplot as plt
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


def parallelizeFit(model, paramGrid, X, y, n_jobs=-1, verbose=10):
    """
    Parallelizes fitting all models using combinations of parameters in paramGrid on provided data.
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
        myModels = parallelizeFit(model, paramGrid, X, y)
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
    :param X: The X data you wish to use for prediction
    :param y: The ground truth y data you wish to compare the predictions to
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


def parallelizeScore(models, X, y, metric=roc_auc_score, predictType=None, n_jobs=-1, verbose=10):
    """
    Parallelizes scoring all models using provided metric for given models on scoring data
    :param models: The lists of fitted models you wish to score, fitted using parallelizeFit
    :param X: The X data ou wish to use for prediction
    :param y: The ground truth y data you wish to compare the predictions to
    :param metric: The metric yo wish to use to score the predictions using
        Defaults to roc_auc_score
    :param predictType: Choice between 'predict_proba' and 'predict' for scoring routine
        Defaults to 'predict_proba' when possible
    :param n_jobs: Number of cores to use in parallelization (defaults to -1: all cores)
    :param verbose: The level of verbosity of reporting updates on parallel process
        Default is 10 (send an update at the completion of each job)
    :return: Returns a list of scores in the same order as the list of models

    Example usage:
        from sklearn.metrics import recall_score

    """
    return Parallel(n_jobs=n_jobs, verbose=verbose)(delayed(scoreOne)(m,
                                                                      X,
                                                                      y,
                                                                      metric,
                                                                      predictType) for m in models)


#------------------------ Plotting routines ------------------------#
def plot1DGrid(scores, paramsToPlot, scoreLabel, vrange):
    """
    Makes a line plot of scores, over the parameter to plot
    :param scores: A list of scores, estimated using parallelizeScore
    :param paramsToPlot: The parameter to plot, chosen automatically by plotScores
    :param scoreLabel: The specified score label (dependent on scoring metric used)
    :param vrange: The yrange of the plot
    """
    key = paramsToPlot.keys()
    plt.figure(figsize=(int(round(len(paramsToPlot[key[0]]) / 1.33)), 6))
    plt.plot(np.linspace(0, max(paramsToPlot[key[0]]), len(paramsToPlot[key[0]])), scores, '-or')
    plt.xlabel(key[0])
    plt.xticks(np.linspace(0, max(paramsToPlot[key[0]]), len(paramsToPlot[key[0]])), paramsToPlot[key[0]])
    plt.title('Scoring')
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
    scoreGrid = np.reshape(scores, (len(paramsToPlot[keysToPlot[1]]), len(paramsToPlot[keysToPlot[0]])))
    plt.figure(figsize=(int(round(len(paramsToPlot[keysToPlot[0]]) / 1.33)), int(round(len(paramsToPlot[keysToPlot[1]]) / 1.33))))
    if vrange is not None:
        plt.imshow(scoreGrid, cmap='jet', vmin=vrange[0], vmax=vrange[1])
    else:
        plt.imshow(scoreGrid, cmap='jet')
    plt.xlabel(keysToPlot[0])
    plt.xticks(np.arange(len(paramsToPlot[keysToPlot[0]])), paramsToPlot[keysToPlot[0]])
    plt.ylabel(keysToPlot[1])
    plt.yticks(np.arange(len(paramsToPlot[keysToPlot[1]])), paramsToPlot[keysToPlot[1]])
    if scoreLabel is not None:
        plt.title(scoreLabel)
    else:
        plt.title('Scoring grid')
    plt.colorbar()
    plt.box(on=False)
    plt.show()


def plotScores(scores, paramGrid, scoreLabel=None, vrange=None):
    """
    Makes a plot representing how the scores vary over the parameter grid
        Automatically decides whether to use a simple line plot (varying over one parameter)
        or a heatmap (varying over two parameters)
    :param scores: A list of scores, estimated using parallelizeScore
    :param paramGrid: The parameter grid specified when fittingt the models using parallelizeFit
    :param scoreLabel: The specified label (dependent on scoring metric used), e.g. 'AUC'
    :param vrange: The visible range over which to display the scores
    :return:
    """
    keys = list(paramGrid)[0].keys()
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
    if numDim > 2:
        print('Too many dimensions to plot. Please select a subset to plot using the plotParams argument.')
    elif numDim == 2:
        plot2DGrid(scores, uniqParams, keysToPlot, scoreLabel, vrange)
    elif numDim == 1:
        plot1DGrid(scores, uniqParams, scoreLabel, vrange)
    else:
        print('No parameters that vary in the grid')
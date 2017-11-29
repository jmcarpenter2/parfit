from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

__all__ = ["scoreModels", "scoreOne", "getBestModel", "getBestScore"]


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
                print(
                    'This model/metric cannot use predict_proba. Using predict for scoring instead.')
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

from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

__all__ = ["scoreModels", "scoreOne", "getBestModel", "getBestScore"]


def scoreOne(model, X, y, metric, predict_proba):
    """
    Scores one model fit using provided metric for given model on scoring data.
    :param model: The fitted model you wish to score
    :param X: The dependent variable data you wish to use for prediction
    :param y: The ground truth independent variable data you wish to compare the predictions to
    :param metric: The metric you wish to use to score the predictions using
    :param predict_proba: Choice between using 'predict_proba' and 'predict' for scoring routine.
        Default True means predict_proba and False means predict
    :return: Returns the score
    """
    if predict_proba:
        try:
            return metric(y, model.predict_proba(X)[:, 1])
        except:
            print('This model/metric cannot use predict_proba. Using predict for scoring instead.')
            return metric(y, model.predict(X))
    else:
        return metric(y, model.predict(X))


def scoreModels(models, X, y, metric=roc_auc_score, predict_proba=True, n_jobs=-1, verbose=10):
    """
    Parallelizes scoring all models using provided metric for given models on scoring data
    :param models: The lists of fitted models you wish to score, fitted using fitModels
    :param X: The X data ou wish to use for prediction
    :param y: The ground truth y data you wish to compare the predictions to
    :param metric: The metric you wish to use to score the predictions using
        Defaults to roc_auc_score
    :param predict_proba: Choice between 'predict_proba' and 'predict' for scoring routine
        Default True means predict_proba and False means predict
    :param n_jobs: Number of cores to use in parallelization (defaults to -1: all cores)
    :param verbose: The level of verbosity of reporting updates on parallel process
        Default is 10 (send an update at the completion of each job)
    :return: Returns a list of scores in the same order as the list of models

    Example usage:
        from sklearn.metrics import recall_score
        myScores = scoreModels(myModels, X_val, y_val, recall_score)

    """
    return Parallel(n_jobs=n_jobs, verbose=np.ceil(verbose/10))(delayed(scoreOne)(m,
                                                                      X,
                                                                      y,
                                                                      metric,
                                                                      predict_proba) for m in models)


def getBestModel(models, scores, greater_is_better=True):
    """
    Returns the best model from the models list based on the scores from
    the scores list. "Best" means 'max' or 'min' of scores, dependent on greater_is_better
    :param models: List of models returned by fitModels or parameters returned by crossvalModels
    :param scores: List of corresponding scores returned by scoreModels
    :param greater_is_better: Choice between optimizing for greater scores or lesser scores
        Default True means greater and False means lesser
    :return: The best model, with associated hyper-parameters
    """
    if greater_is_better:
        return models[np.argmax(scores)]
    else:
        return models[np.argmin(scores)]


def getBestScore(scores, greater_is_better=True):
    """
    Returns the score of the best model from the models list based on the scores from
    the scores lsit. "Best" means 'max' or 'min' of scores, dependent on greater_is_better
    :param scores: List of corresponding scores returned by scoreModels
    :param greater_is_better: Choice between optimizing for greater scores or lesser scores
        Default True means greater and False means lesser
    :return: The score of the best model
    """
    if greater_is_better:
        return np.max(scores)
    else:
        return np.min(scores)

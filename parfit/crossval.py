from joblib import Parallel, delayed
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
import numpy as np
from .fit import fitOne
from .score import scoreOne
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


__all__ = ["crossvalModels", "crossvalOne"]

def crossvalOne(model, X, y, params, nfolds, metric=roc_auc_score, predict_proba=True, n_jobs=-1, verbose=1):
    """
    Makes one cross-validation model fit-score run using provided data and parameters
    :param model: The instantiated model you wish to pass, e.g. LogisticRegression()
    :param X: The independent variable data
    :param y: The response variable data
    :param params: The parameters passed through to the model from the parameter grid
    :param nfolds: The number of folds you wish to use for cross-validation
    :param metric: The metric you wish to use to score the crossval predictions using
    :param predict_proba: Choice between using 'predict_proba' and 'predict' for scoring routine.
        Default True means predict_proba and False means predict
    :param n_jobs: Number of cores to use in parallelization (defaults to -1: all cores)
    :param verbose: The level of verbosity of reporting updates on parallel process
        Default is 10 (send an update at the completion of each job)
    :return: Returns the mean of the cross-validation scores
    """
    kf = KFold(n_splits=nfolds)
    train_indices, test_indices = zip(*kf.split(X))
    fitted_models = Parallel(n_jobs=n_jobs, verbose=verbose)(delayed(fitOne)(model,
                                                                      np.asarray(X)[train_index],
                                                                      np.asarray(y)[train_index],
                                                                      params) for train_index in train_indices)
    scores = Parallel(n_jobs=n_jobs, verbose=0)(delayed(scoreOne)(fitted_model,
                                                                        np.asarray(X)[test_index],
                                                                        np.asarray(y)[test_index],
                                                                        metric,
                                                                        predict_proba) for fitted_model,test_index
                                                                        in zip(fitted_models, test_indices))

    return np.mean(scores)


def crossvalModels(model, paramGrid, X, y, nfolds=5, metric=roc_auc_score, predict_proba=True, n_jobs=-1, verbose=10):
    """
    Parallelizes fitting and scoring all cross-validation models using all combinations of parameters in paramGrid on provided data.
    :param model: The instantiated model you wish to pass, e.g. LogisticRegression()
    :param paramGrid: The ParameterGrid object created from sklearn.model_selection
    :param X: The independent variable data
    :param y: The response variable data
    :param nfolds: The number of folds you wish to use for cross-validation
    :param metric: The metric you wish to use to score the crossval predictions using
    :param predict_proba: Choice between using 'predict_proba' and 'predict' for scoring routine.
        Default True means predict_proba and False means predict
    :param n_jobs: Number of cores to use in parallelization (defaults to -1: all cores)
    :param verbose: The level of verbosity of reporting updates on parallel process
        Default is 10 (send an update at the completion of each job)
    :return: Returns the grid of mean of cross-validation scores for the specified parameters,
        and the associated paramGrid
    """
    return Parallel(n_jobs=n_jobs, verbose=verbose)(delayed(crossvalOne)(model,
                                                                         X,
                                                                         y,
                                                                         params,
                                                                         nfolds,
                                                                         metric,
                                                                         predict_proba,
                                                                         n_jobs,
                                                                         np.ceil(verbose/10)) for params in paramGrid), \
           list(paramGrid)
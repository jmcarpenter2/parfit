from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

__all__ = ["fitModels", "fitOne"]


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

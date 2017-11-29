from sklearn.metrics import roc_auc_score
from .plot import plotScores
from .score import scoreModels, getBestScore, getBestModel
from .fit import fitModels
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

__all__ = ["bestFit"]


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
    scores = scoreModels(models, X_val, y_val, metric,
                         predictType, n_jobs, verbose)
    if showPlot:
        plotScores(scores, paramGrid, scoreLabel, vrange)
    return getBestModel(models, scores, bestScore), getBestScore(models, scores, bestScore), models, scores

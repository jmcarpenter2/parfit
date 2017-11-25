# parfit
A package for parallelizing the fit and flexibly scoring of sklearn machine learning models, with optional plotting routines.

You can **pip install parfit** and then import into your code using *import parfit.parfit as pf*. Once imported, you can use pf.bestFit() or other functions freely.

## Notes
1. You can either use **bestFit()** to automate the steps of the process, and optionally plot the scores over the parameter grid, OR you can do each step in order [*fitModels()* -> *scoreModels()* -> *plotScores()* -> *getBestModel()* -> *getBestScore()*]

2. Be sure to specify ALL parameters in the ParameterGrid, even the ones you are not searching over.

3. Each function is well-documented in the .py file. In Jupyter Notebooks, you can see the docs by pressing Shift+Tab(x3). Also, the documentation is listed below.

4. This package is designed for use with sklearn machine learning models, but in theory will work with any model that has a .fit(X,y) function

5. The plotScores() function will only work for up to a 3D parameterGrid object. That is, you can only view the scores of a grid varying over three parameters. Other parameters which do not vary can still be set.

## Docs
### def **bestFit**(model, paramGrid, X_train, y_train, X_val, y_val, metric=roc_auc_score, bestScore='max', predictType=None, showPlot=True, scoreLabel=None, vrange=None, n_jobs=-1, verbose=10):
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

### def **fitModels**(model, paramGrid, X, y, n_jobs=-1, verbose=10):
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
    
### def **scoreModels**(models, X, y, metric=roc_auc_score, predictType=None, n_jobs=-1, verbose=10):
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
    
### def **getBestModel**(models, scores, bestScore='max'):
    """
    Returns the best model from the models list based on the scores from
    the scores list. Requires "best" to mean 'max' or 'min' of scores
    :param models: List of models returned by fitModels
    :param scores: List of corresponding scores returned by scoreModels
    :param bestScore: Is 'max' of scores list best, or 'min' or scores list best?
        Default: 'max'
    :return: The best model from the models list
    
### def **getBestScore**(models, scores, bestScore='max'):
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


### def **plotScores**(scores, paramGrid, scoreLabel=None, vrange=None):
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


# Documentation

## 1. `bestFit`

```python
def bestFit(model, paramGrid, X_train, y_train, X_val=None, y_val=None, nfolds=5,
	        metric=roc_auc_score, greater_is_better=True, predict_proba=True, 
	        showPlot=True, scoreLabel=None, vrange=None, n_jobs=-1, verbose=10)
```

Parallelizes choosing the best fitting model on the validation set, doing a grid search over the parameter space. 
Models are scored using specified metric. Optional visualization of the scores.

**Parameters:**

`model`: The instantiated model you wish to pass, e.g. LogisticRegression()

`paramGrid`: The ParameterGrid object created from sklearn.model_selection

`X_train`: The independent variable data used to fit the models

`y_train`: The dependent variable data used to fit the models

`X_val`: The independent variable data used to score the models (default None)

`y_val`: The dependent variable data used to score the models (default None)

`nfolds`: The cross-validation number of folds, used if a validation set is not specified

`metric`: The metric used to score the models, e.g. imported from sklearn.metrics

`greater_is_better`: Choice between optimizing for greater scores or lesser scores
Default True means greater and False means lesser

`predict_proba`: Choice between 'predict_proba' and 'predict' for scoring routine
Default True means predict_proba and False means predict

`showPlot`: Whether or not to display the plot of the scores over the parameter grid

`scoreLabel`: The specified label (dependent on scoring metric used), e.g. 'AUC'

`vrange`: The visible range over which to display the scores

`n_jobs`: Number of cores to use in parallelization (defaults to -1: all cores)

`verbose`: The level of verbosity of reporting updates on parallel process Default is 10 (send an update at the completion of each job)

**returns:**

Returns a tuple including the best scoring model, the score of the best model, all models, and all scores




## 2.`crossvalModels`
```python 
def crossvalOne(model, X, y, params, nfolds, metric=roc_auc_score, 
            predict_proba=True, n_jobs=-1, verbose=1)
```

Parallelizes fitting and scoring all cross-validation models using all combinations of parameters in paramGrid on provided data.

**Parameters**:

`model`: The instantiated model you wish to pass, e.g. LogisticRegression()

`paramGrid`: The ParameterGrid object created from sklearn.model_selection

`X`: The independent variable data

`y`: The response variable data

`nfolds`: The number of folds you wish to use for cross-validation

`metric`: The metric you wish to use to score the predictions using Defaults to roc_auc_score

`predict_proba`: Choice between 'predict_proba' and 'predict' for scoring routine
Default True means predict_proba and False means predict

`n_jobs`: Number of cores to use in parallelization (defaults to -1: all cores)

`verbose`: The level of verbosity of reporting updates on parallel process Default is 10 (send an update at the completion of each job)

**returns**: 

Returns the grid of mean of cross-validation scores for the specified parameters, and the associated paramGrid

**Example usage**:

```python
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import ParameterGrid
    model = LogisticRegression()
    grid = {
        'C': [1e-4, 1e-3], # regularization
        'penalty': ['l1','l2'], # penalty type
        'n_jobs': [-1] # parallelize within each fit over all cores
    }
    paramGrid = ParameterGrid(grid)
    myScores, myModels = crossvalModels(model, paramGrid, X_train, y_train, nfolds=5)
```



## 3.`fitModels`

```python
def fitModels(model, paramGrid, X, y, n_jobs=-1, verbose=10)
```

Parallelizes fitting all models using all combinations of parameters in paramGrid on provided data.

**Parameters**:

`model`: The instantiated model you wish to pass, e.g. LogisticRegression()

`paramGrid`: The ParameterGrid object created from sklearn.model_selection

`X`: The independent variable data

`y`: The response variable data

`n_jobs`: Number of cores to use in parallelization (defaults to -1: all cores)

`verbose`: The level of verbosity of reporting updates on parallel process Default is 10 (send an update at the completion of each job)

**returns**: 

Returns a list of fitted models

**Example usage**:

```python
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import ParameterGrid
    model = LogisticRegression()
    grid = {
        'C': [1e-4, 1e-3], # regularization
        'penalty': ['l1','l2'], # penalty type
        'n_jobs': [-1] # parallelize within each fit over all cores
    }
    paramGrid = ParameterGrid(grid)
    myModels = fitModels(model, paramGrid, X_train, y_train)
```



## 4.`scoreModels`

```python
def scoreModels(models, X, y, metric=roc_auc_score, predictType=None, n_jobs=-1, verbose=10)
```

Parallelizes scoring all models using provided metric for given models on scoring data.

**Parameters**:

`models`: The lists of fitted models you wish to score, fitted using fitModels

`X`: The X data you wish to use for prediction

`y`: The ground truth y data you wish to compare the predictions to

`metric`: The metric you wish to use to score the predictions using Defaults to roc_auc_score

`predict_proba`: Choice between 'predict_proba' and 'predict' for scoring routine
Default True means predict_proba and False means predict

`n_jobs`: Number of cores to use in parallelization (defaults to -1: all cores)

`verbose`: The level of verbosity of reporting updates on parallel process Default is 10 (send an update at the completion of each job)

**return**: 

Returns a list of scores in the same order as the list of models

**Example usage**:
```python
    from sklearn.metrics import recall_score
    myScores = scoreModels(myModels, X_val, y_val, recall_score)
```



## 5.`getBestModel`

```python
def getBestModel(models, scores, greater_is_better=True)
```
Returns the best model from the models list based on the scores from
the scores list. "Best" means 'max' or 'min' of scores, dependent on greater_is_better

**Parameters**:

`models`: List of models returned by fitModels

`scores`: List of corresponding scores returned by scoreModels

`greater_is_better`: Choice between optimizing for greater scores or lesser scores
Default True means greater and False means lesser

**return**: 

The best model from the models list.




## 6.`bestScore`

```python
def getBestScore(scores, greater_is_better=True)
```

Returns the score of the best model from the models list based on the scores from
the scores lsit. "Best" means 'max' or 'min' of scores, dependent on greater_is_better

**Parameters**:

`scores`: List of corresponding scores returned by scoreModels

`greater_is_better`: Choice between optimizing for greater scores or lesser scores
Default True means greater and False means lesser

**returns**:

The score of the best model




## 7.`plotScores`

```python
def plotScores(scores, paramGrid, scoreLabel=None, vrange=None)
```
Makes a plot representing how the scores vary over the parameter grid.
Automatically decides whether to use a simple line plot (varying over one parameter) or a heatmap (varying over two/three parameters).

**Parameters**:

`scores`: A list of scores, estimated using scoreModels

`paramGrid`: The parameter grid specified when fitting the models using fitModels

`scoreLabel`: The specified label (dependent on scoring metric used), e.g. 'AUC'

`vrange`: The visible range over which to display the scores

**returns**:

Displays a plot



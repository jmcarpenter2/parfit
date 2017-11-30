	# Documentation

## 1.  `bestFit`



```python
def bestFit(model, paramGrid, X_train, y_train, X_val, y_val, metric=roc_auc_score, bestScore='max', predictType=None, showPlot=True, scoreLabel=None, vrange=None, n_jobs=-1, verbose=10)
```

Parallelizes choosing the best fitting model on the validation set, doing a grid search over the parameter space.Models are scored using specified metric, and user must determine whether the best score is the 'max' or 'min' of scores.

**Parameters: **

`model`: The function name of the model you wish to pass, e.g. LogisticRegression  

*NOTE: do not instantiate with ()*


`paramGrid`: The ParameterGrid object created from sklearn.model_selection

`X_train`: The independent variable data used to fit the models

`y_train`: The dependent variable data used to fit the models

`X_val`: The independent variable data used to score the models

`y_val`: The dependent variable data used to score the models

`metric`: The metric used to score the models, e.g. imported from sklearn.metrics

`bestScore`: Is 'max' of scores list best, or 'min' or scores list best? (default is'max') 

`predictType`: Choice between 'predict_proba' and 'predict' for scoring routine Defaults to 'predict_proba' when possible

`showPlot`: Whether or not to display the plot of the scores over the parameter grid

`scoreLabel`: The specified label (dependent on scoring metric used), e.g. 'AUC'

`vrange`: The visible range over which to display the scores

`n_jobs`: Number of cores to use in parallelization (defaults to -1: all cores)

`verbose`: The level of verbosity of reporting updates on parallel process Default is 10 (send an update at the completion of each job)

**returns: **

Returns a tuple including the best scoring model, the score of the best model, all models, and all scores



## 2. `fitModels`



```python
def fitModels(model, paramGrid, X, y, n_jobs=-1, verbose=10)
```

Parallelizes fitting all models using all combinations of parameters in paramGrid on provided data.

**Parameters**:

`model`: The function name of the model you wish to pass, 
e.g. LogisticRegression 

*NOTE: do not instantiate with ()*

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
    model = LogisticRegression
    grid = {
        'C': [1e-4, 1e-3], # regularization
        'penalty': ['l1','l2'], # penalty type
        'n_jobs': [-1] # parallelize within each fit over all cores
    }
    paramGrid = ParameterGrid(grid)
    myModels = fitModels(model, paramGrid, X_train, y_train)
```



## 3. `scoreModels`



```python
def scoreModels(models, X, y, metric=roc_auc_score, predictType=None, n_jobs=-1, verbose=10)
```

Parallelizes scoring all models using provided metric for given models on scoring data.

**Parameters**:

`models`: The lists of fitted models you wish to score, fitted using fitModels

`X`: The X data ou wish to use for prediction

`y`: The ground truth y data you wish to compare the predictions to

`metric`: The metric you wish to use to score the predictions using Defaults to roc_auc_score

`predictType`: Choice between 'predict_proba' and 'predict' for scoring routine Defaults to 'predict_proba' when possible

`n_jobs`: Number of cores to use in parallelization (defaults to -1: all cores)

`verbose`: The level of verbosity of reporting updates on parallel process Default is 10 (send an update at the completion of each job)

**return**: 

Returns a list of scores in the same order as the list of models

**Example usage**:
```python
    from sklearn.metrics import recall_score
    myScores = scoreModels(myModels, X_val, y_val, recall_score)
```



## 4. `getBestModel`



```python
def getBestModel(models, scores, bestScore='max')
```
Returns the best model from the models list based on the scores from
the scores list. Requires "best" to mean 'max' or 'min' of scores.

**Parameters**:

`models`: List of models returned by fitModels

`scores`: List of corresponding scores returned by scoreModels

`bestScore`: Is 'max' of scores list best, or 'min' or scores list best? (Default: 'max')

**return**: 

The best model from the models list.



## 5. `bestScore`



```python
def getBestScore(models, scores, bestScore='max')
```

Returns the score of the best model from the models list based on the scores from
the scores lsit. Requires "best" to mean 'max' or 'min' of scores

**Parameters**:

`models`: List of models returned by fitModels

`scores`: List of corresponding scores returned by scoreModels

`bestScore`: Is 'max' of scores list best, or 'min' or scores list best? Default: 'max'

**returns**:

The score of the best model




## 6. `plotScores`



```python
def plotScores(scores, paramGrid, scoreLabel=None, vrange=None)
```


Makes a plot representing how the scores vary over the parameter grid Automatically decides whether to use a simple line plot (varying over one parameter) or a heatmap (varying over two parameters).

**Parameters**:

`scores`: A list of scores, estimated using scoreModels

`paramGrid`: The parameter grid specified when fitting the models using fitModels

`scoreLabel`: The specified label (dependent on scoring metric used), e.g. 'AUC'

`vrange`: The visible range over which to display the scores

**returns**:

returns a plot



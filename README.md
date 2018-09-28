# parfit
A package for parallelizing the fit and flexibly scoring of sklearn machine learning models, with visualization routines.

Installation:
```
$pip install parfit # first time installation
$pip install -U parfit # upgrade to latest version
``` 

CURRENT VERSION == 0.200

and then import into your code using:
```
from parfit import bestFit # Necessary if you wish to use bestFit

# Necessary if you wish to run each step sequentially
from parfit.fit import *
from parfit.score import *
from parfit.plot import *
from parfit.crossval import *
```

 Once imported, you can use bestFit() or other functions freely.

## Easy to use
```
grid = {
    'min_samples_leaf': [1, 5, 10, 15, 20, 25],
    'max_features': ['sqrt', 'log2', 0.5, 0.6, 0.7],
    'n_estimators': [60],
    'n_jobs': [-1],
    'random_state': [42]
}
paramGrid = ParameterGrid(grid)

best_model, best_score, all_models, all_scores = bestFit(RandomForestClassifier(), paramGrid,
                                                    X_train, y_train, X_val, y_val, # nfolds=5 [optional, instead of validation set]
                                                    metric=roc_auc_score, greater_is_better=True, 
                                                    scoreLabel='AUC')

print(best_model, best_score)
```
```
{max_features': 'sqrt', 'min_samples_leaf': 1, 'n_estimators': 60, 'n_jobs': -1, 'random_state': 42}
0.9627794057231478
```

## Interpretable Visualizations
![Alt text](/assets/scoring_grid_2D.png?raw=true)

## Notes
1. You can either use **bestFit()** to automate the steps of the process, and optionally plot the scores over the parameter grid, OR you can do each step in order: 

> `fitModels()` -> `scoreModels()` -> `plotScores()` -> `getBestModel()` -> `getBestScore()`

or

> `crossvalModels()` -> `plotScores()` -> `getBestModel()` -> `getBestScore()`

2. Be sure to specify ALL parameters in the ParameterGrid, even the ones you are not searching over.

3. For example usage, see parfit_ex.ipynb. Each function is well-documented in the .py file. In Jupyter Notebooks, you can see the docs by pressing Shift+Tab(x3). Also, check out the complete documentation [here](docs/documentation.md) along with the [changelog](docs/changelog.md).

4. This package is designed for use with sklearn machine learning models, but in theory will work with any model that has a .fit(X,y) function. Furthermore, the sklearn scoring metrics are typically used, but any function that reads in two vectors and returns a score will work.

5. The plotScores() function will only work for up to a 3D parameterGrid object. That is, you can only view the scores of a grid varying over 1-3 parameters. Other parameters which do not vary can still be set, and you can still train and scores models over a higher dimensional grid.


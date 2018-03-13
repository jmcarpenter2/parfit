# Changelog

## Version 0.18

### Major functionality addition:
Parfit now allows for cross-validation. For usage, see documentation [here](docs/documentation.md).


## Version 0.17

### Major functionality change: 
Parfit now requires instantiated models, e.g. LogisticRegression() instead of the function name e.g. LogisticRegression. This is to improve flexibility of using parfit.

### Parameter name changes:
Parfit changed the following parameter names to increase ease of use and understanding.

`bestScore` -> `greater_is_better` now uses True/False just like sklearn's parameter scoring routines for determing which model is optimal.

`predictType` -> `predict_proba` now uses True/False to determine what type of prediction we wish to utilize. Probability prediction is default.

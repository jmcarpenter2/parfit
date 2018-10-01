# Changelog

## Version 0.21
Switched from using KFold to StratifiedKFold for cross-validation because we want to generally pay attention to class imbalances.

## Version 0.20

### MAJOR BUG FIX:
Due to shared memory across parallel jobs, cross-validation performance was consistently inflated. The latest bug fix resolves this issue by creating a new model for each fitted model in the cross-validation schema.

## Version 0.19

### Aesthetic improvement:
Parfit now uses YlOrRd cmap by default because there is [research providing evidence](https://cdn.mprog.nl/dataviz/excerpts/w4/Borland_Rainbow_Color_Map.pdf) that rainbow color maps can be harmful and difficult to interpret.
Parfit additionally allows the user to specify the matplotlib colormap of their choosing. Please see the [matplotlib colormap reference](https://matplotlib.org/examples/color/colormaps_reference.html) for options.
    If you wish to still use the original colormap, specify `cmap='jet'` as a parameter when calling `bestFit` or `plotScores`.
    
Additionally, by default plotScores now always makes the "better" values the top end of the original scale and the "worse" values the bottom end of the original scale by reversing the colorscale based on the value of `greater_is_better`. 
If you wish to flip the colorscale, add '_r' to the end of your cmap name. For example, `cmap='YlOrRd_r'` will flip the default scale such that yellow values are "better" and red values are "worse".


## Version 0.18

### Major functionality addition:
Parfit now allows for cross-validation. For usage, see documentation [here](documentation.md).


## Version 0.17

### Major functionality change: 
Parfit now requires instantiated models, e.g. LogisticRegression() instead of the function name e.g. LogisticRegression. This is to improve flexibility of using parfit.

### Parameter name changes:
Parfit changed the following parameter names to increase ease of use and understanding.

`bestScore` -> `greater_is_better` now uses True/False just like sklearn's parameter scoring routines for determing which model is optimal.

`predictType` -> `predict_proba` now uses True/False to determine what type of prediction we wish to utilize. Probability prediction is default.

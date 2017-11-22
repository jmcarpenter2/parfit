# parFit
A package for parallelizing the fit and flexibly scoring of sklearn machine learning models, with optional plotting routines. Edit

## Notes
You can either use **bestFit()** to automate the steps of the process, and optionally plot the scores over the parameter grid, OR you can do each step in order [*fitModels()* -> *scoreModels()* -> *plotScores()* -> *getBestModel()*]
Be sure to specify ALL parameters in the ParameterGrid, even the ones you are not searching over

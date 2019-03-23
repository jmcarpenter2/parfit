import dask.delayed
import numpy as np
import logging
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score


__all__ = ["Parfit"]
logger = logging.getLogger(__name__)


class Parfit:
    def __init__(
        self,
        model,
        param_grid,
        X_train,
        y_train,
        X_val=None,
        y_val=None,
        nfolds=5,
        metric=roc_auc_score,
        greater_is_better=True,
        predict_proba=True,
    ):
        self._model = model
        self._param_grid = param_grid
        self._x_trn = X_train
        self._y_trn = y_train
        self._x_val = X_val
        self._y_val = y_val
        self._nfolds = nfolds
        self._metric = metric
        self._greater_is_better = greater_is_better
        self._predict_proba = predict_proba
        self.fitted_models = {}
        self.scores = {}

    @staticmethod
    @dask.delayed
    def _fit(model, X, y, params):
        model.set_params(**params)
        return model.fit(X, y)

    def fit_models(self):
        fitted_models = []
        for params in list(self._param_grid):
            fitted_models.append(self._fit(self._model, self._x_trn, self._y_trn, params))

        self.fitted_models = {
            str(params): fitted_model
            for params, fitted_model in zip(list(self._param_grid), dask.compute(*fitted_models, scheduler="processes"))
        }

    def _crossval(self, model, X, y, params, nfolds, metric=roc_auc_score, predict_proba=True):
        cv = StratifiedKFold(n_splits=nfolds, shuffle=True, random_state=params.get("random_state"))
        train_indices, test_indices = zip(*cv.split(X, y))
        fitted_models = [
            self._fit(model, np.asarray(X)[train_index], np.asarray(y)[train_index], params)
            for train_index in train_indices
        ]
        scores = [
            self._score(fitted_model, np.asarray(X)[test_index], np.asarray(y)[test_index], metric, predict_proba)
            for fitted_model, test_index in zip(fitted_models, test_indices)
        ]
        return np.mean(scores)

    def crossval_models(self):
        scores = []
        for params in list(self._param_grid):
            scores.append(
                self._crossval(
                    self._model, self._x_trn, self._y_trn, params, self._nfolds, self._metric, self._predict_proba
                )
            )

        self.scores = {
            str(params): score
            for params, score in zip(list(self._param_grid), dask.compute(*scores, scheduler="processes"))
        }

    @staticmethod
    @dask.delayed
    def _score(model, X, y, metric, predict_proba):
        if predict_proba:
            if "predict_proba" in dir(model):
                return metric(y, model.predict_proba(X)[:, 1])
            else:
                logger.warning("This model/metric cannot use predict_proba. Using predict for scoring instead.")
                return metric(y, model.predict(X))
        else:
            return metric(y, model.predict(X))

    def score_models(self):
        scores = []
        for params, model in self.fitted_models.items():
            scores.append(self._score(model, self._x_val, self._y_val, self._metric, self._predict_proba))
        self.scores = {
            str(params): score
            for params, score in zip(list(self._param_grid), dask.compute(*scores, scheduler="processes"))
        }

    def get_best_model(self):
        models_sorted_by_score = [
            self.fitted_models[params]
            for (params, scores) in sorted(self.scores.items(), key=lambda kv: kv[1], reverse=self._greater_is_better)
        ]
        return models_sorted_by_score[0]

    def get_best_params(self):
        sorted_params = [
            params
            for (params, scores) in sorted(self.scores.items(), key=lambda kv: kv[1], reverse=self._greater_is_better)
        ]
        return sorted_params[0]

    def get_best_score(self):
        sorted_scores = [
            scores
            for (params, scores) in sorted(self.scores.items(), key=lambda kv: kv[1], reverse=self._greater_is_better)
        ]
        return sorted_scores[0]

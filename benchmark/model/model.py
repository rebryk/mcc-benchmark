from abc import ABC
from abc import abstractmethod

from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.multiclass import BaseEstimator
from sklearn.multiclass import ClassifierMixin
from sklearn.multiclass import OneVsRestClassifier
from xgboost import XGBClassifier


class Model(ABC, BaseEstimator, ClassifierMixin):
    """Class represents model interface."""

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass


class OneVsRestModel(Model):
    """Class implement One vs. Rest scheme."""

    _estimator_class = None

    def __init__(self, **kwargs):
        self._model = None
        self._estimator = self._estimator_class(**kwargs)

    def __setattr__(self, key, value):
        if key.startswith('_'):
            super().__setattr__(key, value)
        else:
            setattr(self._estimator, key, value)

    def get_params(self, deep=True):
        return self._estimator.get_params(deep=deep)

    def fit(self, X, y):
        self._model = OneVsRestClassifier(self._estimator)
        self._model.fit(X, y)

    def predict(self, X):
        return self._model.predict(X)


class OneVsRestXGBClassifier(OneVsRestModel):
    _estimator_class = XGBClassifier


class OneVsRestLGBMClassifier(OneVsRestModel):
    _estimator_class = LGBMClassifier


class OneVsRestGradientBoostingClassifier(OneVsRestModel):
    _estimator_class = GradientBoostingClassifier


class OneVsRestCatBoostClassifier(OneVsRestModel):
    _estimator_class = CatBoostClassifier

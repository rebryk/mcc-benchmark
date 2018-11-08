from abc import ABC
from abc import abstractmethod

from sklearn.multiclass import BaseEstimator
from sklearn.multiclass import ClassifierMixin


class Model(ABC, BaseEstimator, ClassifierMixin):
    """Class represents model interface."""

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass

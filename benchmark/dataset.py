import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generator
from typing import Union

import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split

from .utils import download_file


class Dataset(ABC):
    """Class represents dataset interface."""

    @abstractmethod
    def load(self,
             data_folder: Union[str, Path],
             n_splits: int = 1,
             test_size: Union[int, float, None] = None) -> Generator:
        """Load the dataset.

        :param data_folder: A path to the folder with datasets.
        :param n_splits: Number of splits to generate.
        :param test_size: Represents the proportion of the dataset to include in the test split.
        :return: X_train, X_test, y_train, y_test
        """
        pass


class LibsvmDataset(Dataset):
    """
    Class represents libsvm datasets.
    Visit https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass.html for more information.
    """

    SOURCE = 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/'

    def __init__(self, train: str, test: str = None, n_classes: int = None):
        self.train = train
        self.test = test
        self.n_classes = n_classes
        self.logger = logging.getLogger(self.__class__.__name__)

    @staticmethod
    def _select_classes(y: np.ndarray, n_classes: int):
        np.random.seed(0)
        classes = np.unique(y)

        if n_classes is None:
            return classes

        if len(classes) < n_classes:
            raise RuntimeError(f'Dataset contains just {len(classes)} unique labels!')

        return np.random.choice(classes, size=n_classes, replace=False)

    @staticmethod
    def _reduce_classes(X: np.ndarray, y: np.ndarray, classes: np.ndarray):
        mask = np.isin(y, classes)
        X, y = X[mask], y[mask]
        mapping = {y: i for i, y in enumerate(classes)}
        y = np.array([mapping[it] for it in y])
        return X, y

    def load(self, data_folder, n_splits: int = 1, test_size: float = None):
        if n_splits < 1:
            raise ValueError('n_splits should be positive!')

        if not isinstance(data_folder, Path):
            data_folder = Path(data_folder)

        if not self.train:
            raise ValueError('Train argument is not specified!')

        if not data_folder.exists():
            data_folder.mkdir()

        train = data_folder / self.train
        test = data_folder / self.test if self.test else None

        if not train.exists():
            url = self.SOURCE + self.train
            self.logger.info(f'Downloading train data from {url}')
            download_file(url, train)

        if test and not test.exists():
            url = self.SOURCE + self.test
            self.logger.info(f'Downloading test data from {url}')
            download_file(url, test)

        X_train, y_train = load_svmlight_file(str(train))
        X_train, y_train = X_train.toarray(), y_train.astype(np.int32)

        classes = self._select_classes(y_train, self.n_classes)
        X_train, y_train = self._reduce_classes(X_train, y_train, classes)

        if test:
            X_test, y_test = load_svmlight_file(str(test))
            X_test, y_test = X_test.toarray(), y_test.astype(np.int32)
            X_test, y_test = self._reduce_classes(X_test, y_test, classes)
            X_train = np.vstack((X_train, X_test))
            y_train = np.concatenate((y_train, y_test))

        for it in range(n_splits):
            yield train_test_split(X_train, y_train, test_size=test_size, random_state=it, stratify=y_train)

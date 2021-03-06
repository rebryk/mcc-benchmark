import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generator
from typing import Union, Tuple

import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
import pandas as pd

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

    def __init__(self, train: str, test: str = None, n_classes: int = None, imbalanced: bool = False):
        self.train = train
        self.test = test
        self.n_classes = n_classes
        self.imbalanced = imbalanced
        self.logger = logging.getLogger(self.__class__.__name__)

    @staticmethod
    def _select_classes(y: np.ndarray, n_classes: int) -> np.ndarray:
        np.random.seed(0)
        classes = np.unique(y)

        if n_classes is None:
            return classes

        if len(classes) < n_classes:
            raise RuntimeError(f'Dataset contains just {len(classes)} unique labels!')

        return np.random.choice(classes, size=n_classes, replace=False)

    @staticmethod
    def _reduce_classes(X: np.ndarray, y: np.ndarray, classes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mask = np.isin(y, classes)
        X, y = X[mask], y[mask]
        mapping = {y: i for i, y in enumerate(classes)}
        y = np.array([mapping[it] for it in y])
        return X, y

    @staticmethod
    def _imbalance(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        np.random.seed(0)

        result = []
        indices = np.arange(len(y))
        classes, cnt = np.unique(y, return_counts=True)

        for clazz in classes:
            fraction = 0.5 # np.random.random()
            count = max(int(cnt[clazz] * fraction), 2)
            result.append(np.random.choice(indices[y == clazz], size=count, replace=False))

        indices = np.concatenate(result)
        return X[indices], y[indices]

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

        if self.imbalanced:
            X_train, y_train = self._imbalance(X_train, y_train)

        classes = self._select_classes(y_train, self.n_classes)
        X_train, y_train = self._reduce_classes(X_train, y_train, classes)

        if test:
            X_test, y_test = load_svmlight_file(str(test))
            X_test, y_test = X_test.toarray(), y_test.astype(np.int32)
            if X_test.shape[1] < X_train.shape[1]:
                X_train = X_train[:,:X_test.shape[1]]
            if X_train.shape[1] < X_test.shape[1]:
                X_test = X_test[:,:X_train.shape[1]]
            X_test, y_test = self._reduce_classes(X_test, y_test, classes)
            X_train = np.vstack((X_train, X_test))
            y_train = np.concatenate((y_train, y_test))

        for it in range(n_splits):
            yield train_test_split(X_train, y_train, test_size=test_size, random_state=it, stratify=y_train)


class CSVDataset(Dataset):
    def __init__(self, class_name, source, train, test, sep, skiprows, compression, label_column):
        self.source = source
        self.train = train
        self.test = test
        self.sep = sep
        self.skiprows = skiprows
        self.compression = compression
        self.label_column = label_column
        self.logger = logging.getLogger(class_name)

    def _load_csv(self, data_folder, dataset):
        dataset_path = data_folder / dataset
        if not dataset_path.exists():
            url = self.source + dataset
            self.logger.info(f'Downloading dataset from {url}')
            download_file(url, dataset_path)

        df = pd.read_csv(dataset_path, header=None, index_col=None, sep=self.sep, skiprows=self.skiprows, compression=self.compression)
        X, y = np.array(df.drop(df.columns[[self.label_column]], axis=1)), np.array(df.iloc[:,self.label_column])

        classes = np.unique(y)
        for c in classes:
            cnt = 0
            for i in range(len(y)):
                if y[i] == c:
                    cnt += 1
            if cnt > 1:
                continue

            for i in range(len(y)):
                if y[i] == c:
                    X = np.delete(X, i, axis=0)
                    y = np.delete(y, i, axis=0)
                    break

        return X, y

    def load(self, data_folder, n_splits: int = 1, test_size: float = None):
        if n_splits < 1:
            raise ValueError('n_splits should be positive!')

        if not isinstance(data_folder, Path):
            data_folder = Path(data_folder)

        if not data_folder.exists():
            data_folder.mkdir()

        X_train, y_train = self._load_csv(data_folder, self.train)

        if self.test is not None:
            X_test, y_test = self._load_csv(data_folder, self.test)
            X_train = np.vstack((X_train, X_test))
            y_train = np.concatenate((y_train, y_test))

        for i in range(X_train.shape[1]):
            if isinstance(X_train[0, i], str):
                values = np.unique(X_train[:, i])
                mapping = {y : j for j, y in enumerate(values)}
                X_train[:, i] = np.array([mapping[x] for x in X_train[:, i]])

        X_train = X_train.astype(np.float32)

        classes = np.unique(y_train)
        mapping = {y : i for i, y in enumerate(classes)}
        y_train = np.array([mapping[y] for y in y_train], dtype=np.int32)

        for it in range(n_splits):
            yield train_test_split(X_train, y_train, test_size=test_size, random_state=it, stratify=y_train)


class ImageSegmentation(CSVDataset):
    """
    Class represents Image Segmentation Dataset.
    Visit http://archive.ics.uci.edu/ml/datasets/image+segmentation for more information
    """

    SOURCE = 'http://archive.ics.uci.edu/ml/machine-learning-databases/image/'
    TRAIN = 'segmentation.data'
    TEST = 'segmentation.test'

    def __init__(self):
        super().__init__(
            class_name=self.__class__.__name__,
            source=self.SOURCE,
            train=self.TRAIN,
            test=self.TEST,
            sep=',',
            skiprows=6,
            compression=None,
            label_column=0
        )


class Covertype(CSVDataset):
    """
    Class represents Covertype Dataset.
    Visit https://archive.ics.uci.edu/ml/datasets/Covertype for more information
    """

    SOURCE = 'https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/'
    TRAIN = 'covtype.data.gz'

    def __init__(self):
        super().__init__(
            class_name=self.__class__.__name__,
            source=self.SOURCE,
            train=self.TRAIN,
            test=None,
            sep=',',
            skiprows=0,
            compression='gzip',
            label_column=-1
        )


class WinequalityWhite(CSVDataset):
    """
    Class represents Wine Quality (White) Dataset.
    Visit https://archive.ics.uci.edu/ml/datasets/wine+quality for more information
    """

    SOURCE = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/'
    TRAIN = 'winequality-white.csv'

    def __init__(self):
        super().__init__(
            class_name=self.__class__.__name__,
            source=self.SOURCE,
            train=self.TRAIN,
            test=None,
            sep=';',
            skiprows=1,
            compression=None,
            label_column=-1
        )


class Abalone(CSVDataset):
    """
    Class represents Abalone Dataset.
    Visit https://archive.ics.uci.edu/ml/datasets/Abalone for more information
    """

    SOURCE = 'https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/'
    TRAIN = 'abalone.data'

    def __init__(self):
        super().__init__(
            class_name=self.__class__.__name__,
            source=self.SOURCE,
            train=self.TRAIN,
            test=None,
            sep=',',
            skiprows=0,
            compression=None,
            label_column=-1
        )


class CSVDataset(Dataset):
    def __init__(self, class_name, source, train, test, sep, skiprows, compression, label_column):
        self.source = source
        self.train = train
        self.test = test
        self.sep = sep
        self.skiprows = skiprows
        self.compression = compression
        self.label_column = label_column
        self.logger = logging.getLogger(class_name)

    def _load_csv(self, data_folder, dataset):
        dataset_path = data_folder / dataset
        if not dataset_path.exists():
            url = self.source + dataset
            self.logger.info(f'Downloading dataset from {url}')
            download_file(url, dataset_path)

        df = pd.read_csv(dataset_path, header=None, index_col=None, sep=self.sep, skiprows=self.skiprows,
                         compression=self.compression)
        X, y = np.array(df.drop(df.columns[[self.label_column]], axis=1)), np.array(df.iloc[:, self.label_column])

        classes = np.unique(y)
        for c in classes:
            cnt = 0
            for i in range(len(y)):
                if y[i] == c:
                    cnt += 1
            if cnt > 1:
                continue

            for i in range(len(y)):
                if y[i] == c:
                    X = np.delete(X, i, axis=0)
                    y = np.delete(y, i, axis=0)
                    break

        return X, y

    def load(self, data_folder, n_splits: int = 1, test_size: float = None):
        if n_splits < 1:
            raise ValueError('n_splits should be positive!')

        if not isinstance(data_folder, Path):
            data_folder = Path(data_folder)

        if not data_folder.exists():
            data_folder.mkdir()

        X_train, y_train = self._load_csv(data_folder, self.train)

        if self.test is not None:
            X_test, y_test = self._load_csv(data_folder, self.test)
            X_train = np.vstack((X_train, X_test))
            y_train = np.concatenate((y_train, y_test))

        for i in range(X_train.shape[1]):
            if isinstance(X_train[0, i], str):
                values = np.unique(X_train[:, i])
                mapping = {y: j for j, y in enumerate(values)}
                X_train[:, i] = np.array([mapping[x] for x in X_train[:, i]])

        X_train = X_train.astype(np.float32)

        classes = np.unique(y_train)
        mapping = {y: i for i, y in enumerate(classes)}
        y_train = np.array([mapping[y] for y in y_train], dtype=np.int32)

        for it in range(n_splits):
            yield train_test_split(X_train, y_train, test_size=test_size, random_state=it, stratify=y_train)


class IMAT(Dataset):
    SOURCE = 'http://eranik.me/~education/machine-learning/datasets/'

    def __init__(self, train: str):
        self.train = train
        self.logger = logging.getLogger(self.__class__.__name__)

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
        if not train.exists():
            url = self.SOURCE + self.train
            self.logger.info(f'Downloading dataset from {url}')
            download_file(url, train)

        if not train.exists():
            raise ValueError('Train dataset does not exist')

        X_train, y_train = load_svmlight_file(str(train))
        _, y_train = np.unique(np.round(y_train), return_inverse=True)
        X_train, y_train = X_train.toarray(), y_train.astype(np.int32)

        assert len(np.unique(y_train)) == 5

        for it in range(n_splits):
            yield train_test_split(X_train, y_train, test_size=test_size, random_state=it, stratify=y_train)
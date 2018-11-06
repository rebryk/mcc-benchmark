import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union

from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split

from .utils import download_file


class Dataset(ABC):
    """Class represents dataset interface."""

    @abstractmethod
    def load(self, data_folder: Union[str, Path], test_size: Union[int, float, None] = None):
        """Load the dataset.

        :param data_folder: A path to the folder with datasets.
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

    def __init__(self, train: str, test: str = None):
        self.train = train
        self.test = test
        self.logger = logging.getLogger(self.__class__.__name__)

    def load(self, data_folder, test_size=None):
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

        if test:
            X_test, y_test = load_svmlight_file(str(test))
        else:
            X_train, X_test, y_train, y_test = train_test_split(X_train,
                                                                y_train,
                                                                test_size=test_size,
                                                                random_state=0,
                                                                stratify=y_train)

        return X_train.toarray(), X_test.toarray(), y_train, y_test

import os
import subprocess
from pathlib import Path

import numpy as np

from benchmark.model.model import Model


class FMCBoosting(Model):
    def __init__(self,
                 path: str = None,
                 n_iter: int = 2000,
                 lr: float = 5,
                 gamma: float = 100,
                 max_iter: int = 2000,
                 depth: int = 5,
                 n_bins: int = 32):
        """Factorized MultiClass Boosting

        :param path: A path to jmll jar file with FMCBoosting.
        :param n_iter: The number of weak learners.
        :param lr: Learning rate.
        :param gamma: Learning rate for StochasticALS.
        :param max_iter: Max iterations count for StochasticALS.
        :param depth: The maximum depth of the weak tree.
        :param n_bins: Bin factor.
        """
        self.path = path
        self.n_iter = n_iter
        self.lr = lr
        self.gamma = gamma
        self.max_iter = max_iter
        self.depth = depth
        self.n_bins = n_bins

        tmp_folder = Path().absolute() / 'tmp'

        if not tmp_folder.exists():
            tmp_folder.mkdir()

        self._train_path = tmp_folder / 'train.tsv'
        self._test_path = tmp_folder / 'test.tsv'
        self._train_pred_path = tmp_folder / 'train_pred.txt'
        self._test_pred_path = tmp_folder / 'test_pred.txt'
        self._model_path = tmp_folder / 'model.txt'

    def fit(self, X, y):
        """Fit the estimator."""
        if self.path is None:
            raise RuntimeError('Path to JMLL is not specified!')

        self._remove_files([self._train_path, self._train_pred_path, self._model_path])
        self._save_data_to_tsv(X, y, self._train_path)

        subprocess.run(['java', '-jar', self.path,
                        '--model', str(self._model_path),
                        '--n_iter', str(self.n_iter),
                        '--step', str(self.lr),
                        '--gamma', str(self.gamma),
                        '--max_iter', str(self.max_iter),
                        '--depth', str(self.depth),
                        '--n_bins', str(self.n_bins),
                        '--train', str(self._train_path),
                        '--train_pred', str(self._train_pred_path)], stdout=subprocess.DEVNULL)

        return self

    def predict(self, X):
        """Predict class for X."""
        if self.path is None:
            raise RuntimeError('Path to JMLL is not specified!')

        self._remove_files([self._test_path, self._test_pred_path])
        fake_y = np.zeros(len(X))
        self._save_data_to_tsv(X, fake_y, self._test_path)

        subprocess.run(['java', '-jar', self.path,
                        '--model', str(self._model_path),
                        '--test', str(self._test_path),
                        '--test_pred', str(self._test_pred_path)], stdout=subprocess.DEVNULL)

        return self._read_prediction(self._test_pred_path)

    @staticmethod
    def _remove_files(files: [Path]):
        """Remove the specified files."""
        for file in filter(lambda it: it.exists(), files):
            os.remove(file)

    @staticmethod
    def _read_prediction(path: Path) -> np.ndarray:
        """Read predicted classes from the file."""
        return np.loadtxt(str(path), dtype=np.int, delimiter=',')

    @staticmethod
    def _save_data_to_tsv(X, y, path: Path):
        """Save numpy array data to tsv format."""
        with path.open('w') as f:
            for features, target in zip(X, y):
                values = '\t'.join(map(str, features))
                f.write(f'{0}\t{int(target)}\turl\t{0}\t{values}\n')
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
                 beta: float = 0.0,
                 max_iter: int = 2000,
                 depth: int = 5,
                 n_bins: int = 32,
                 ensemble_size: int = 1,
                 is_gbdt: bool = False,
                 verbose: bool = False):
        """Factorized MultiClass Boosting

        :param path: A path to jmll jar file with FMCBoosting.
        :param n_iter: The number of weak learners.
        :param lr: Learning rate.
        :param gamma: Learning rate for StochasticALS.
        :param beta: L1 loss coefficient.
        :param max_iter: Max iterations count for StochasticALS.
        :param depth: The maximum depth of the weak tree.
        :param n_bins: Bin factor.
        :param ensemble_size: The size of weak ensemble.
        :param is_gbdt: Whether we should use Gradient Boosting Decision Trees instead of Random Forest.
        :param verbose: Verbose output.
        """
        self.path = path
        self.n_iter = n_iter
        self.lr = lr
        self.gamma = gamma
        self.beta = beta
        self.max_iter = max_iter
        self.depth = depth
        self.n_bins = n_bins
        self.ensemble_size = ensemble_size
        self.is_gbdt = is_gbdt
        self.verbose = verbose

        tmp_folder = Path().absolute() / 'tmp'

        if not tmp_folder.exists():
            tmp_folder.mkdir()

        self._train_path = tmp_folder / 'train.tsv'
        self._valid_path = tmp_folder / 'valid.tsv'
        self._test_path = tmp_folder / 'test.tsv'
        self._pred_path = tmp_folder / 'test_pred.txt'
        self._model_path = tmp_folder / 'model.txt'

    def fit(self, X, y, eval_set=None, early_stopping_rounds=None):
        """Fit the estimator."""
        if self.path is None:
            raise RuntimeError('Path to JMLL is not specified!')

        self._remove_files([self._train_path, self._valid_path, self._model_path])
        self._save_data_to_tsv(X, y, self._train_path)

        if eval_set:
            X_valid, y_valid = eval_set[0]
            self._save_data_to_tsv(X_valid, y_valid, self._valid_path)

        out = subprocess.DEVNULL if not self.verbose else None
        params = ['java', '-jar', self.path,
                  '--model', str(self._model_path),
                  '--train', str(self._train_path),
                  '--n_iter', str(self.n_iter),
                  '--step', str(self.lr),
                  '--gamma', str(self.gamma),
                  '--lambda', str(self.beta),
                  '--max_iter', str(self.max_iter),
                  '--depth', str(self.depth),
                  '--n_bins', str(self.n_bins),
                  '--ensemble_size', str(self.ensemble_size),
                  '--is_gbdt', str(self.is_gbdt)]

        if eval_set:
            params += ['--valid', str(self._valid_path)]

        if early_stopping_rounds:
            params += ['--early_stopping_rounds', str(early_stopping_rounds)]

        subprocess.run(params, stdout=out)
        return self

    def predict(self, X):
        """Predict class for X."""
        if self.path is None:
            raise RuntimeError('Path to JMLL is not specified!')

        self._remove_files([self._test_path, self._pred_path])
        fake_y = np.zeros(len(X))
        self._save_data_to_tsv(X, fake_y, self._test_path)

        out = subprocess.DEVNULL if not self.verbose else None
        subprocess.run(['java', '-Xmx16G', '-jar', self.path,
                        '--model', str(self._model_path),
                        '--test', str(self._test_path),
                        '--test_pred', str(self._pred_path)], stdout=out)

        return self._read_prediction(self._pred_path)

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

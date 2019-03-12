import inspect
import logging
import math
import sys
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from benchmark import get_selection_method, get_model_class, get_dataset
from benchmark.result import Result, RunResult
from benchmark.utils import Timer, parse_params, eval_params, AttributeDict, get_write_method


class Experiment:
    DEFAULT_DATA_FOLDER = Path('dataset')
    DEFAULT_LOG_FOLDER = Path('logs')

    def __init__(self,
                 model: str,
                 params: str,
                 dataset: str,
                 test_size: float,
                 valid_size: float = None,
                 n_runs: int = 1,
                 n_skip_runs: int = 0,
                 selection: str = None,
                 selection_params: str = None,
                 param_grid: str = None):
        """Create an experiment.

        :param model: model to evaluate.
        :param params: parameters of the model in json format.
        :param dataset: dataset name.
        :param test_size: represents the proportion of the dataset to include in the test split.
        :param valid_size: represents the proportion of the dataset to include in the valid split.
        :param n_runs: number of runs.
        :param n_skip_runs: number of runs to be skipped.
        :param selection: method of hyperparameter tuning.
        :param selection_params: parameters of the selection method in json format.
        :param param_grid: enables searching over any sequence of parameter settings.
        """
        self.model = model
        self.params = params
        self.dataset = dataset
        self.test_size = test_size
        self.valid_size = valid_size
        self.n_runs = n_runs
        self.n_skip_runs = n_skip_runs
        self.selection = selection
        self.selection_params = selection_params
        self.param_grid = param_grid
        self.runs = []
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.write = get_write_method(self.logger)
        self.logger.flush = lambda: None

    PRECISION = 4

    def _create_log_handler(self, file_name: str):
        """Create handler to write log to the file."""
        if not Experiment.DEFAULT_LOG_FOLDER.exists():
            Experiment.DEFAULT_LOG_FOLDER.mkdir()

        file_handler = logging.FileHandler(Experiment.DEFAULT_LOG_FOLDER / file_name)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', '%m/%d/%Y %I:%M:%S %p')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def _get_selection_params(self) -> dict:
        """Initialize selection method."""
        if self.selection is None:
            return {}

        selection_class = get_selection_method(self.selection)
        selection_params = parse_params(self.selection_params or '')

        if self.param_grid is None:
            raise RuntimeError('Parameters grid is not specified!')

        param_grid = eval_params(parse_params(self.param_grid))

        self.logger.info(f'Selection method: {self.selection}')
        self.logger.info(f'Selection parameters: {selection_params if selection_params else "default"}')
        self.logger.info(f'Parameters grid: {param_grid}')

        return AttributeDict({'selection_class': selection_class, 'param_grid': param_grid, 'params': selection_params})

    def _run(self, model_class, params, selection_params, X_train, X_test, y_train, y_test) -> RunResult:
        """Train and evaluate the given model."""
        result = RunResult()

        X_valid = None
        y_valid = None
        dataset_size = len(X_train) + len(X_test)
        valid_size = math.floor(self.valid_size * dataset_size) if self.valid_size else 0

        if valid_size:
            X_train, X_valid, y_train, y_valid = train_test_split(X_train,
                                                                  y_train,
                                                                  test_size=valid_size,
                                                                  random_state=0,
                                                                  stratify=y_train)

        fit_params = {}
        early_stopping_rounds = params.get('early_stopping_rounds', None)
        if early_stopping_rounds:
            if not valid_size:
                raise RuntimeError('Valid set is not specified!')

            del params['early_stopping_rounds']
            if inspect.signature(model_class.fit).parameters.get('early_stopping_rounds') is not None:
                fit_params['early_stopping_rounds'] = early_stopping_rounds

        if valid_size and inspect.signature(model_class.fit).parameters.get('eval_set') is not None:
            fit_params['eval_set'] = [(X_valid, y_valid)]

        if selection_params:
            self.logger.info('Searching the best parameters...')

            X = np.vstack((X_train, X_valid)) if valid_size else X_train
            y = np.concatenate((y_train, y_valid)) if valid_size else y_train

            if valid_size:
                indices = list(range(len(X)))
                selection_params.params['cv'] = [(indices[:len(X_train)], indices[len(X_train):])]

            selection = selection_params.selection_class(model_class(**params),
                                                         selection_params.param_grid,
                                                         **selection_params.params)

            with redirect_stdout(self.logger):
                with Timer('Searching time', self.logger) as timer:
                    selection.fit(X, y, **fit_params) 

            result.search_time = timer.total_seconds()
            self.logger.info('Grid scores on validate set:')

            means = selection.cv_results_['mean_test_score']
            stds = selection.cv_results_['std_test_score']
            for mean, std, curr_params in zip(means, stds, selection.cv_results_['params']):
                self.logger.info(f'Valid score: {mean:0.4f} (+/-{std * 2:0.03f}) for {curr_params}')

            self.logger.info(f'Best parameters: {selection.best_params_}')
            result.params = str({**params, **selection.best_params_})
            model = model_class(**{**params, **selection.best_params_})
        else:
            result.params = str(params)
            model = model_class(**params)

        self.logger.info('Training the model...')
        with redirect_stdout(self.logger):
            with Timer('Training time', self.logger) as timer:
                model.fit(X_train, y_train, **fit_params)
        result.train_time = timer.total_seconds()

        with Timer('Prediction time (train)') as timer:
            result.score_train = round(model.score(X_train, y_train), self.PRECISION)
        result.pred_time_train = timer.total_seconds()
        self.logger.info(f'Train score:\t{result.score_train:0.4f}')

        if valid_size:
            with Timer('Prediction time (valid)', self.logger) as timer:
                result.score_valid = round(model.score(X_valid, y_valid), self.PRECISION)
            result.pred_time_valid = timer.total_seconds()
            self.logger.info(f'Valid score:\t{result.score_valid:0.4f}')

        with Timer('Prediction time (test)', self.logger) as timer:
            result.score_test = round(model.score(X_test, y_test), self.PRECISION)
        result.pred_time_test = timer.total_seconds()
        self.logger.info(f'Test score:\t{result.score_test:0.4f}')

        return result

    def run(self):
        """Run an experiment with the specified parameters."""
        current_date = datetime.now().strftime('%Y.%m.%d %H.%M.%S')
        file_name = f'{self.dataset} {self.model} {current_date}'

        self._create_log_handler(f'{file_name}.log')

        model_class = get_model_class(self.model)
        params = parse_params(self.params)
        self.logger.info(f'Model: {self.model}')
        self.logger.info(f'Model parameters: {params if params else "default"}')

        selection_params = self._get_selection_params()

        self.logger.info(f'Loading {self.dataset} dataset...')
        dataset = get_dataset(self.dataset)
        dataset_generator = dataset.load(Experiment.DEFAULT_DATA_FOLDER,
                                         n_splits=self.n_runs + self.n_skip_runs,
                                         test_size=self.test_size)

        with Timer('Total time', self.logger) as timer:
            for run, (X_train, X_test, y_train, y_test) in enumerate(dataset_generator, 1):
                if run <= self.n_skip_runs:
                    continue

                if run == self.n_skip_runs + 1:
                    dataset_size = len(X_train) + len(X_test)
                    valid_size = math.floor(self.valid_size * dataset_size) if self.valid_size else 0

                    self.logger.info(f'Train size: {len(X_train) - valid_size}')
                    if valid_size:
                        self.logger.info(f'Valid size: {valid_size}')
                    self.logger.info(f'Test size: {len(X_test)}')

                self.logger.info(f'Run #{run}')
                self.runs.append(self._run(model_class, params, selection_params, X_train, X_test, y_train, y_test))

        result = Result(model=self.model,
                        params=self.params,
                        dataset=self.dataset,
                        test_size=self.test_size,
                        valid_size=self.valid_size,
                        selection=self.selection,
                        selection_params=self.selection_params,
                        param_grid=self.param_grid,
                        runs=self.runs,
                        total_time=timer.total_seconds())

        result.save(f'{file_name}.json')

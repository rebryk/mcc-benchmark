import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

from benchmark import get_dataset
from benchmark import get_model
from benchmark import get_selection_method

DEFAULT_DATA_FOLDER = 'dataset'
DEFAULT_LOG_FOLDER = Path('logs')

logger = logging.getLogger('eval')

parser = argparse.ArgumentParser('Evaluate the given model on the specified dataset.')
parser.add_argument('--model', type=str, required=True, help='model to evaluate')
parser.add_argument('--params', type=str, default='', help='parameters of the model in json format')
parser.add_argument('--dataset', type=str, required=True, help='dataset name')
parser.add_argument('--test_size', type=float, default=None,
                    help='represents the proportion of the dataset to include in the test split')
parser.add_argument('--selection', type=str, default=None, help='method of hyperparameter tuning')
parser.add_argument('--selection_params', type=str, default='',
                    help='parameters of the selection method in json format')
parser.add_argument('--param_grid', type=str, default=None,
                    help='enables searching over any sequence of parameter settings')


def parse_params(params: str) -> dict:
    """Parse the given parameters to dictionary."""
    return json.loads(params.replace('\'', '\"')) if params else dict()


def eval_params(params: dict) -> dict:
    """Evaluate values if they are not string."""
    return {key: eval(value) if isinstance(value, str) else value for key, value in params.items()}


def create_log_handler(file_name: str):
    """Create handler to write log to the file."""
    if not DEFAULT_LOG_FOLDER.exists():
        DEFAULT_LOG_FOLDER.mkdir()

    file_handler = logging.FileHandler(DEFAULT_LOG_FOLDER / file_name)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', '%m/%d/%Y %I:%M:%S %p')
    file_handler.setFormatter(formatter)
    return file_handler


if __name__ == '__main__':
    args = parser.parse_args()

    current_date = datetime.now().strftime('%Y.%m.%d %H.%M.%S')
    file_name = f'{args.model} {args.dataset} {current_date}.log'
    logger.addHandler(create_log_handler(file_name))

    model_class = get_model(args.model)
    logger.info(f'Model: {args.model}')

    params = parse_params(args.params)
    logger.info(f'Model parameters: {params if params else "default"}')

    model = model_class(**params)
    selection = None

    if args.selection is not None:
        selection_class = get_selection_method(args.selection)
        logger.info(f'Selection method: {args.selection}')

        selection_params = parse_params(args.selection_params)
        logger.info(f'Selection method parameters: {selection_params if selection_params else "default"}')

        if args.param_grid is None:
            raise RuntimeError('Parameters grid does not specified!')

        param_grid = parse_params(args.param_grid)
        logger.info(f'Parameters grid: {param_grid}')
        param_grid = eval_params(param_grid)
        selection = selection_class(model, param_grid, **selection_params)

    logger.info(f'Loading {args.dataset} dataset...')
    dataset = get_dataset(args.dataset)
    X_train, X_test, y_train, y_test = list(dataset.load(DEFAULT_DATA_FOLDER, test_size=args.test_size))[0]
    logger.info(f'Train size:\t{len(X_train)}')
    logger.info(f'Test size:\t{len(X_test)}')

    if selection is not None:
        logger.info('Searching the best parameters...')
        searching_start_time = datetime.now()
        selection.fit(X_train, y_train)
        searching_time = datetime.now() - searching_start_time

        logger.info(f'Best parameters: {selection.best_params_}')
        logger.info('Grid scores on validate set:')
        means = selection.cv_results_['mean_test_score']
        stds = selection.cv_results_['std_test_score']
        for mean, std, curr_params in zip(means, stds, selection.cv_results_['params']):
            logger.info(f'Valid score: {mean:0.3f} (+/-{std * 2:0.03f}) for {curr_params}')
        logger.info(f'Searching time: {searching_time} ({searching_time.total_seconds():.0f} seconds)')
        model = model_class(**{**params, **selection.best_params_})

    logger.info('Training the model...')
    training_start_time = datetime.now()
    model.fit(X_train, y_train)
    training_time = datetime.now() - training_start_time
    logger.info(f'Training time: {training_time} ({training_time.total_seconds():.0f} seconds)')

    logger.info(f'Train score:\t{model.score(X_train, y_train):0.3f}')
    logger.info(f'Test score:\t{model.score(X_test, y_test):0.3f}')

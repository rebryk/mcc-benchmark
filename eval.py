import argparse
import json
import logging
from datetime import datetime
from typing import Union

from benchmark import get_model
from benchmark import load_dataset

logger = logging.getLogger('eval')

parser = argparse.ArgumentParser('Evaluate the given model on the specified dataset.')
parser.add_argument('--model', type=str, required=True, help='model to evaluate')
parser.add_argument('--params', type=str, default='', help='params of the model in json format')
parser.add_argument('--dataset', type=str, required=True, help='dataset name')
parser.add_argument('--test_size', type=Union[int, float], default=None,
                    help='represents the proportion of the dataset to include in the train split')

if __name__ == '__main__':
    args = parser.parse_args()

    logger.info(f'Use {args.model} model')
    model_class = get_model(args.model)

    params = json.loads(args.params.replace('\'', '\"'))
    params_str = ', '.join(f'{key}={value}' for key, value in params.items())
    logger.info(f'Model parameters: {params_str if params else "default"}')

    model = model_class(**params)

    logger.info(f'Loading {args.dataset} dataset...')
    X_train, X_test, y_train, y_test = load_dataset(args.dataset, test_size=args.test_size)
    logger.info(f'Train size:\t{len(X_train)}')
    logger.info(f'Test size:\t{len(X_test)}')

    logger.info('Training the model...')
    training_start_time = datetime.now()
    model.fit(X_train, y_train)
    training_time = datetime.now() - training_start_time
    logger.info(f'Training time: {training_time} ({training_time.total_seconds():.0f} seconds)')

    logger.info(f'Train score:\t{model.score(X_train, y_train):0.3f}')
    logger.info(f'Test score:\t{model.score(X_test, y_test):0.3f}')

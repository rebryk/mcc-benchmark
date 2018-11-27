import argparse

from benchmark.experiment import Experiment


def get_config():
    parser = argparse.ArgumentParser('Evaluate the given model on the specified dataset.')
    parser.add_argument('--model', type=str, required=True, help='model to evaluate')
    parser.add_argument('--params', type=str, default='', help='parameters of the model in json format')
    parser.add_argument('--dataset', type=str, required=True, help='dataset name')
    parser.add_argument('--test_size', type=float, default=0.25,
                        help='represents the proportion of the dataset to include in the test split')
    parser.add_argument('--valid_size', type=float, default=None,
                        help='represents the proportion of the dataset to include in the valid split')
    parser.add_argument('--n_runs', type=int, default=1, help='number of runs')
    parser.add_argument('--selection', type=str, default=None, help='method of hyperparameter tuning')
    parser.add_argument('--selection_params', type=str, default=None,
                        help='parameters of the selection method in json format')
    parser.add_argument('--param_grid', type=str, default=None,
                        help='enables searching over any sequence of parameter settings')
    return parser.parse_args()


if __name__ == '__main__':
    config = get_config()
    experiment = Experiment(model=config.model,
                            params=config.params,
                            dataset=config.dataset,
                            test_size=config.test_size,
                            valid_size=config.valid_size,
                            n_runs=config.n_runs,
                            selection=config.selection,
                            selection_params=config.selection_params,
                            param_grid=config.param_grid)
    experiment.run()

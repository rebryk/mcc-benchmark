import json
from pathlib import Path


class RunResult:
    """Class represents run results."""

    def __init__(self, *,
                 params: dict = None,
                 search_time: int = None,
                 train_time: int = None,
                 pred_time_train: int = None,
                 pred_time_valid: int = None,
                 pred_time_test: int = None,
                 score_train: float = None,
                 score_valid: float = None,
                 score_test: float = None):
        self.params = params
        self.search_time = search_time
        self.train_time = train_time
        self.pred_time_train = pred_time_train
        self.pred_time_valid = pred_time_valid
        self.pred_time_test = pred_time_test
        self.score_train = score_train
        self.score_valid = score_valid
        self.score_test = score_test


class Result:
    """Class represents experiment results."""

    RESULTS_FOLDER = Path('results')

    def __init__(self, *,
                 model: str = None,
                 params: str = None,
                 dataset: str = None,
                 test_size: float = None,
                 valid_size: float = None,
                 selection: str = None,
                 selection_params: str = None,
                 param_grid: str = None,
                 runs: list = None,
                 total_time: int = None):
        self.model = model
        self.params = params
        self.dataset = dataset
        self.test_size = test_size
        self.valid_size = valid_size
        self.selection = selection
        self.selection_params = selection_params
        self.param_grid = param_grid
        self.runs = runs
        self.total_time = total_time

    def __repr__(self):
        dict = vars(self)
        dict['runs'] = [vars(it) for it in self.runs]
        return json.dumps(dict, indent=4)

    def save(self, file_name: str):
        """Save the results of the experiment."""
        if not self.RESULTS_FOLDER.exists():
            self.RESULTS_FOLDER.mkdir()

        file = self.RESULTS_FOLDER / file_name
        with file.open('w') as f:
            f.write(self.__repr__())
